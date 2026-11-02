"""Herfstfenologie van het loofbos in Spanderswoud uit Sentinel-2 NDVI (s2/cube.npz van s2_extract.py).

Per jaar wordt tussen het zomerniveau (jul-aug) en de winterbasis (meerjarig, half dec - feb) een logistische
afname gefit. Uitkomsten: t50 (midden van de groenafname) en t15 (15% van de amplitude over; vergelijkbaar met de
HR-VPP 'end of season'-definitie), met bootstrap-onzekerheid. Naaldbos dient als controle op zonnestand-effecten.

Gebruik:  uv run --no-project --with rasterio,numpy,scipy,matplotlib,pandas python s2_phenology.py
"""
import json
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from scipy.optimize import curve_fit

HERE = Path(__file__).parent
S2 = HERE / "s2"
MIN_CLEAR = 0.8  # minimaal aandeel heldere loofbospixels per opname
BLUE_MAX = 0.06  # nevel/dunne bewolking die SCL mist: blauwe reflectie boven bos is normaal ~0.02-0.04
TARGET_DOY = 301  # 28 oktober (geen schrikkeljaar; in schrikkeljaren 27 okt) -- datum van de heldere opname in 2026
T15_OFFSET = np.log(0.85 / 0.15)  # afstand t50 -> t15 in eenheden s
CUBES = ["cube.npz", "cube_l2a_2017_2018.npz", "cube_l2a_2022.npz"]  # C1 eerst; oude collectie vult gaten


def load():
    """Voeg de kubussen samen; bij dubbele datums wint de eerste (C1 gaat voor de oude collectie)."""
    grid = json.loads((S2 / "cube_grid.json").read_text())
    parts = [np.load(S2 / f) for f in CUBES if (S2 / f).exists()]
    dates = np.concatenate([d["dates"] for d in parts])
    _, first = np.unique(dates, return_index=True)
    stack = lambda k: np.concatenate([d[k] for d in parts])[first]
    ndvi = stack("ndvi").astype(np.float32)
    clear = np.isin(stack("scl"), (4, 5)) & (stack("blue").astype(np.float32) < BLUE_MAX) & np.isfinite(ndvi)
    return pd.to_datetime(dates[first]), ndvi, clear, grid


def site_mask(grid):
    b = grid["bounds"]
    transform = from_origin(b[0], b[3], 10, 10)
    gj = json.loads((HERE / "spanderswoud_osm.geojson").read_text())
    t = Transformer.from_crs(4326, 32631, always_xy=True)
    ring = [t.transform(lon, lat) for lon, lat in gj["features"][0]["geometry"]["coordinates"][0]]
    geom = {"type": "Polygon", "coordinates": [ring]}
    return geometry_mask([geom], out_shape=tuple(grid["shape"]), transform=transform, invert=True), transform, ring


def pixel_climatology(dates, ndvi, clear):
    doy = dates.day_of_year.values
    summer = (doy >= 166) & (doy <= 243)  # half juni - augustus
    winter = (doy >= 349) | (doy <= 59)  # half december - februari
    def med(sel):
        a = np.where(clear[sel], ndvi[sel], np.nan)
        return np.nanmedian(a, axis=0)
    return med(summer), med(winter)


def site_series(dates, ndvi, clear, mask):
    rows = []
    n = mask.sum()
    for i, t in enumerate(dates):
        c = clear[i] & mask
        frac = c.sum() / n
        rows.append({"date": t, "clear_frac": frac, "ndvi": np.median(ndvi[i][c]) if frac >= MIN_CLEAR else np.nan})
    return pd.DataFrame(rows)


def logistic(t, t50, s, top, base):
    return base + (top - base) / (1 + np.exp((t - t50) / s))


def fit_year(df, year, base, rng, nboot=300, s_fixed=None, col="ndvi"):
    """Fit t50 en s voor de groenafname van `year`; top = mediaan half jun-aug van dat jaar, base = meerjarige winterbasis.
    Met `s_fixed` ligt de steilheid vast en wordt alleen t50 geschat."""
    t0 = pd.Timestamp(year, 1, 1)
    sel = df[(df.date >= pd.Timestamp(year, 6, 15)) & (df.date <= pd.Timestamp(year + 1, 2, 28))].dropna(subset=[col])
    summer = sel[sel.date <= pd.Timestamp(year, 8, 31)][col]
    if len(summer) < 2:
        return None
    top = float(summer.median())
    t = (sel.date - t0).dt.days.values + 1.0
    y = sel[col].values
    if s_fixed is None:
        f = lambda tt, t50, s: logistic(tt, t50, s, top, base)
        p0, lims = (300, 10), ((230, 2), (380, 40))
    else:
        f = lambda tt, t50: logistic(tt, t50, s_fixed, top, base)
        p0, lims = (300,), ((230,), (380,))
    try:
        p, _ = curve_fit(f, t, y, p0=p0, bounds=lims)
    except RuntimeError:
        return None
    resid = y - f(t, *p)
    boots = []
    for _ in range(nboot):
        yb = f(t, *p) + rng.choice(resid, len(resid), replace=True)
        try:
            boots.append(curve_fit(f, t, yb, p0=p, bounds=lims)[0])
        except RuntimeError:
            pass
    if s_fixed is not None:
        p = np.r_[p, s_fixed]
        boots = [np.r_[b, s_fixed] for b in boots]
    boots = np.array(boots)
    t15 = p[0] + p[1] * T15_OFFSET
    t15b = boots[:, 0] + boots[:, 1] * T15_OFFSET
    last_obs = sel.date.max()
    autumn_obs = int(((sel.date.dt.month >= 9) | (sel.date.dt.year > year)).sum())
    return {"year": year, "top": top, "base": base, "t50": p[0], "s": p[1], "t15": t15,
            "t50_lo": np.percentile(boots[:, 0], 5), "t50_hi": np.percentile(boots[:, 0], 95),
            "t15_lo": np.percentile(t15b, 5), "t15_hi": np.percentile(t15b, 95),
            "frac_left_at_target": float(1 / (1 + np.exp((TARGET_DOY - p[0]) / p[1]))),
            "n_obs": len(sel), "n_autumn_obs": autumn_obs, "last_obs": last_obs.date().isoformat(),
            "rmse": float(np.sqrt(np.mean(resid ** 2)))}


def doy_to_str(year, doy):
    return (date(year, 1, 1) + timedelta(days=float(doy) - 1)).strftime("%d %b")


def analyse(sd, col):
    """Vrije fit per jaar, daarna een fit met vaste steilheid (mediaan van de afgeronde jaren) voor alle jaren."""
    c = sd.dropna(subset=[col])
    base = float(c[(c.date.dt.day_of_year >= 349) | (c.date.dt.day_of_year <= 59)][col].median())
    rng = np.random.default_rng(42)
    free = pd.DataFrame([f for y in range(2017, 2027) if (f := fit_year(sd, y, base, rng, col=col)) is not None])
    s_med = float(free[free.year < 2026].s.median())
    res = pd.DataFrame([fit_year(sd, int(y), base, rng, s_fixed=s_med, col=col) for y in free.year])
    res["s_fixed"] = s_med
    for k in ("t50", "t50_lo", "t50_hi", "t15", "rmse"):
        res[k + "_free"] = free[k].values
    return res


def report(res, label):
    print(f"--- {label}: winterbasis {res.base.iloc[0]:.3f}, vaste steilheid s = {res.s_fixed.iloc[0]:.1f} dagen")
    for _, r in res.iterrows():
        y = int(r.year)
        print(f"{y}: t50 {doy_to_str(y, r.t50)} [{doy_to_str(y, r.t50_lo)} - {doy_to_str(y, r.t50_hi)}], "
              f"t15 {doy_to_str(y, r.t15)}, groen over op 28 okt {r.frac_left_at_target:.0%}, "
              f"herfstobs {r.n_autumn_obs}, rmse {r.rmse:.3f} (vrije fit: t50 {doy_to_str(y, r.t50_free)})")


def pixel_map(dates, ndvi, clear, decid, conif, summer, winter, day):
    """Aandeel groen per loofbospixel op `day`, gecorrigeerd voor het scene-effect via het naaldbos."""
    i = int(np.where(dates == pd.Timestamp(day))[0][0])
    conif_clim = np.nanmedian(np.where(clear & conif, ndvi, np.nan))
    conif_now = np.median(ndvi[i][clear[i] & conif])
    corr = ndvi[i] + (conif_clim - conif_now)
    frac = (corr - winter) / (summer - winter)
    frac[~(decid & clear[i])] = np.nan
    return frac, float(conif_now - conif_clim)


def main():
    dates, ndvi, clear, grid = load()
    inside, transform, ring = site_mask(grid)
    summer, winter = pixel_climatology(dates, ndvi, clear)
    decid = inside & (summer >= 0.75) & (summer - winter >= 0.25)
    conif = inside & (summer >= 0.70) & (summer - winter <= 0.10) & (winter >= 0.65)
    print(f"pixels binnen grens {inside.sum()}, loofbos {decid.sum()} ({decid.sum() / 100:.1f} ha), "
          f"naaldbos {conif.sum()} ({conif.sum() / 100:.1f} ha)")

    sd = site_series(dates, ndvi, clear, decid)
    sd["ndvi_conif"] = site_series(dates, ndvi, clear, conif).ndvi
    # verschil loof - naald: scene-brede effecten (nevel, belichting, kijkhoek) vallen grotendeels weg
    sd["delta"] = sd.ndvi - sd.ndvi_conif
    sd.to_csv(S2 / "site_series.csv", index=False)

    res = analyse(sd, "delta")  # hoofdsignaal
    res.to_csv(S2 / "phenology_by_year.csv", index=False)
    report(res, "loof - naald (hoofdsignaal)")
    res_ndvi = analyse(sd, "ndvi")  # gevoeligheid: ruwe NDVI loofbos
    res_ndvi.to_csv(S2 / "phenology_by_year_ndvi.csv", index=False)
    report(res_ndvi, "ruwe NDVI loofbos (gevoeligheid)")

    frac, scene = pixel_map(dates, ndvi, clear, decid, conif, summer, winter, "2026-10-28")
    np.savez_compressed(S2 / "maps.npz", inside=inside, decid=decid, conif=conif, summer=summer, winter=winter,
                        frac_20261028=frac)
    print(f"kaart 28 okt 2026: mediaan aandeel groen {np.nanmedian(frac):.0%}, scene-correctie {scene:+.3f}")
    plot(sd, res, decid, conif, inside, ring, grid)


def plot(sd, res, decid, conif, inside, ring, grid):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={"width_ratios": [1, 1.6, 1.1]})
    ax = axes[0]
    img = np.ones(inside.shape + (3,))
    img[inside] = (0.88, 0.88, 0.86)
    img[conif] = (0.36, 0.42, 0.40)
    img[decid] = (0.92, 0.41, 0.20)
    b = grid["bounds"]
    ax.imshow(img, extent=(b[0], b[2], b[1], b[3]))
    xs, ys = zip(*ring)
    ax.plot(xs, ys, color="k", lw=0.8)
    ax.set_title(f"Loofbos ({decid.sum() / 100:.0f} ha) en naaldbos ({conif.sum() / 100:.0f} ha)")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    for _, r in res.iterrows():
        y = int(r.year)
        now = y == 2026
        col = "#eb6834" if now else "#a7a69f"
        o = sd[(sd.date >= pd.Timestamp(y, 8, 1)) & (sd.date <= pd.Timestamp(y + 1, 1, 15))].dropna(subset=["delta"])
        doy = (o.date - pd.Timestamp(y, 1, 1)).dt.days + 1
        ax.plot(doy, o.delta, "o", ms=5 if now else 3, color=col, zorder=3 if now else 2)
        tt = np.arange(213, 381)
        if now:
            tt = tt[tt <= (pd.Timestamp(r.last_obs) - pd.Timestamp(y, 1, 1)).days + 1]
        ax.plot(tt, logistic(tt, r.t50, r.s_fixed, r.top, r.base), color=col, lw=2.5 if now else 1,
                label="2026" if now else ("2017-2025" if y == 2017 else None), zorder=3 if now else 1)
    ax.axvline(TARGET_DOY, color="#52514e", ls=":", lw=1)
    ax.set_xticks([213, 244, 274, 305, 335, 366])
    ax.set_xticklabels(["1 aug", "1 sep", "1 okt", "1 nov", "1 dec", "1 jan"])
    ax.set_ylabel("NDVI loofbos - NDVI naaldbos")
    ax.set_title("Groenafname loofbos, gecorrigeerd met naaldbos als referentie")
    ax.legend(loc="lower left")

    ax = axes[2]
    yy = res.year.astype(int).values
    cols = ["#eb6834" if y == 2026 else "#52514e" for y in yy]
    for y, t, lo, hi, c in zip(yy, res.t50, res.t50_lo, res.t50_hi, cols):
        ax.errorbar(y, t, yerr=[[t - lo], [hi - t]], fmt="o", color=c, capsize=3)
    ax.axhline(res[res.year < 2026].t50.median(), color="#a7a69f", lw=1)
    ax.set_yticks([288, 296, 305, 313])
    ax.set_yticklabels(["15 okt", "23 okt", "1 nov", "9 nov"])
    ax.set_title("Midden van de groenafname (t50) per jaar, 90%-interval")
    fig.tight_layout()
    fig.savefig(HERE / "s2_herfst_spanderswoud.png", dpi=110)


if __name__ == "__main__":
    main()
