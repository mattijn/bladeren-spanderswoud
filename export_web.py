"""Zet de analyse-uitvoer om naar docs/data.json voor de Observable Plot-pagina.

Gebruik (na s2_phenology.py, s2_spring.py en knmi_summary.py):
    uv run --no-project --with numpy,pandas,pyproj python export_web.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.stats import spearmanr

from brackets import brackets

HERE = Path(__file__).parent
S2 = HERE / "s2"
DOCS = HERE / "docs"


def logistic(t, t50, s, top, base):
    return base + (top - base) / (1 + np.exp((t - t50) / s))


def r(x, n=4):
    return None if x is None or not np.isfinite(float(x)) else round(float(x), n)


def stats(sd, res, res_ndvi, spr, knmi, px):
    """Getallen voor de conclusie: verbanden over de referentiejaren en de gevoeligheid van de 2026-uitkomst."""
    ref = res[res.year < 2026].merge(knmi, on="year")
    rho = lambda a, b: dict(zip(("rho", "p"), (r(v, 2) for v in spearmanr(a, b))), n=len(a))
    sp = spr.reset_index()[["year", "sos50"]]
    link = sp[sp.year < 2026].merge(res[["year", "t50"]], on="year")
    now = res[res.year == 2026].iloc[0]
    last = sd.dropna(subset=["delta"]).date.max()
    obs_now = float(sd.set_index("date").loc[last, "delta"])
    tops = [now.top, ref.top.median(), ref.top.min(), ref.top.max()]
    band = [(obs_now - now.base) / (t - now.base) for t in tops]
    nd = res_ndvi.sort_values("frac_left_at_target", ascending=False).year.tolist()
    pxsos = {y: float(np.nanmedian(px[f"sos_{y}"])) for y in range(2018, 2027) if f"sos_{y}" in px}
    return {"rho_sep": rho(ref.T_sep, ref.t50), "rho_okt": rho(ref.T_okt, ref.t50), "rho_link": rho(link.sos50, link.t50),
            "last_obs": last.date().isoformat(), "frac_obs_last": r(band[0], 3),
            "frac_band": [r(min(band), 3), r(max(band), 3)],
            "rank_ndvi": nd.index(2026) + 1, "n_ndvi": len(nd),
            "px_sos_2026": r(pxsos.get(2026), 1),
            "px_sos_ref_median": r(np.median([v for y, v in pxsos.items() if y < 2026]), 1)}


def main():
    DOCS.mkdir(exist_ok=True)
    sd = pd.read_csv(S2 / "site_series.csv", parse_dates=["date"])
    res = pd.read_csv(S2 / "phenology_by_year.csv")
    res_ndvi = pd.read_csv(S2 / "phenology_by_year_ndvi.csv")
    knmi = pd.read_csv(S2 / "knmi_sep_okt.csv")
    grid = json.loads((S2 / "cube_grid.json").read_text())
    maps = np.load(S2 / "maps.npz")

    spr = pd.read_csv(S2 / "spring_by_year.csv").set_index("year")
    br = brackets().set_index(["year", "phase"])

    # waarnemingen feb - half jan, met dag-van-het-seizoen t.o.v. 1 jan van het seizoensjaar
    obs = []
    for _, row in sd.dropna(subset=["delta"]).iterrows():
        d = row.date
        season = d.year if d.month >= 2 else d.year - 1
        doy = (d - pd.Timestamp(season, 1, 1)).days + 1
        if 32 <= doy <= 380:
            obs.append({"season": season, "date": d.date().isoformat(), "doy": doy, "delta": r(row.delta),
                        "ndvi": r(row.ndvi), "ndvi_conif": r(row.ndvi_conif)})

    def bk(y, phase, side):
        v = br.loc[(y, phase), side] if (y, phase) in br.index else None
        return None if v is None or pd.isna(v) else int(v)

    curves, years = [], []
    for _, f in res.iterrows():
        y = int(f.year)
        sp = spr.loc[y] if y in spr.index else None
        last = (pd.Timestamp(f.last_obs) - pd.Timestamp(y, 1, 1)).days + 1
        tt = np.arange(32 if sp is not None else 166, (min(last, 380) if y == 2026 else 380) + 1)
        # dubbel-logistisch: voorjaarstoename x herfstafname tussen dezelfde winterbasis en hetzelfde zomerniveau
        rise = 1 / (1 + np.exp(-(tt - sp.sos50) / sp.s)) if sp is not None else np.ones_like(tt, float)
        fall = 1 / (1 + np.exp((tt - f.t50) / f.s_fixed))
        curves += [{"season": y, "doy": int(t), "delta": r(v)} for t, v in zip(tt, f.base + (f.top - f.base) * rise * fall)]
        k = knmi[knmi.year == y]
        n = res_ndvi[res_ndvi.year == y]
        years.append({"season": y, "t50": r(f.t50, 1), "t50_lo": r(f.t50_lo, 1), "t50_hi": r(f.t50_hi, 1),
                      "t15": r(f.t15, 1), "frac_28okt": r(f.frac_left_at_target, 3), "top": r(f.top), "base": r(f.base),
                      "n_autumn_obs": int(f.n_autumn_obs), "rmse": r(f.rmse),
                      "aut_lo": bk(y, "herfst", "lo"), "aut_hi": bk(y, "herfst", "hi"),
                      "sos50": r(sp.sos50, 1) if sp is not None else None,
                      "sos50_free": r(sp.sos50_free, 1) if sp is not None else None,
                      "n_spring_obs": int(sp.n_spring_obs) if sp is not None else None,
                      "spr_lo": bk(y, "voorjaar", "lo"), "spr_hi": bk(y, "voorjaar", "hi"),
                      "season_days": r(f.t50 - sp.sos50, 0) if sp is not None else None,
                      "t50_ndvi": r(n.t50.iloc[0], 1) if len(n) else None,
                      "frac_28okt_ndvi": r(n.frac_left_at_target.iloc[0], 3) if len(n) else None,
                      "T_sep": r(k.T_sep.iloc[0], 1) if len(k) else None, "T_okt": r(k.T_okt.iloc[0], 1) if len(k) else None,
                      "vorstdagen": int(k.vorstdagen_sep_okt.iloc[0]) if len(k) else None,
                      "grondvorst": int(k.grondvorst_sep_okt.iloc[0]) if len(k) else None,
                      "eerste_grondvorst": (None if not len(k) or k.eerste_grondvorst.iloc[0] == "-"
                                            else k.eerste_grondvorst.iloc[0])})

    # kaart als raster in UTM 31N (rij 0 = noord); 0 = buiten grens, 1 = overig, 2 = naaldbos, 3 = loofbos
    b = grid["bounds"]
    h, w = grid["shape"]
    frac, decid, conif, inside = maps["frac_20261028"], maps["decid"], maps["conif"], maps["inside"]
    kind = np.where(decid, 3, np.where(conif, 2, np.where(inside, 1, 0)))
    t = Transformer.from_crs(4326, 32631, always_xy=True)
    boundary = json.loads((HERE / "spanderswoud_osm.geojson").read_text())
    ring = [[r(c, 1) for c in t.transform(lon, lat)] for lon, lat in boundary["features"][0]["geometry"]["coordinates"][0]]
    raster = {"x1": b[0], "x2": b[2], "y1": b[1], "y2": b[3], "width": w, "height": h,
              "kind": kind.ravel().tolist(), "frac": [r(v, 3) for v in np.where(kind == 3, frac, np.nan).ravel()],
              "boundary": ring}

    # dag van uitloop/verkleuring per loofbospixel (s2_pixel_doy.py); -1 = op de laatste opname nog niet bereikt
    px = np.load(S2 / "pixel_doy.npz")
    pxs = pd.read_csv(S2 / "pixel_doy_summary.csv")
    enc = lambda a: [None if not np.isfinite(v) else int(round(v)) for v in a]
    pixel = {"index": px["pixels"].tolist(), "years": {}}
    for _, row in pxs.iterrows():
        y = int(row.year)
        pixel["years"][str(y)] = {"sos": enc(px[f"sos_{y}"]), "eos": enc(px[f"eos_{y}"]),
                                  "gap_spring": r(row.max_gap_spring, 0), "gap_autumn": r(row.max_gap_autumn, 0)}

    data = {"generated": pd.Timestamp.now().isoformat(timespec="minutes"), "target_doy": 301,
            "s_fixed": r(res.s_fixed.iloc[0], 2), "obs": obs, "curves": curves, "years": years,
            "map": raster, "pixel": pixel, "stats": stats(sd, res, res_ndvi, spr, knmi, px),
            "area_ha": {"loof": round(int(decid.sum()) / 100, 1), "naald": round(int(conif.sum()) / 100, 1)}}
    (DOCS / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"docs/data.json: {len(obs)} waarnemingen, {len(years)} jaren, kaart {w}x{h}, "
          f"{(DOCS / 'data.json').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
