"""Dag van uitloop en verkleuring per loofbospixel (10 m), per jaar.

Zelfde idee als de 'day of year'-methode (per pixel de eerste dag waarop een drempel overschreden wordt), aangepast
voor optische vegetatiedata:
- per pixel genormaliseerd: g = (NDVI - winter) / (zomer - winter), met de eigen winterbasis en het zomerniveau van
  dat jaar (half juni - augustus), zodat de drempel 0,5 overal 'de helft' betekent;
- per opname gecorrigeerd voor scene-effecten met het naaldbos (NDVI naald van die opname t.o.v. het meerjarig gemiddelde);
- een overschrijding telt pas als de volgende geldige opname hem bevestigt (geen enkele ruizige opname);
- de dag wordt lineair geinterpoleerd tussen de laatste opname voor en de eerste na de drempel, zodat wolkengaten
  de datum niet systematisch naar later schuiven.

Uitvoer s2/pixel_doy.npz: per jaar kaarten 'sos' (uitloop) en 'eos' (verkleuring, t50) als dag van het jaar.
NaN = geen bepaling mogelijk; -1 (alleen eos) = drempel op de laatste opname nog niet bereikt.

Gebruik (na s2_phenology.py):  uv run --no-project --with rasterio,pyproj,numpy,scipy,matplotlib,pandas python s2_pixel_doy.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from s2_phenology import load

HERE = Path(__file__).parent
S2 = HERE / "s2"
YEARS = range(2017, 2027)
HALF = 0.5


def crossing(t, g, rising):
    """Eerste bevestigde overschrijding van HALF per pixel, lineair geinterpoleerd.

    t: (n,) dagnummers; g: (n, p) genormaliseerde groenheid met NaN voor ontbrekend.
    Geeft (dag, voorlopig, niet_bereikt): voorlopig = alleen op de laatste opname gezien, niet bevestigd.
    """
    n, p = g.shape
    out = np.full(p, np.nan)
    done = np.zeros(p, bool)
    last_t, last_g = np.full(p, np.nan), np.full(p, np.nan)
    pend_t, pend_g = np.full(p, np.nan), np.full(p, np.nan)
    pend_pt, pend_pg = np.full(p, np.nan), np.full(p, np.nan)
    pending = np.zeros(p, bool)
    seen = np.zeros(p, bool)
    for i in range(n):
        v = np.isfinite(g[i]) & ~done
        c = (g[i] >= HALF) if rising else (g[i] < HALF)
        # bevestigd: tweede opname op rij aan de overkant van de drempel
        conf = v & pending & c
        out[conf] = interp(pend_pt[conf], pend_pg[conf], pend_t[conf], pend_g[conf])
        done |= conf
        # terugval: overschrijding niet bevestigd
        pending &= ~(v & ~c)
        # nieuwe kandidaat
        new = v & ~pending & c & ~done
        pend_t[new], pend_g[new] = t[i], g[i][new]
        pend_pt[new], pend_pg[new] = last_t[new], last_g[new]
        pending |= new
        upd = v & ~done
        last_t[upd], last_g[upd] = t[i], g[i][upd]
        seen |= v
    provisional = pending & ~done
    out[provisional] = interp(pend_pt[provisional], pend_pg[provisional], pend_t[provisional], pend_g[provisional])
    not_reached = seen & ~done & ~pending
    return out, provisional, not_reached


def interp(t0, g0, t1, g1):
    # zonder opname voor de overschrijding is het moment niet af te bakenen
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(t0), t0 + (HALF - g0) / (g1 - g0) * (t1 - t0), np.nan)


def main():
    dates, ndvi, clear, grid = load()
    m = np.load(S2 / "maps.npz")
    decid, summer_clim, winter = m["decid"], m["summer"], m["winter"]
    sd = pd.read_csv(S2 / "site_series.csv", parse_dates=["date"]).set_index("date")
    conif_clim = float(sd.ndvi_conif.median())

    idx = np.where(decid.ravel())[0]
    win = winter.ravel()[idx]
    res = {"pixels": idx, "shape": np.array(grid["shape"])}
    rows = []
    for y in YEARS:
        sel = [i for i, d in enumerate(dates) if pd.Timestamp(y, 2, 1) <= d <= pd.Timestamp(y + 1, 1, 31)
               and d in sd.index and np.isfinite(sd.loc[d, "delta"])]
        if not sel:
            continue
        t = np.array([(dates[i] - pd.Timestamp(y, 1, 1)).days + 1 for i in sel], float)
        corr = np.array([conif_clim - sd.loc[dates[i], "ndvi_conif"] for i in sel])
        v = np.stack([np.where(clear[i], ndvi[i], np.nan).ravel()[idx] for i in sel]) + corr[:, None]
        summ = (t >= 166) & (t <= 243)
        top = np.nanmedian(v[summ], axis=0) if summ.sum() >= 2 else summer_clim.ravel()[idx]
        if summ.sum() >= 2:
            few = np.sum(np.isfinite(v[summ]), axis=0) < 2
            top[few] = summer_clim.ravel()[idx][few]
        g = (v - win) / (top - win)

        spring = (t >= 60) & (t <= 166)
        sos, sos_prov, _ = crossing(t[spring], g[spring], rising=True) if spring.sum() >= 2 else (np.full(len(idx), np.nan),) * 3
        autumn = (t >= 227)
        eos, eos_prov, eos_open = crossing(t[autumn], g[autumn], rising=False)
        eos = np.where(eos_open, -1, eos)
        res[f"sos_{y}"], res[f"eos_{y}"] = sos, eos
        ok_s, ok_e = np.isfinite(sos), np.isfinite(eos) & (eos > 0)
        rows.append({"year": y, "sos_median": np.nanmedian(sos) if ok_s.any() else np.nan, "sos_cover": ok_s.mean(),
                     "eos_median": np.median(eos[ok_e]) if ok_e.any() else np.nan, "eos_cover": ok_e.mean(),
                     "eos_not_reached": float(np.mean(eos == -1)), "eos_provisional": float(np.mean(eos_prov)),
                     "max_gap_spring": float(np.max(np.diff(t[spring]))) if spring.sum() > 1 else np.nan,
                     "max_gap_autumn": float(np.max(np.diff(t[autumn & (t <= 335)]))) if (autumn & (t <= 335)).sum() > 1 else np.nan})
    np.savez_compressed(S2 / "pixel_doy.npz", **res)
    summary = pd.DataFrame(rows)
    summary.to_csv(S2 / "pixel_doy_summary.csv", index=False)
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
