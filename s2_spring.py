"""Bladuitloop (voorjaar) van het loofbos in Spanderswoud en het verband met het herfstmoment.

Zelfde signaal als s2_phenology.py (NDVI loof - naald), nu een logistische toename tussen de winterbasis en het
zomerniveau van dat jaar, met vaste steilheid. Uitkomst: sos50 (helft van de toename bereikt).

Gebruik (na s2_phenology.py):  uv run --no-project --with numpy,scipy,pandas python s2_spring.py
"""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

HERE = Path(__file__).parent
S2 = HERE / "s2"


def rise(t, t50, s, top, base):
    return base + (top - base) / (1 + np.exp(-(t - t50) / s))


def fit(sd, year, base, rng, s_fixed=None, nboot=300):
    t0 = pd.Timestamp(year, 1, 1)
    sel = sd[(sd.date >= pd.Timestamp(year, 2, 1)) & (sd.date <= pd.Timestamp(year, 8, 31))].dropna(subset=["delta"])
    summer = sel[sel.date >= pd.Timestamp(year, 6, 15)].delta
    spring = sel[(sel.date >= pd.Timestamp(year, 3, 15)) & (sel.date <= pd.Timestamp(year, 5, 31))]
    if len(summer) < 2 or len(spring) < 3:
        return None
    top = float(summer.median())
    t = (sel.date - t0).dt.days.values + 1.0
    y = sel.delta.values
    if s_fixed is None:
        f = lambda tt, t50, s: rise(tt, t50, s, top, base)
        p0, lims = (120, 8), ((80, 2), (170, 30))
    else:
        f = lambda tt, t50: rise(tt, t50, s_fixed, top, base)
        p0, lims = (120,), ((80,), (170,))
    p, _ = curve_fit(f, t, y, p0=p0, bounds=lims)
    resid = y - f(t, *p)
    boots = []
    for _ in range(nboot):
        try:
            boots.append(curve_fit(f, t, f(t, *p) + rng.choice(resid, len(resid)), p0=p, bounds=lims)[0][0])
        except RuntimeError:
            pass
    return {"year": year, "sos50": p[0], "s": p[1] if s_fixed is None else s_fixed, "top": top, "base": base,
            "sos50_lo": np.percentile(boots, 5), "sos50_hi": np.percentile(boots, 95),
            "n_spring_obs": len(spring), "rmse": float(np.sqrt(np.mean(resid ** 2)))}


def d(year, doy):
    return (date(year, 1, 1) + timedelta(days=float(doy) - 1)).strftime("%d %b")


def main():
    sd = pd.read_csv(S2 / "site_series.csv", parse_dates=["date"])
    c = sd.dropna(subset=["delta"])
    base = float(c[(c.date.dt.day_of_year >= 349) | (c.date.dt.day_of_year <= 59)].delta.median())
    rng = np.random.default_rng(7)
    free = pd.DataFrame([f for y in range(2017, 2027) if (f := fit(sd, y, base, rng))])
    s_med = float(free.s.median())
    res = pd.DataFrame([fit(sd, int(y), base, rng, s_fixed=s_med) for y in free.year])
    res["sos50_free"] = free.sos50.values
    aut = pd.read_csv(S2 / "phenology_by_year.csv")[["year", "t50", "frac_left_at_target"]]
    res = res.merge(aut, on="year", how="left")
    res["season_days"] = res.t50 - res.sos50
    res.to_csv(S2 / "spring_by_year.csv", index=False)

    print(f"winterbasis {base:.3f}, vaste steilheid voorjaar s = {s_med:.1f} dagen")
    for _, r in res.iterrows():
        y = int(r.year)
        print(f"{y}: uitloop (sos50) {d(y, r.sos50)} [{d(y, r.sos50_lo)} - {d(y, r.sos50_hi)}] "
              f"(vrij {d(y, r.sos50_free)}), voorjaarsobs {r.n_spring_obs}, rmse {r.rmse:.3f} | "
              f"herfst t50 {d(y, r.t50)} | seizoen {r.season_days:.0f} d")
    ref = res[res.year < 2026]
    now = res[res.year == 2026].iloc[0]
    print(f"\nmediaan uitloop 2017-2025: {d(2026, ref.sos50.median())}; 2026: {d(2026, now.sos50)} "
          f"({now.sos50 - ref.sos50.median():+.0f} d); rang (vroeg->laat) {int((res.sos50 < now.sos50).sum()) + 1} van {len(res)}")
    rho, p = spearmanr(res.sos50, res.t50)
    print(f"verband uitloop - herfst-t50 (alle jaren): Spearman rho = {rho:.2f}, p = {p:.2f}, n = {len(res)}")
    rho2, p2 = spearmanr(ref.sos50, ref.t50)
    print(f"zonder 2026: rho = {rho2:.2f}, p = {p2:.2f}, n = {len(ref)}")


if __name__ == "__main__":
    main()
