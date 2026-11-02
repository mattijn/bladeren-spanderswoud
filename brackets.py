"""Afbakening van t50 (herfst) en sos50 (voorjaar) door de waarnemingen zelf: laatste opname voor en eerste na
het halverwegepunt tussen winterbasis en zomerniveau. Ontbreekt de opname erna, dan is het moment 'nog niet bereikt'."""
from pathlib import Path

import pandas as pd

S2 = Path(__file__).parent / "s2"


def doy(d, year):
    return (d - pd.Timestamp(year, 1, 1)).days + 1


def bracket(obs, half, rising):
    """obs: gesorteerde (datum, waarde). Eerste opname aan de 'na'-kant van half, en de laatste ervoor."""
    after = obs[obs.delta >= half] if rising else obs[obs.delta < half]
    if after.empty:
        return obs.date.max(), None
    first = after.date.min()
    before = obs[obs.date < first]
    return (before.date.max() if len(before) else None), first


def brackets():
    sd = pd.read_csv(S2 / "site_series.csv", parse_dates=["date"]).dropna(subset=["delta"]).sort_values("date")
    aut = pd.read_csv(S2 / "phenology_by_year.csv")
    spr = pd.read_csv(S2 / "spring_by_year.csv")
    rows = []
    for _, a in aut.iterrows():
        y = int(a.year)
        o = sd[(sd.date >= pd.Timestamp(y, 8, 15)) & (sd.date <= pd.Timestamp(y + 1, 1, 31))]
        lo, hi = bracket(o, (a.top + a.base) / 2, rising=False)
        rows.append({"year": y, "phase": "herfst", "lo": doy(lo, y) if lo is not None else None,
                     "hi": doy(hi, y) if hi is not None else None})
    for _, s in spr.iterrows():
        y = int(s.year)
        o = sd[(sd.date >= pd.Timestamp(y, 3, 1)) & (sd.date <= pd.Timestamp(y, 6, 15))]
        lo, hi = bracket(o, (s.top + s.base) / 2, rising=True)
        rows.append({"year": y, "phase": "voorjaar", "lo": doy(lo, y) if lo is not None else None,
                     "hi": doy(hi, y) if hi is not None else None})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    b = brackets()
    b.to_csv(S2 / "brackets.csv", index=False)
    print(b.to_string(index=False))
