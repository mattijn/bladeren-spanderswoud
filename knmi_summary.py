"""September/oktober-temperatuur en nachtvorst in De Bilt (KNMI 260) per jaar.

Luchtvorst: TN < 0 (minimum op 1,5 m). Grondvorst: T10N < 0 (minimum op 10 cm). Beide geteld over 1 sep - 31 okt,
zodat 2026 (data t/m 31 okt) vergelijkbaar is met de andere jaren. Eerste grondvorst: vanaf 1 aug.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
df = pd.read_csv(HERE / "s2" / "knmi_debilt.csv", comment="#", header=None, names=["stn", "date", "TG", "TN", "T10N"],
                 skipinitialspace=True)
df["date"] = pd.to_datetime(df.date.astype(str))
df[["TG", "TN", "T10N"]] = df[["TG", "TN", "T10N"]] / 10
rows = []
for y in range(2017, 2027):
    sep = df[(df.date.dt.year == y) & (df.date.dt.month == 9)]
    okt = df[(df.date.dt.year == y) & (df.date.dt.month == 10)]
    so = df[(df.date >= f"{y}-09-01") & (df.date <= f"{y}-10-31")]
    autumn = df[(df.date >= f"{y}-08-01") & (df.date <= f"{y}-12-31")]
    ground = autumn[autumn.T10N < 0].date
    rows.append({"year": y, "T_sep": round(sep.TG.mean(), 1), "T_okt": round(okt.TG.mean(), 1),
                 "vorstdagen_sep_okt": int((so.TN < 0).sum()), "grondvorst_sep_okt": int((so.T10N < 0).sum()),
                 "eerste_grondvorst": ground.min().date().isoformat() if len(ground) else "-",
                 "data_tot": autumn.date.max().date().isoformat()})
out = pd.DataFrame(rows)
out.to_csv(HERE / "s2" / "knmi_sep_okt.csv", index=False)
print(out.to_string(index=False))
