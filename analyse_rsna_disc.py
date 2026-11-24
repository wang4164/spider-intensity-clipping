#!/usr/bin/env python
"""Summaries of the RSNA disc replay (rsna_disc_replay.csv): incidence of |disc/CSF ratio error| above thresholds,
error components, disc clipping, and the clipping-based screening rule, overall and by series type
(plain T2 vs fat-suppressed, from the T2 fat screen). CIs: 2000 bootstrap resamples of studies.
Writes results/tables/Table_rsna_disc*.csv."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

from paths import RESULTS as RES
tag = sys.argv[1] if len(sys.argv) > 1 else ""
D = pd.read_csv(RES / f"rsna_disc_replay{tag}.csv")
D = D[D.t2_type.isin(["plain_T2", "fat_suppressed"])].copy()
D["err_total"] = D.R2 / D.R0 - 1
D["err_offset"] = D.R1 / D.R0 - 1
D["err_clip"] = D.R2 / D.R1 - 1
rng = np.random.default_rng(20261124)


def boot(d, flag, B=2000):
    ids = d.study.unique(); grp = {s: x[flag].values for s, x in d.groupby("study")}
    bs = [np.concatenate([grp[s] for s in rng.choice(ids, len(ids))]).mean() for _ in range(B)]
    return f"{100*d[flag].mean():.0f} ({100*np.percentile(bs, 2.5):.0f}–{100*np.percentile(bs, 97.5):.0f})"


groups = [("All", D), ("Plain T2", D[D.t2_type == "plain_T2"]), ("Fat-suppressed", D[D.t2_type == "fat_suppressed"])]
rows = []
for name, d in groups:
    d = d.copy()
    r = {"Series type": name, "Discs (studies)": f"{len(d)} ({d.study.nunique()})"}
    for t in (0.10, 0.20, 0.30):
        d["f"] = (d.err_total.abs() > t).astype(float); r[f"Error >{int(100*t)}%"] = boot(d, "f")
    d["f"] = (d.err_total > 0).astype(float); r["Ratio raised"] = boot(d, "f")
    d["f"] = (d.disc_clipped > 0.5).astype(float); r["Disc median censored"] = boot(d, "f")
    d["f"] = (d.csf_clipped > 0.5).astype(float); r["CSF median censored"] = boot(d, "f")
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(RES / "tables" / f"Table_rsna_disc{tag}.csv", index=False)
print("== RSNA disc replay: % of discs (95% CI, study bootstrap)\n", T.to_string(index=False))

f = lambda s: f"{100*s.median():+.1f} [{100*s.quantile(.25):+.1f}, {100*s.quantile(.75):+.1f}]"
print("\n== Components, median [IQR] %")
for name, d in groups:
    print(f"  {name:15s} offset {f(d.err_offset)} | clipping {f(d.err_clip)} | total {f(d.err_total)} | "
          f"|offset|>10% {100*(d.err_offset.abs() > .1).mean():.0f}% | |clip|>10% {100*(d.err_clip.abs() > .1).mean():.0f}%")

print("\n== Screening rule (CSF median censored = level 3) against |error| > 10%")
q = []
for name, d in groups:
    bad = d.err_total.abs() > 0.10; l3 = d.csf_clipped > 0.5
    off = bad & (d.err_offset.abs() > 0.10); clipd = bad & ~off
    tp, fn = int((l3 & bad).sum()), int((~l3 & bad).sum()); fp, tn = int((l3 & ~bad).sum()), int((~l3 & ~bad).sum())
    q.append({"Series type": name, "Discs with error >10%": f"{int(bad.sum())}/{len(d)}",
              "Flagged by level 3": f"{tp}/{tp+fn} ({100*tp/max(tp+fn,1):.0f}%)",
              "Clipping-driven errors flagged": f"{int((l3 & clipd).sum())}/{int(clipd.sum())}",
              "Offset-driven errors flagged": f"{int((l3 & off).sum())}/{int(off.sum())}",
              "Specificity": f"{tn}/{tn+fp} ({100*tn/max(tn+fp,1):.0f}%)",
              "Missed errors >10%": f"{fn}/{len(d)} ({100*fn/len(d):.0f}% of discs)"})
Q = pd.DataFrame(q); Q.to_csv(RES / "tables" / f"Table_rsna_disc_qc{tag}.csv", index=False)
print(Q.to_string(index=False))
print("\nlevels:", D.level.value_counts().to_dict())
geo = ((D.disc_row - D.canal_row) * D.ps).describe().round(1).to_dict()
print("disc centroid minus canal point, cranio-caudal mm:", {k: geo[k] for k in ["min", "50%", "max"]})
