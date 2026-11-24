#!/usr/bin/env python
"""Does the inferred transform change the association between Pfirrmann grade and the disc-to-CSF ratio?

Uses ratio_incidence_discs.csv (native SPIDER T2 and SPACE, per disc and CSF region):
  native ratio R0 and transformed ratio R2 = R0 * (1 + err_total).
Per sequence x CSF region, native vs transformed:
  * Spearman rho between ratio and grade;
  * within-patient slope of log ratio per Pfirrmann grade (patient-demeaned, i.e. fixed effects), with the change
    in slope and its 95% CI from 2000 bootstrap resamples of patients;
  * contrast grade 5 vs grade 2: ratio of median ratios;
  * within-patient pairs of discs whose ratio order flips after the transform.
Also reports how often re-selecting the core region on the transformed image enlarged it (ties at the ceiling).
"""
from __future__ import annotations
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding="utf-8")

from paths import RESULTS as RES
D = pd.read_csv(RES / "ratio_incidence_discs.csv").dropna(subset=["pfirrmann"])
D["R0"] = D.ratio_native; D["R2"] = D.ratio_native * (1 + D.err_total)
rng = np.random.default_rng(20261124)


def fe_slope(d, col):
    x = d.pfirrmann - d.groupby("pid").pfirrmann.transform("mean")
    y = np.log(d[col]) - d.groupby("pid")[col].transform(lambda s: np.log(s).mean())
    return float((x * y).sum() / (x * x).sum())


rows = []
for (kind, roi), d in D[D.roi.isin(["whole", "ero1", "core"])].groupby(["kind", "roi"], sort=False):
    s0, s2 = fe_slope(d, "R0"), fe_slope(d, "R2")
    pids = d.pid.unique(); grp = {p: x for p, x in d.groupby("pid")}
    bs = []
    for _ in range(2000):
        b = pd.concat([grp[p].assign(pid=i) for i, p in enumerate(rng.choice(pids, len(pids)))])
        bs.append(fe_slope(b, "R2") - fe_slope(b, "R0"))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    m = lambda col, g: d.loc[d.pfirrmann == g, col].median()
    flips = tot = 0
    for _, x in d.groupby("pid"):
        for (_, a), (_, b) in combinations(x.iterrows(), 2):
            if a.R0 == b.R0:
                continue
            tot += 1; flips += int(np.sign(a.R0 - b.R0) != np.sign(a.R2 - b.R2))
    rows.append({"Sequence": kind, "CSF region": roi, "Discs": len(d),
                 "rho native": round(spearmanr(d.R0, d.pfirrmann)[0], 3), "rho transformed": round(spearmanr(d.R2, d.pfirrmann)[0], 3),
                 "slope native (%/grade)": round(100 * (np.exp(s0) - 1), 1), "slope transformed (%/grade)": round(100 * (np.exp(s2) - 1), 1),
                 "slope change (log, 95% CI)": f"{s2 - s0:+.3f} ({lo:+.3f} to {hi:+.3f})",
                 "G5/G2 native": round(m("R0", 5) / m("R0", 2), 3), "G5/G2 transformed": round(m("R2", 5) / m("R2", 2), 3),
                 "within-patient order flips": f"{flips}/{tot} ({100*flips/max(tot,1):.1f}%)"})
T = pd.DataFrame(rows)
T.to_csv(RES / "tables" / "Table_grade_association.csv", index=False)
print(T.to_string(index=False))

# medians by grade, for Figure 5
G = (D[D.roi == "ero1"].groupby(["kind", "pfirrmann"])
     .agg(n=("R0", "size"), native=("R0", "median"), transformed=("R2", "median"), offset=("err_offset", "median"))
     .reset_index())
G.to_csv(RES / "grade_medians_ero1.csv", index=False)
print(G.round(3).to_string(index=False))
