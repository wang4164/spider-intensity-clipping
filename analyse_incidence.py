#!/usr/bin/env python
"""Summaries of ratio_incidence_discs.csv: incidence of |ratio error| above thresholds (patient bootstrap CIs),
sign of the error, fixed vs reselected core ROI, a three-level QC rule against |error| > 10%, offset error by
Pfirrmann grade, and leave-one-patient-out robustness. Writes results/tables/Table_incidence*.csv and prints."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

from paths import RESULTS as RES
D = pd.read_csv(RES / "ratio_incidence_discs.csv")
rng = np.random.default_rng(20261123)
B = 2000
ROIS = ["whole", "ero1", "ero2", "core", "core_reselected"]


def boot_share(d, flag, B=B):
    pids = d.pid.unique(); grp = {p: x[flag].values for p, x in d.groupby("pid")}
    est = d[flag].mean()
    bs = [np.concatenate([grp[p] for p in rng.choice(pids, len(pids))]).mean() for _ in range(B)]
    return f"{100*est:.0f} ({100*np.percentile(bs, 2.5):.0f}–{100*np.percentile(bs, 97.5):.0f})"


# 1. incidence of |total error| above thresholds, by sequence and ROI
rows = []
for (kind, roi), d in D.groupby(["kind", "roi"], sort=False):
    d = d.copy(); r = {"Sequence": kind, "CSF region": roi, "Discs (patients)": f"{len(d)} ({d.pid.nunique()})"}
    for t in (0.05, 0.10, 0.20, 0.30):
        d[f"f{t}"] = (d.err_total.abs() > t).astype(float)
        r[f"|error| >{int(100*t)}%"] = boot_share(d, f"f{t}")
    d["pos"] = (d.err_total > 0).astype(float); d["neg"] = (d.err_total < -0.05).astype(float)
    r["Raised (>0)"] = boot_share(d, "pos")
    d["qn10"] = (d.err_total_qn.abs() > 0.10).astype(float)
    r["|error| >10% with quantile-normal CSF"] = boot_share(d, "qn10")
    rows.append(r)
T1 = pd.DataFrame(rows); T1.to_csv(RES / "tables" / "Table_incidence.csv", index=False)
print("== Incidence of |disc/CSF ratio error|, % of discs (95% patient-bootstrap CI)\n", T1.to_string(index=False))

# 2. components: offset vs clipping, per ROI (median and share above 10%)
comp = D.assign(err_clip=(1 + D.err_total) / (1 + D.err_offset) - 1)
print("\n== Error components, median [IQR] %")
for (kind, roi), d in comp.groupby(["kind", "roi"], sort=False):
    f = lambda s: f"{100*s.median():+.1f} [{100*s.quantile(.25):+.1f}, {100*s.quantile(.75):+.1f}]"
    print(f"  {kind:6s} {roi:16s} offset {f(d.err_offset)} | clipping {f(d.err_clip)} | total {f(d.err_total)} | "
          f"|offset|>10%: {100*(d.err_offset.abs()>.1).mean():.0f}% | |clip|>10%: {100*(d.err_clip.abs()>.1).mean():.0f}%")

# 3. fixed vs reselected core ROI
fx = D[D.roi == "core"].set_index(["series", "ivd"]); rs = D[D.roi == "core_reselected"].set_index(["series", "ivd"])
j = fx.join(rs, lsuffix="_fixed", rsuffix="_re", how="inner")
diff = 100 * (j.err_total_re - j.err_total_fixed)
print(f"\n== Core ROI re-selected on released image vs fixed native coordinates: n={len(j)}; voxel overlap median "
      f"{100*j.roi_overlap_re.median():.1f}% (min {100*j.roi_overlap_re.min():.1f}%); error difference median {diff.median():+.2f} pp, "
      f"IQR {diff.quantile(.25):+.2f} to {diff.quantile(.75):+.2f}, max |diff| {diff.abs().max():.2f} pp")

# 4. three-level QC rule (observable on released data) vs |total error| > 10%
print("\n== QC levels from the clipped share of the CSF region (released image) vs true |error| > 10%")
qc = []
for roi in ["whole", "ero1", "core_reselected"]:
    d = D[D.roi == roi].copy()
    d["level"] = np.select([d.csf_clipped == 0, d.csf_clipped <= 0.5], ["L1 none", "L2 <=50%"], "L3 >50%")
    d["bad"] = d.err_total.abs() > 0.10
    for lvl, g in d.groupby("level"):
        qc.append({"CSF region": roi, "Level": lvl, "Discs": len(g), "|error| >10%": f"{int(g.bad.sum())}/{len(g)} ({100*g.bad.mean():.0f}%)",
                   "of which offset-driven": f"{int((g.bad & (g.err_offset.abs() > 0.10)).sum())}"})
    flag = d.level == "L3 >50%"
    tp, fn_ = int((flag & d.bad).sum()), int((~flag & d.bad).sum()); fp, tn = int((flag & ~d.bad).sum()), int((~flag & ~d.bad).sum())
    print(f"  {roi:16s} L3 flag: sensitivity {tp}/{tp+fn_} ({100*tp/max(tp+fn_,1):.0f}%), specificity {tn}/{tn+fp} ({100*tn/max(tn+fp,1):.0f}%)")
    bad_clip = d.bad & (d.err_offset.abs() <= 0.10)
    print(f"  {'':16s} among clipping-driven errors only: L3 sensitivity {int((flag & bad_clip).sum())}/{int(bad_clip.sum())}")
Q = pd.DataFrame(qc); Q.to_csv(RES / "tables" / "Table_qc_levels.csv", index=False); print(Q.to_string(index=False))

# 5. offset error by Pfirrmann grade (2D T2, whole canal) and by disc darkness
print("\n== Offset component by Pfirrmann grade (eroded 1 mm CSF), median [IQR] %")
for kind in ["2D T2", "SPACE"]:
    d = D[(D.roi == "ero1") & (D.kind == kind)].dropna(subset=["pfirrmann"])
    print(" ", kind, {int(gr): f"{100*x.err_offset.median():+.1f} [{100*x.err_offset.quantile(.25):+.1f}, {100*x.err_offset.quantile(.75):+.1f}] n={len(x)}"
                      for gr, x in d.groupby("pfirrmann")})
    rho = d[["err_offset", "pfirrmann"]].corr(method="spearman").iloc[0, 1]
    print(f"   Spearman rho(offset error, Pfirrmann) = {rho:.2f}; rho(offset error, native ratio) = "
          f"{d[['err_offset','ratio_native']].corr(method='spearman').iloc[0,1]:.2f}")

# 6. leave-one-patient-out: range of the share with |error| > 10% (ero1)
print("\n== Leave-one-patient-out range of share with |error| >10% (ero1)")
for kind in ["2D T2", "SPACE"]:
    d = D[(D.roi == "ero1") & (D.kind == kind)]
    vals = [100 * (d[d.pid != p].err_total.abs() > .1).mean() for p in d.pid.unique()]
    print(f"  {kind}: full {100*(d.err_total.abs()>.1).mean():.0f}%, LOO {min(vals):.0f}–{max(vals):.0f}%")
