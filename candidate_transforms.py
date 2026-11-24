#!/usr/bin/env python
"""Which intensity transform explains the converted SPIDER volumes? Candidate comparison on the 73 native volumes.

Each candidate maps native values x to 4097 integer levels (0..4096, i.e. -1000..3096 after the offset):
  percentile clips p_lo..p_hi (5-95, 2-98, 1-99, 0.5-99.5), min-max scaling without clipping, and the DICOM
  display window (WC +/- WW/2 from overview.csv). Rounding is to the nearest integer after clipping.
For each candidate and volume: share of voxels at the ceiling and floor, and integer-lattice occupancy.
These are compared with the envelope observed in the 374 converted volumes.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
sys.stdout.reconfigure(encoding="utf-8")

from paths import SPIDER as ROOT, RESULTS, T1_REGIONS
RES = RESULTS


def num(v):
    try:
        return float(str(v).strip("[]").replace('"', "").split("\\")[0].split(",")[0])
    except Exception:
        return np.nan


def mapped(x, lo, hi):
    y = np.clip((x - lo) / (hi - lo), 0, 1) * 4096
    return np.round(y)


def stats(y):
    return {"ceiling": float((y >= 4096).mean()), "floor": float((y <= 0).mean()),
            "occupancy": float(np.unique(y).size / 4097)}


P = pd.read_csv(RES / "spider_encoding_probe.csv")
conv = P[(P["min"] == -1000) & (P["max"] == 3096)]
nat = P[~((P["min"] == -1000) & (P["max"] == 3096))]
ov = pd.read_csv(ROOT / "overview.csv").set_index("new_file_name")
rows = []
for fn in nat.file:
    x = sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT / "images" / f"{fn}.mha"))).astype(np.float64).ravel()
    q = lambda p: float(np.quantile(x, p))
    wc, ww = num(ov.loc[fn, "WindowCenter"]), num(ov.loc[fn, "WindowWidth"])
    cands = {"p5–p95 clip": (q(.05), q(.95)), "p2–p98 clip": (q(.02), q(.98)), "p1–p99 clip": (q(.01), q(.99)),
             "p0.5–p99.5 clip": (q(.005), q(.995)), "min–max scaling": (x.min(), x.max()),
             "DICOM window (WC ± WW/2)": (wc - ww / 2, wc + ww / 2),
             "min–p95 clip": (x.min(), q(.95)), "p1–p95 clip": (q(.01), q(.95)), "p10–p95 clip": (q(.10), q(.95)),
             "p5–p99 clip": (q(.05), q(.99))}
    for name, (lo, hi) in cands.items():
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        rows.append({"file": fn, "candidate": name, **stats(mapped(x, lo, hi))})
C = pd.DataFrame(rows)
C.to_csv(RES / "candidate_transforms.csv", index=False)

obs = {"ceiling": (conv.frac_max.min(), conv.frac_max.max()), "floor": (conv.frac_min.min(), conv.frac_min.max()),
       "occupancy": (conv.occupancy.min(), conv.occupancy.max())}
out = []
for name, d in C.groupby("candidate", sort=False):
    r = {"Candidate": name, "Volumes": len(d)}
    for m in ["ceiling", "floor", "occupancy"]:
        r[f"{m}, % median (range)"] = f"{100*d[m].median():.2f} ({100*d[m].min():.2f}–{100*d[m].max():.2f})"
    inside = ((d.ceiling >= obs["ceiling"][0]) & (d.ceiling <= obs["ceiling"][1]) & (d.floor >= obs["floor"][0]))
    r["Smallest floor share, % (observed 5.00)"] = f"{100*d.floor.min():.2f}"
    r["Within observed ceiling and floor envelope"] = f"{int(inside.sum())}/{len(d)}"
    out.append(r)
T = pd.DataFrame(out)
T.loc[len(T)] = {"Candidate": "Observed, converted release", "Volumes": len(conv),
                 "ceiling, % median (range)": f"{100*conv.frac_max.median():.2f} ({100*obs['ceiling'][0]:.2f}–{100*obs['ceiling'][1]:.2f})",
                 "floor, % median (range)": f"{100*conv.frac_min.median():.2f} ({100*obs['floor'][0]:.2f}–{100*obs['floor'][1]:.2f})",
                 "occupancy, % median (range)": f"{100*conv.occupancy.median():.2f} ({100*obs['occupancy'][0]:.2f}–{100*obs['occupancy'][1]:.2f})",
                 "Within observed ceiling and floor envelope": "—", "Smallest floor share, % (observed 5.00)": f"{100*obs['floor'][0]:.2f}"}
T.to_csv(RES / "tables" / "Table_candidate_transforms.csv", index=False)
print(T.to_string(index=False))
