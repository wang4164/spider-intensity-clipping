#!/usr/bin/env python
"""How much do CSF censoring and ratio bias depend on the CSF ROI definition? (SPIDER T2 and T2 SPACE)

CSF ROIs (all inside the canal label 100):
  whole  - the full canal label
  ero1   - eroded 1 mm (as in forward_spider_native.py)
  ero2   - eroded 2 mm
  core   - ero1 restricted to its upper 60% of intensities (drops darker cauda equina / nerve roots; on T2 CSF is bright)
Converted series: observed fraction of ROI voxels at the stored ceiling (3096).
Native series: forward transform (per-volume p5-p95 clip, rescale to 4096 levels), then clip-only error of the CSF
estimate and total error of disc/CSF ratios (discs = labels 201-209 eroded 1 mm), naive median vs quantile-normal CSF.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, find_objects

sys.path.insert(0, str(Path(__file__).parent))
from run_clipping_pilot import quantile_normal  # noqa: E402

from paths import SPIDER as ROOT, RESULTS, T1_REGIONS
RES = RESULTS
ROIS = ["whole", "ero1", "ero2", "core"]


def erode(mask, sp, mm):
    """Erosion by Euclidean distance, computed on the bounding box (with 1-voxel pad) for speed."""
    out = np.zeros_like(mask, dtype=bool)
    sl = find_objects(mask.astype(np.uint8))
    if not sl or sl[0] is None:
        return out
    box = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl[0])
    out[box] = distance_transform_edt(mask[box], sampling=sp) >= mm
    return out


def csf_rois(lab, x, sp):
    canal = lab == 100
    e1 = erode(canal, sp, 1.0)
    e2 = erode(canal, sp, 2.0)
    core = e1 & (x >= np.quantile(x[e1], 0.40)) if e1.sum() > 50 else e1
    return {"whole": canal, "ero1": e1, "ero2": e2, "core": core}


def run_one(fn):
    img = sitk.ReadImage(str(ROOT / "images" / f"{fn}.mha"))
    x = sitk.GetArrayFromImage(img).astype(np.float64)
    lab = sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT / "masks" / f"{fn}.mha")))
    sp = img.GetSpacing()[::-1]
    conv = x.min() == -1000 and x.max() == 3096
    rois = csf_rois(lab, x, sp)
    r = {"series": fn, "kind": "SPACE" if "SPACE" in fn else "t2", "converted": conv}
    if conv:
        for k, m in rois.items():
            r[f"f_{k}"] = float((x[m] >= 3096).mean()) if m.sum() > 50 else np.nan
        return r
    p5, p95 = np.quantile(x, [0.05, 0.95]); kf = 4096 / (p95 - p5)
    lin = kf * (x - p5); y = np.clip(np.round(lin), 0, 4096)
    discs = []
    for L in [int(z) for z in np.unique(lab) if 200 < z < 300]:
        d = erode(lab == L, sp, 1.0)
        if d.sum() >= 20:
            discs.append(d)
    for k, m in rois.items():
        if m.sum() <= 50:
            continue
        yv = y[m]; f = float((yv >= 4096).mean())
        c_nat, c_lin = float(np.median(x[m])), float(np.median(lin[m]))
        c_med = float(np.median(yv)); c_qn = quantile_normal(yv, f) if f > 0 else c_med
        r.update({f"f_{k}": f, f"csf_err_median_{k}": c_med / c_lin - 1, f"csf_err_mean_{k}": float(np.mean(yv)) / float(np.mean(lin[m])) - 1,
                  f"csf_err_qn_{k}": c_qn / c_lin - 1})
        if discs:
            nat = np.array([np.median(x[d]) / c_nat for d in discs])
            r[f"ratio_err_median_{k}"] = float(np.median(np.array([np.median(y[d]) / c_med for d in discs]) / nat - 1))
            r[f"ratio_err_qn_{k}"] = float(np.median(np.array([np.median(y[d]) / c_qn for d in discs]) / nat - 1))
    return r


if __name__ == "__main__":
    files = sorted(p.stem for p in (ROOT / "images").glob("*_t2*.mha"))
    rows = []
    for i, fn in enumerate(files):
        try:
            rows.append(run_one(fn))
        except Exception as e:
            print("fail", fn, e)
        if i % 40 == 0:
            print(f"{i}/{len(files)}", flush=True)
    D = pd.DataFrame(rows); D.to_csv(RES / "csf_roi_sensitivity.csv", index=False)
    cut = ([-1, 0, .05, .5, 1], ["0", "0-5%", "5-50%", ">50%"])
    q = lambda s: f"{100*s.median():+.1f} [{100*s.quantile(.25):+.1f}, {100*s.quantile(.75):+.1f}] (min {100*s.min():+.1f}, max {100*s.max():+.1f})"
    for (kind, conv), d in D.groupby(["kind", "converted"]):
        print(f"\n== {kind}, {'converted (observed)' if conv else 'native (forward)'}: {len(d)} series")
        for k in ROIS:
            c = d[f"f_{k}"].dropna()
            print(f"  {k:5s} regimes {pd.cut(c, *cut).value_counts().sort_index().to_dict()}  censored {100*(c > .5).mean():.0f}%")
        if not conv:
            for k in ROIS:
                if f"ratio_err_median_{k}" in d:
                    print(f"  {k:5s} CSF median {q(d[f'csf_err_median_{k}'].dropna())} | mean {q(d[f'csf_err_mean_{k}'].dropna())}")
                    print(f"        ratio total, naive {q(d[f'ratio_err_median_{k}'].dropna())} | quantile-normal {q(d[f'ratio_err_qn_{k}'].dropna())}")
