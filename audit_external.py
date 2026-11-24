#!/usr/bin/env python
"""External audit of ceiling pile-up in native-DICOM lumbar MRI.

RSNA 2024 Lumbar Spine Degenerative Classification (train set, local copy):
  per series: stored max, fraction of voxels at the series max / min, BitsStored, vendor, field strength;
  for Sagittal T2/STIR also a canal ROI (disc of radius R_MM around each labelled 'Spinal Canal Stenosis'
  point on its instance) and the fraction of ROI pixels at the series max.
AISSLab sagittal lumbar dataset (local copy): series-level audit only.
Usage: py -3 audit_external.py [n_rsna_studies]   (0 = all)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom

from paths import RSNA
from paths import AISS
from paths import RESULTS as OUT
R_MM = 2.5
SEED = 20261123


def read_series(files):
    arrs, meta = {}, None
    for f in files:
        try:
            d = pydicom.dcmread(str(f))
            a = d.pixel_array.astype(np.float64)
        except Exception:
            continue
        slope = float(getattr(d, "RescaleSlope", 1) or 1); icpt = float(getattr(d, "RescaleIntercept", 0) or 0)
        arrs[int(getattr(d, "InstanceNumber", len(arrs)))] = (a, d)
        if meta is None:
            meta = {"bits_stored": int(getattr(d, "BitsStored", -1)), "bits_alloc": int(getattr(d, "BitsAllocated", -1)),
                    "pixel_rep": int(getattr(d, "PixelRepresentation", -1)), "slope": slope, "intercept": icpt,
                    "manufacturer": str(getattr(d, "Manufacturer", "")), "model": str(getattr(d, "ManufacturerModelName", "")),
                    "field_T": float(getattr(d, "MagneticFieldStrength", np.nan) or np.nan)}
    return arrs, meta


def series_stats(arrs):
    allv = np.concatenate([a.ravel() for a, _ in arrs.values()])
    mx, mn = allv.max(), allv.min()
    fg = allv > np.quantile(allv, 0.5)                      # crude foreground: upper half of voxels
    return mx, mn, {"n_vox": int(allv.size), "max": float(mx), "min": float(mn),
                    "frac_at_max": float((allv == mx).mean()), "frac_at_min": float((allv == mn).mean()),
                    "frac_fg_at_max": float((allv[fg] == mx).mean()) if fg.any() else np.nan,
                    "p999": float(np.quantile(allv, 0.999)), "n_distinct_top": int(np.unique(allv[allv >= np.quantile(allv, 0.999)]).size)}


def rsna(n):
    desc = pd.read_csv(RSNA / "train_series_descriptions.csv")
    coords = pd.read_csv(RSNA / "train_label_coordinates.csv")
    canal = coords[coords.condition == "Spinal Canal Stenosis"]
    studies = desc.study_id.unique()
    if n:
        studies = np.random.default_rng(SEED).choice(studies, size=min(n, len(studies)), replace=False)
    rows = []
    for k, sid in enumerate(studies):
        for s in desc[desc.study_id == sid].itertuples():
            if not s.series_description.startswith("Sagittal"):
                continue
            files = sorted((RSNA / "train_images" / str(sid) / str(s.series_id)).glob("*.dcm"))
            if not files:
                continue
            arrs, meta = read_series(files)
            if not arrs:
                continue
            mx, mn, st = series_stats(arrs)
            r = {"dataset": "RSNA", "study": sid, "series": s.series_id, "desc": s.series_description, **meta, **st}
            pts = canal[canal.series_id == s.series_id]
            if len(pts):
                roi_vals = []
                for p in pts.itertuples():
                    if p.instance_number not in arrs:
                        continue
                    a, d = arrs[p.instance_number]
                    ps = [float(x) for x in getattr(d, "PixelSpacing", [1, 1])]
                    yy, xx = np.ogrid[:a.shape[0], :a.shape[1]]
                    m = ((yy - p.y) * ps[0]) ** 2 + ((xx - p.x) * ps[1]) ** 2 <= R_MM ** 2
                    roi_vals.append(a[m])
                if roi_vals:
                    v = np.concatenate(roi_vals)
                    r.update(canal_roi_n=int(v.size), canal_roi_frac_at_max=float((v == mx).mean()),
                             canal_roi_median=float(np.median(v)), canal_roi_p95=float(np.quantile(v, .95)))
            rows.append(r)
        if k % 50 == 0:
            print(f"RSNA {k}/{len(studies)}", flush=True)
    return pd.DataFrame(rows)


def aiss():
    rows = []
    for sd in sorted(p for p in AISS.iterdir() if p.is_dir()):
        files = sorted(sd.rglob("*.dcm"))
        groups = {}
        for f in files:
            try:
                d = pydicom.dcmread(str(f), stop_before_pixels=True)
                groups.setdefault(str(getattr(d, "SeriesInstanceUID", "x")), []).append(f)
            except Exception:
                pass
        for uid, fs in groups.items():
            arrs, meta = read_series(fs)
            if not arrs:
                continue
            d0 = next(iter(arrs.values()))[1]
            mx, mn, st = series_stats(arrs)
            rows.append({"dataset": "AISSLab", "study": sd.name, "series": uid, "desc": str(getattr(d0, "SeriesDescription", "")),
                         **meta, **st})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    A = aiss(); A.to_csv(OUT / "external_aisslab_series.csv", index=False)
    print("AISSLab series:", len(A))
    R = rsna(n); R.to_csv(OUT / f"external_rsna_series_n{n}.csv", index=False)
    for name, df in [("AISSLab", A), ("RSNA", R)]:
        print(f"\n== {name}: {len(df)} series")
        print(df.groupby("desc").agg(n=("series", "size"), frac_at_max_median=("frac_at_max", "median"),
                                     frac_fg_at_max_p95=("frac_fg_at_max", lambda v: v.quantile(.95)),
                                     n_pileup_gt_0_1pct=("frac_fg_at_max", lambda v: int((v > 0.001).sum()))).to_string())
        print(df.bits_stored.value_counts().to_string())
        print(df.groupby("manufacturer").frac_fg_at_max.agg(["size", "median", lambda v: int((v > 0.001).sum())]).to_string())
    if "canal_roi_frac_at_max" in R:
        c = R.dropna(subset=["canal_roi_frac_at_max"])
        print("\nRSNA canal ROI:", len(c), "series; regimes:")
        print(pd.cut(c.canal_roi_frac_at_max, [-1, 0, .005, .05, .5, 1]).value_counts().sort_index().to_string())
