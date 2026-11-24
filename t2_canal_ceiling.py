#!/usr/bin/env python
"""Step 2. Share of the whole canal label (label 100) at the stored maximum in every SPIDER T2 and SPACE volume.

Writes results/t2_canal_ceiling.csv (series, kind, enc, csf_clip_hi), used by run_clipping_pilot.py (Part A).
"""
import numpy as np
import pandas as pd
import SimpleITK as sitk

from paths import SPIDER, RESULTS

rows = []
for ip in sorted((SPIDER / "images").glob("*_t2*.mha")):
    series = ip.stem
    kind = "space" if "SPACE" in series.upper() else "t2"
    raw = sitk.GetArrayFromImage(sitk.ReadImage(str(ip))).astype(np.float64)
    lab = sitk.GetArrayFromImage(sitk.ReadImage(str(SPIDER / "masks" / ip.name)))
    lo, hi = float(raw.min()), float(raw.max())
    cv = raw[lab == 100]
    rows.append({"series": series, "kind": kind, "enc": "-1000_3096" if (lo == -1000 and hi == 3096) else "other",
                 "csf_clip_hi": float((cv == hi).mean()) if cv.size else np.nan})
pd.DataFrame(rows).to_csv(RESULTS / "t2_canal_ceiling.csv", index=False)
print(len(rows), "series")
