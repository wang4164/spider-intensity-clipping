#!/usr/bin/env python
"""Series description retained in the anonymised DICOM headers of RSNA sagittal T2/STIR series.

Covers every Sagittal T2/STIR series of a study with spinal-canal annotations (1974 series). Writes
results/rsna_t2_seriesdesc.csv (series, desc) and results/rsna_t2_group.csv (series, grp) for the 1973 series
with canal points on the T2/STIR series itself; grp is T2, STIR or unlabelled (no description).
"""
import os

import pandas as pd
import pydicom

from paths import RSNA, RESULTS

d = pd.read_csv(RSNA / "train_series_descriptions.csv")
c = pd.read_csv(RSNA / "train_label_coordinates.csv")
c = c[c.condition == "Spinal Canal Stenosis"]
t2 = d[(d.series_description == "Sagittal T2/STIR") & d.study_id.isin(c.study_id)]
rows = []
for s in t2.itertuples():
    folder = RSNA / "train_images" / str(s.study_id) / str(s.series_id)
    f = sorted(os.listdir(folder))[0]
    h = pydicom.dcmread(folder / f, stop_before_pixels=True)
    rows.append({"series": s.series_id, "desc": str(getattr(h, "SeriesDescription", "") or "").strip()})
A = pd.DataFrame(rows)
A.to_csv(RESULTS / "rsna_t2_seriesdesc.csv", index=False)
G = A[A.series.isin(c.series_id)].copy()
G["grp"] = G.desc.map({"T2": "T2", "STIR": "STIR"}).fillna("unlabelled")
G[["series", "grp"]].to_csv(RESULTS / "rsna_t2_group.csv", index=False)
print(len(A), "series;", G.grp.value_counts().to_dict())
