#!/usr/bin/env python
"""Detector of ceiling pile-up across thresholds (Supplementary Table S3).

Positives: the 374 SPIDER volumes in the -1000..3096 encoding; negatives: 73 native SPIDER, 500 AISSLab and
3954 RSNA sagittal series. A volume is classified as clipped when the share of voxels at its stored maximum
exceeds the threshold. Reads spider_encoding_probe.csv, external_aisslab_series.csv, external_rsna_series_n0.csv.
"""
import numpy as np
import pandas as pd

from paths import RESULTS

P = pd.read_csv(RESULTS / "spider_encoding_probe.csv")
A = pd.read_csv(RESULTS / "external_aisslab_series.csv")
R = pd.read_csv(RESULTS / "external_rsna_series_n0.csv")
pos = P.loc[P["min"] < 0, "frac_max"].values
neg = np.concatenate([P.loc[P["min"] >= 0, "frac_max"].values, A.frac_at_max.values, R.frac_at_max.values])
rows = []
for t in [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 4]:
    rows.append({"threshold_%": t, "sensitivity": f"{int((pos > t / 100).sum())}/{len(pos)}",
                 "false_pos": f"{int((neg > t / 100).sum())}/{len(neg)}"})
T = pd.DataFrame(rows)
T.to_csv(RESULTS / "detector_threshold_sweep.csv", index=False)
print(f"positives {len(pos)} negatives {len(neg)}; positive min {pos.min():.5f}; negative max {neg.max():.3g}, "
      f"p99.9 {np.percentile(neg, 99.9):.3g}")
print(T.to_string(index=False))
