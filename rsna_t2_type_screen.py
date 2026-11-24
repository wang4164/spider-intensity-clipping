#!/usr/bin/env python
"""Classify RSNA sagittal T2/STIR series as plain T2 or fat-suppressed from the images (Methods 2.6).

The anonymised headers keep no inversion time, and the series description is 'T2' or empty. On the mid-sagittal
slice the brightest 5% of T1-weighted pixels are mostly fat (subcutaneous, epidural, marrow). They are mapped to
the matched T2/STIR slice through the DICOM geometry, and the median within-image percentile of their T2 signal
is computed: fat stays bright on plain T2 and goes dark after fat suppression.
Classes, set by visual review of 24 studies across the score range: plain T2 >= 90; fat-suppressed < 87;
indeterminate 87-90 (excluded). Writes results/rsna_t2_type.csv.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pydicom

from paths import RSNA, RESULTS


def load_series(study, series):
    d = os.path.join(RSNA, "train_images", str(study), str(series))
    hs = sorted((pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True) for f in os.listdir(d)),
                key=lambda h: int(h.InstanceNumber))
    return d, hs


def affine(h):
    iop = np.array(h.ImageOrientationPatient, float)
    dr, dc = map(float, h.PixelSpacing)
    return np.array(h.ImagePositionPatient, float), iop[:3], iop[3:], dr, dc


def one(args):
    study, s1, s2 = args
    try:
        d1, h1 = load_series(study, s1)
        d2, h2 = load_series(study, s2)
        n = np.cross(*affine(h1[0])[1:3])
        pos1 = np.array([np.dot(affine(h)[0], n) for h in h1])
        pos2 = np.array([np.dot(affine(h)[0], n) for h in h2])
        mid = float(np.median(pos2))
        k1, k2 = int(np.argmin(abs(pos1 - mid))), int(np.argmin(abs(pos2 - mid)))
        a1 = pydicom.dcmread(os.path.join(d1, f"{h1[k1].InstanceNumber}.dcm")).pixel_array.astype(float)
        a2 = pydicom.dcmread(os.path.join(d2, f"{h2[k2].InstanceNumber}.dcm")).pixel_array.astype(float)
        ipp1, r1, c1, dr1, dc1 = affine(h1[k1])
        ipp2, r2, c2, dr2, dc2 = affine(h2[k2])
        thr = np.percentile(a1, 95)
        ys, xs = np.nonzero(a1 >= thr)
        if len(ys) > 4000:
            sel = np.random.default_rng(0).choice(len(ys), 4000, replace=False)
            ys, xs = ys[sel], xs[sel]
        P = ipp1 + np.outer(xs * dc1, r1) + np.outer(ys * dr1, c1)
        x2 = np.rint((P - ipp2) @ r2 / dc2).astype(int)
        y2 = np.rint((P - ipp2) @ c2 / dr2).astype(int)
        ok = (x2 >= 0) & (x2 < a2.shape[1]) & (y2 >= 0) & (y2 < a2.shape[0])
        v = a2[y2[ok], x2[ok]]
        pct = np.searchsorted(np.sort(a2.ravel()), v) / a2.size * 100
        return dict(study_id=study, t1_series=s1, t2_series=s2, fat_pct_on_t2=float(np.median(pct)), n_px=int(ok.sum()),
                    plane_offset_mm=float(abs(pos1[k1] - pos2[k2])))
    except Exception as e:  # noqa: BLE001
        return dict(study_id=study, t1_series=s1, t2_series=s2, error=str(e)[:150])


if __name__ == "__main__":
    desc = pd.read_csv(RSNA / "train_series_descriptions.csv")
    s1 = desc[desc.series_description == "Sagittal T1"].groupby("study_id").series_id.first()
    s2 = desc[desc.series_description == "Sagittal T2/STIR"].groupby("study_id").series_id.first()
    jobs = [(st, int(s1[st]), int(s2[st])) for st in sorted(set(s1.index) & set(s2.index))]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        df = pd.DataFrame(list(ex.map(one, jobs, chunksize=8)))
    f = df.fat_pct_on_t2
    df["t2_type"] = np.select([f >= 90, f < 87, f.notna()], ["plain_T2", "fat_suppressed", "indeterminate"], "failed")
    df.to_csv(RESULTS / "rsna_t2_type.csv", index=False)
    print(len(df), "studies;", df.t2_type.value_counts().to_dict())
