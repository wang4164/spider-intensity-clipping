#!/usr/bin/env python
"""Step 1. T1 vertebral marrow regions (L1-L4) and the L3 canal-fluid reference for every SPIDER T1 volume.

Writes results/t1_regions/t1_vertebrae.csv (one row per vertebra) and, per patient, the masks
results/t1_regions/masks/{pid}_{level}_vroi.mha and {pid}_L3_canal_reference.mha.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import SimpleITK as sitk

from paths import SPIDER, T1_REGIONS
from spine_regions import (CANAL_LABEL, LEVEL_OF_LABEL, LEVELS, eroded, infer_anatomical_axes, intensity_baseline,
                           t1_canal_reference, vertebral_body)

MASKS = T1_REGIONS / "masks"
MASKS.mkdir(parents=True, exist_ok=True)


def write_mask(ref_img, mask, path):
    out = sitk.GetImageFromArray(mask.astype(np.uint8))
    out.CopyInformation(ref_img)
    sitk.WriteImage(out, str(path))


def main():
    rows, skipped = [], []
    for ip in sorted((SPIDER / "images").glob("*_t1.mha")):
        pid = re.match(r"(.+)_t1$", ip.stem).group(1)
        mp = SPIDER / "masks" / ip.name
        if not mp.exists():
            skipped.append((pid, "no mask")); continue
        img, lab_img = sitk.ReadImage(str(ip)), sitk.ReadImage(str(mp))
        raw = sitk.GetArrayFromImage(img).astype(np.float32)
        lab = sitk.GetArrayFromImage(lab_img).astype(np.int16)
        if raw.shape != lab.shape:
            skipped.append((pid, "geometry")); continue
        sp = tuple(float(x) for x in img.GetSpacing())[::-1]
        base = intensity_baseline(raw)
        if not np.isfinite(base):
            skipped.append((pid, "baseline")); continue
        cor = raw - np.float32(base)
        axes = infer_anatomical_axes(img.GetDirection())
        canal = lab == CANAL_LABEL
        bodies = {}
        for vl, lev in LEVEL_OF_LABEL.items():
            if lev in LEVELS and (lab == vl).any():
                b = vertebral_body(lab == vl, canal, axes, sp)
                if b.any():
                    bodies[lev] = (vl, b)
        if "L3" not in bodies:
            skipped.append((pid, "no L3 body")); continue
        ref, ref_mask = t1_canal_reference(cor, canal, bodies["L3"][1], axes, sp)
        if not np.isfinite(ref) or ref == 0:
            skipped.append((pid, "reference")); continue
        write_mask(img, ref_mask, MASKS / f"{pid}_L3_canal_reference.mha")
        raw_max = float(raw.max())
        for lev in LEVELS:
            if lev not in bodies:
                continue
            vl, body = bodies[lev]
            vroi = eroded(body, sp, 2.0)
            if not vroi.any():
                continue
            write_mask(img, vroi, MASKS / f"{pid}_{lev}_vroi.mha")
            rows.append({"patient_id": pid, "vertebra_label": vl, "anatomic_level": lev, "intensity_baseline": base,
                         "reference_signal": ref, "signal_v": float(np.median(cor[vroi])),
                         "volume_roi_voxels": int(vroi.sum()),
                         "vertebral_high_clip_fraction": float(np.mean(np.isclose(raw[vroi], raw_max, rtol=0, atol=1e-6)))})
        print(pid, len(bodies), flush=True)
    pd.DataFrame(rows).to_csv(T1_REGIONS / "t1_vertebrae.csv", index=False)
    pd.DataFrame(skipped, columns=["patient_id", "reason"]).to_csv(T1_REGIONS / "skipped.csv", index=False)
    print(f"{len(rows)} vertebrae; {len(skipped)} patients skipped")


if __name__ == "__main__":
    main()
