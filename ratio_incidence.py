#!/usr/bin/env python
"""Per-disc forward replay on SPIDER native T2 and SPACE volumes (responds to pre-submission review).

For every disc of every native volume and every CSF region definition (whole, ero1, ero2, core):
  * native ratio      R0 = median(disc) / median(CSF)                      on native values x
  * linear ratio      R1 = same on k(x - p5)                                (offset only)
  * released ratio    R2 = same on clip(round(k(x - p5)), 0, 4096)          (offset + clipping)
  * quantile-normal   R3 = median(disc) / quantile-normal CSF estimate       on the released values
  ROI coordinates are fixed from the native image (primary). For the core region a second variant
  re-selects the upper 60% on the released image ("reselected"), as a user of the release would.
Outputs per-disc errors with Pfirrmann grade -> results/ratio_incidence_discs.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent))
from run_clipping_pilot import quantile_normal  # noqa: E402
from csf_roi_sensitivity import erode  # noqa: E402

from paths import SPIDER as ROOT, RESULTS, T1_REGIONS
RES = RESULTS


def main():
    P = pd.read_csv(RES / "spider_encoding_probe.csv")
    nat = P[~((P["min"] == -1000) & (P["max"] == 3096)) & P.file.str.contains("_t2")].file.tolist()
    g = pd.read_csv(ROOT / "radiological_gradings.csv")
    g = g.rename(columns={"Patient": "pid", "IVD label": "ivd", "Pfirrman grade": "pfirrmann"})[["pid", "ivd", "pfirrmann"]]
    rows = []
    for fn in nat:
        img = sitk.ReadImage(str(ROOT / "images" / f"{fn}.mha"))
        x = sitk.GetArrayFromImage(img).astype(np.float64)
        lab = sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT / "masks" / f"{fn}.mha")))
        sp = img.GetSpacing()[::-1]
        p5, p95 = np.quantile(x, [0.05, 0.95]); k = 4096 / (p95 - p5)
        lin = k * (x - p5); y = np.clip(np.round(lin), 0, 4096)
        canal = lab == 100
        e1 = erode(canal, sp, 1.0); e2 = erode(canal, sp, 2.0)
        core = e1 & (x >= np.quantile(x[e1], 0.40))
        core_re = e1 & (y >= np.quantile(y[e1], 0.40))
        rois = {"whole": (canal, canal), "ero1": (e1, e1), "ero2": (e2, e2), "core": (core, core), "core_reselected": (core, core_re)}
        csf = {}
        for name, (m_nat, m_rel) in rois.items():
            if m_nat.sum() <= 50 or m_rel.sum() <= 50:
                continue
            yv = y[m_rel]; f = float((yv >= 4096).mean())
            csf[name] = dict(nat=np.median(x[m_nat]), lin=np.median(lin[m_nat]), rel=np.median(yv),
                             qn=quantile_normal(yv, f) if f > 0 else np.median(yv), f=f,
                             overlap=float((m_nat & m_rel).sum() / m_nat.sum()))
        pid = int(fn.split("_")[0])
        for L in [int(z) for z in np.unique(lab) if 200 < z < 300]:
            d = erode(lab == L, sp, 1.0)
            if d.sum() < 20:
                continue
            Dn, Dl, Dr = np.median(x[d]), np.median(lin[d]), np.median(y[d])
            for name, c in csf.items():
                R0 = Dn / c["nat"]
                rows.append({"series": fn, "kind": "SPACE" if "SPACE" in fn else "2D T2", "pid": pid, "ivd": L - 200, "roi": name,
                             "csf_clipped": c["f"], "roi_overlap": c["overlap"], "disc_clipped": float((y[d] >= 4096).mean()),
                             "p5_over_disc": (p5 - x.min()) / Dn if Dn else np.nan,
                             "err_offset": (Dl / c["lin"]) / R0 - 1, "err_total": (Dr / c["rel"]) / R0 - 1,
                             "err_total_qn": (Dr / c["qn"]) / R0 - 1, "ratio_native": R0})
        print(fn, flush=True)
    D = pd.DataFrame(rows).merge(g, on=["pid", "ivd"], how="left")
    D.to_csv(RES / "ratio_incidence_discs.csv", index=False)
    print(len(D), "disc-ROI rows")


if __name__ == "__main__":
    main()
