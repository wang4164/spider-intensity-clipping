#!/usr/bin/env python
"""Apply the inferred SPIDER conversion to native RSNA sagittal T2 series and measure what it does to a CSF ROI.

Inferred transform (spider_encoding_probe.csv: all 374 converted volumes have 5.0-5.1% of voxels at the ceiling
and >=5% at the floor):  y = clip((x - p5) / (p95 - p5), 0, 1) * 4096 - 1000, with p5/p95 over the whole volume.
After the usual baseline correction (+1000):  y' = clip(k (x - p5), 0, 4096),  k = 4096 / (p95 - p5).

Per series (canal ROI = 2.5-mm discs around the labelled 'Spinal Canal Stenosis' points):
  * reproduction check: fraction of volume voxels at the ceiling, occupancy of the integer lattice;
  * fraction of ROI pixels clipped; regime counts compared with SPIDER;
  * clipping-only error of each estimator: estimate(clipped) / estimate(unclipped linear map) - 1;
  * offset term p5 / median(ROI) (the p5 subtraction biases any ratio unless p5 ~ 0).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom

sys.path.insert(0, str(Path(__file__).parent))
from run_clipping_pilot import estimates  # noqa: E402

from paths import RSNA
from paths import RESULTS as OUT
R_MM = 2.5
SEED = 20261123


def load(sid, ser):
    sl = {}
    for f in sorted((RSNA / "train_images" / str(sid) / str(ser)).glob("*.dcm")):
        try:
            d = pydicom.dcmread(str(f)); sl[int(d.InstanceNumber)] = (d.pixel_array.astype(np.float64), d)
        except Exception:
            pass
    return sl


def main(n):
    desc = pd.read_csv(RSNA / "train_series_descriptions.csv")
    coords = pd.read_csv(RSNA / "train_label_coordinates.csv")
    canal = coords[coords.condition == "Spinal Canal Stenosis"]
    studies = desc.study_id.unique()
    if n:
        studies = np.random.default_rng(SEED).choice(studies, size=min(n, len(studies)), replace=False)
    t2 = desc[desc.study_id.isin(studies) & (desc.series_description == "Sagittal T2/STIR")]
    rows = []
    for i, s in enumerate(t2.itertuples()):
        pts = canal[canal.series_id == s.series_id]
        if pts.empty:
            continue
        sl = load(s.study_id, s.series_id)
        if not sl:
            continue
        vol = np.concatenate([a.ravel() for a, _ in sl.values()])
        p5, p95 = np.quantile(vol, [0.05, 0.95])
        if p95 <= p5:
            continue
        k = 4096.0 / (p95 - p5)
        yv = np.clip(np.round(k * (vol - p5)), 0, 4096)
        vals = []
        for p in pts.itertuples():
            if p.instance_number not in sl:
                continue
            a, d = sl[p.instance_number]
            ps = [float(x) for x in d.PixelSpacing]
            yy, xx = np.ogrid[:a.shape[0], :a.shape[1]]
            vals.append(a[((yy - p.y) * ps[0]) ** 2 + ((xx - p.x) * ps[1]) ** 2 <= R_MM ** 2])
        if not vals:
            continue
        v = np.concatenate(vals)
        lin = k * (v - p5)                                  # linear map, no clipping (reference)
        clp = np.clip(np.round(lin), 0, 4096)
        f = float((clp >= 4096).mean())
        r = {"study": s.study_id, "series": s.series_id, "n_roi": int(v.size), "p5": float(p5), "p95": float(p95), "k": k,
             "vol_frac_ceiling": float((yv >= 4096).mean()), "vol_occupancy": float(np.unique(yv).size / 4097),
             "roi_frac_clipped": f, "roi_native_median": float(np.median(v)), "offset_ratio": float(p5 / np.median(v)) if np.median(v) else np.nan}
        for q in (0.95, 0.98, 0.99, 0.995, 0.999):          # alternative upper clipping percentiles
            r[f"roi_frac_clipped_q{q}"] = float((v >= np.quantile(vol, q)).mean())
        e_lin = {"median": float(np.median(lin)), "mean": float(np.mean(lin))}
        e = estimates(clp, 4096.0, f) if f > 0 else {"median": float(np.median(clp)), "mean": float(np.mean(clp)),
                                                     "tobit_normal": float(np.median(clp)), "quantile_normal": float(np.median(clp))}
        for key in ["median", "mean", "tobit_normal", "quantile_normal"]:
            ref = e_lin["mean"] if key == "mean" else e_lin["median"]
            r[f"err_{key}"] = e[key] / ref - 1
        rows.append(r)
        if i % 50 == 0:
            print(f"{i}/{len(t2)}", flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / f"rsna_spider_transform_n{n}.csv", index=False)
    print(f"\nseries analysed: {len(R)}")
    print("reproduction: volume fraction at ceiling", R.vol_frac_ceiling.describe().round(4).to_dict())
    print("               lattice occupancy median", round(R.vol_occupancy.median(), 3))
    reg = pd.cut(R.roi_frac_clipped, [-1, 0, .005, .05, .5, 1], labels=["0", "0-0.5%", "0.5-5%", "5-50%", ">50%"]).value_counts().sort_index()
    print("\ncanal ROI clipping regimes (RSNA under SPIDER transform):\n" + reg.to_string())
    print("\nSPIDER observed (T2 canal, -1000..3096 series, n=188): 0:0 | 0-0.5%:3 | 0.5-5%:19 | 5-50%:133 | >50%:33")
    R["regime"] = pd.cut(R.roi_frac_clipped, [-1, 0, .05, .5, 1], labels=["0", "0-5%", "5-50%", ">50%"])
    g = R.groupby("regime", observed=True)[[c for c in R.columns if c.startswith("err_")]]
    print("\nclipping-only error by regime, % median [IQR]:")
    print(g.agg(lambda x: f"{100*x.median():+.1f} [{100*x.quantile(.25):+.1f}, {100*x.quantile(.75):+.1f}]").assign(n=R.groupby("regime", observed=True).size()).to_string())
    print("\noffset term p5/median(CSF):", R.offset_ratio.describe().round(3).to_dict())
    print("\nupper clipping percentile -> share of series whose CSF median is censored (>50% of ROI clipped) / any ROI clipping:")
    for q in (0.95, 0.98, 0.99, 0.995, 0.999):
        c = R[f"roi_frac_clipped_q{q}"]
        print(f"  p{100*q:g}: censored {100*(c > .5).mean():.1f}%  | >5% clipped {100*(c > .05).mean():.1f}%  | any {100*(c > 0).mean():.1f}%")
    print("\nby series type: STIR in description is not available; regimes for p95 by k-tertile:")
    print(pd.crosstab(pd.qcut(R.k, 3), R.regime).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
