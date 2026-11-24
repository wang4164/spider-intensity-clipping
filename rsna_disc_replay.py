#!/usr/bin/env python
"""External replay on RSNA: disc-to-CSF ratio on native MRI vs the same image after the inferred transform.

For each RSNA sagittal T2/STIR series with annotated canal points:
  * disc ROI: SpineNetV2 (unchanged; https://github.com/rwindsor1/SpineNet) detects vertebral bodies; the intervertebral disc quadrilateral is
    built exactly as SpineNet does (lower corners of the upper body, upper corners of the lower body, on the mid slice of
    the two bodies) and shrunk by 40% towards its centroid to exclude endplates and the outer annulus;
  * CSF ROI: union of 2.5-mm discs around every annotated canal point on its annotated slice (as in the RSNA replay);
  * transform: y = clip(round(k (x - p5)), 0, 4096), k = 4096 / (p95 - p5), percentiles over the whole series;
  * ROIs are fixed in native coordinates; ratio = median(disc) / median(CSF), native, linear-only and transformed.
Series type (plain T2 vs fat-suppressed) is taken from results/rsna_t2_type.csv (rsna_t2_type_screen.py).
Run in an environment with SpineNet installed:  python rsna_disc_replay.py [--limit n] [--qc k]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pydicom
import torch

from paths import SPINENET, RSNA as _RSNA, RESULTS as _RES
SN = str(SPINENET)
import numpy.core.multiarray as _npcm  # noqa: E402
torch.serialization.add_safe_globals([(_npcm.scalar, "numpy.core.multiarray.scalar"), np.dtype, type(np.dtype("float64")),
                                      type(np.dtype("float32")), type(np.dtype("int64"))])
sys.path.insert(0, SN)
from spinenet import SpineNet  # noqa: E402
from spinenet.utils.gen_utils import vert_dicts_to_classification_format, get_ivd_level_names  # noqa: E402

RAW = str(_RSNA)
OUT = str(_RES)
SHRINK = 0.6
R_MM = 2.5


def load_series(folder):
    ds = []
    for f in os.listdir(folder):
        try:
            d = pydicom.dcmread(os.path.join(folder, f))
            if "PixelData" in d:
                ds.append(d)
        except Exception:
            pass
    shp = pd.Series([d.pixel_array.shape for d in ds]).value_counts().index[0]
    ds = [d for d in ds if d.pixel_array.shape == shp]
    o = np.array(ds[0].ImageOrientationPatient, float); n = np.cross(o[:3], o[3:])
    pos = np.array([np.dot(np.array(d.ImagePositionPatient, float), n) * np.sign(n[0] or 1) for d in ds])
    order = np.argsort(-pos)
    vol = np.stack([ds[i].pixel_array.astype(float) for i in order], axis=-1)
    inst = [int(ds[i].InstanceNumber) for i in order]
    ps = np.array(ds[order[0]].PixelSpacing, float)
    return vol, ps, inst


def in_quad(h, w, xs, ys):
    """Boolean mask of pixels inside the quadrilateral (xs = columns, ys = rows); corners are sorted by angle around
    the centroid first, so that a crossed corner order cannot produce a bow-tie."""
    from matplotlib.path import Path as MPath
    cx, cy = np.mean(xs), np.mean(ys)
    o = np.argsort(np.arctan2(ys - cy, xs - cx))
    yy, xx = np.mgrid[:h, :w]
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return MPath(np.column_stack([xs[o], ys[o]])).contains_points(pts).reshape(h, w)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--qc", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0, help="stop after this many new series (0 = all remaining)")
    ap.add_argument("--chunk", type=int, default=50, help="append results to disk every this many series")
    a = ap.parse_args()
    tag = f"_n{a.limit}" if a.limit else ""
    f_rows = os.path.join(OUT, f"rsna_disc_replay{tag}.csv")
    f_log = os.path.join(OUT, f"rsna_disc_replay_log{tag}.csv")
    done = set(pd.read_csv(f_log).series.astype(int)) if os.path.exists(f_log) else set()
    desc = pd.read_csv(os.path.join(RAW, "train_series_descriptions.csv"))
    coords = pd.read_csv(os.path.join(RAW, "train_label_coordinates.csv"))
    canal = coords[coords.condition == "Spinal Canal Stenosis"]
    fat = pd.read_csv(os.path.join(OUT, "rsna_t2_type.csv"), dtype={"t2_series": str})
    t2type = dict(zip(fat.t2_series.astype(str), fat.t2_type))
    t2 = desc[desc.series_description == "Sagittal T2/STIR"]
    t2 = t2[t2.series_id.isin(canal.series_id)]
    if a.limit:
        t2 = t2.sample(n=min(a.limit, len(t2)), random_state=20261124)
    todo = t2[~t2.series_id.isin(done)]
    if a.batch:
        todo = todo.head(a.batch)
    print(f"{len(done)} series already done; {len(todo)} to run now; {len(t2) - len(done) - len(todo)} left after this run", flush=True)
    spnt = SpineNet(device="cuda:0", verbose=False)
    rows, log, qc = [], [], []
    n_rows_total, n_err_total = 0, 0

    def flush():
        nonlocal rows, log, n_rows_total, n_err_total
        if rows:
            pd.DataFrame(rows).to_csv(f_rows, mode="a", header=not os.path.exists(f_rows), index=False)
        if log:
            pd.DataFrame(log).to_csv(f_log, mode="a", header=not os.path.exists(f_log), index=False)
        n_rows_total += len(rows); n_err_total += sum(1 for l in log if l.get("error"))
        rows, log = [], []

    t0 = time.time()
    for k, s in enumerate(todo.itertuples()):
        try:
            vol, ps, inst = load_series(os.path.join(RAW, "train_images", str(s.study_id), str(s.series_id)))
            h, w, ns = vol.shape
            pts = canal[canal.series_id == s.series_id]
            csf = np.zeros(vol.shape, bool)
            for p in pts.itertuples():
                if p.instance_number in inst:
                    yy, xx = np.ogrid[:h, :w]
                    csf[:, :, inst.index(p.instance_number)] |= ((yy - p.y) * ps[0]) ** 2 + ((xx - p.x) * ps[1]) ** 2 <= R_MM ** 2
            vd = spnt.detect_vb(vol, float(ps.mean()))
            ax, ay, amid, _alab, vb_names = vert_dicts_to_classification_format(vd, ns)
            levels = get_ivd_level_names(vb_names)
            p5, p95 = np.quantile(vol, [0.05, 0.95]); kf = 4096 / (p95 - p5)
            lin = kf * (vol - p5); y = np.clip(np.round(lin), 0, 4096)
            c_nat, c_lin, c_y = np.median(vol[csf]), np.median(lin[csf]), np.median(y[csf])
            f_csf = float((y[csf] >= 4096).mean())
            for i, lev in enumerate(levels):
                mid = int(np.round(np.mean([amid[i], amid[i + 1]])))
                xs = np.array([ax[1, i + 1, mid], ax[0, i, mid], ax[3, i, mid], ax[2, i + 1, mid]], float)
                ys = np.array([ay[1, i + 1, mid], ay[0, i, mid], ay[3, i, mid], ay[2, i + 1, mid]], float)
                cx, cy = xs.mean(), ys.mean()
                xs2, ys2 = cx + SHRINK * (xs - cx), cy + SHRINK * (ys - cy)
                m2 = in_quad(h, w, xs2, ys2)
                if m2.sum() < 15:
                    continue
                dv, dl, dy = vol[:, :, mid][m2], lin[:, :, mid][m2], y[:, :, mid][m2]
                # match to the annotated canal point nearest in the cranio-caudal direction (within 10 mm), not by label
                dist = (pts.y - cy).abs() * ps[0]
                if dist.empty or dist.min() > 10:
                    continue
                cp = pts.loc[[dist.idxmin()]]
                lv = cp.level.iloc[0]
                if ((rows and rows[-1]["series"] == s.series_id and any(r["level"] == lv for r in rows if r["series"] == s.series_id))):
                    continue
                rows.append({"study": s.study_id, "series": s.series_id, "t2_type": t2type.get(str(s.series_id), "unknown"),
                             "level": lv, "slice": mid, "disc_n": int(m2.sum()), "disc_row": cy, "disc_col": cx,
                             "canal_row": float(cp.y.iloc[0]) if len(cp) else np.nan, "canal_col": float(cp.x.iloc[0]) if len(cp) else np.nan,
                             "ps": float(ps.mean()), "csf_clipped": f_csf, "disc_clipped": float((dy >= 4096).mean()),
                             "R0": np.median(dv) / c_nat, "R1": np.median(dl) / c_lin, "R2": np.median(dy) / c_y})
                if len(qc) < a.qc and lv == "L4/L5":
                    qc.append((vol[:, :, mid], xs2, ys2, cp))
            log.append({"series": s.series_id, "n_vb": len(vd), "n_ivd": len(levels), "error": ""})
        except Exception as e:
            log.append({"series": s.series_id, "error": repr(e)[:200]})
        if (k + 1) % a.chunk == 0:
            flush()
            print(f"{k + 1}/{len(todo)} saved, {time.time() - t0:.0f} s", flush=True)
    flush()
    if qc:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(qc), figsize=(3 * len(qc), 3))
        for ax_, (im, xs2, ys2, cp) in zip(np.atleast_1d(axs), qc):
            ax_.imshow(im, cmap="gray"); ax_.plot(list(xs2) + [xs2[0]], list(ys2) + [ys2[0]], "y-", lw=0.8)
            if len(cp):
                ax_.plot(cp.x.iloc[0], cp.y.iloc[0], "r+", ms=8)
            ax_.axis("off")
        fig.savefig(os.path.join(OUT, f"rsna_disc_qc{tag}.png"), dpi=110, bbox_inches="tight")
    left = len(t2) - len(done) - len(todo)
    print(f"done {len(todo)} series in {time.time() - t0:.0f} s; discs {n_rows_total}; errors {n_err_total}; "
          f"{len(done) + len(todo)}/{len(t2)} series complete, {left} left", flush=True)


if __name__ == "__main__":
    main()
