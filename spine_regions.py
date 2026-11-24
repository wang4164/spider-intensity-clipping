"""Regions of interest on SPIDER T1-weighted volumes, as described in Methods 2.4.

* additive intensity baseline: the dominant value on the volume border when it lies within 2 units of the
  global minimum and covers at least 5% of border voxels (-1000 in the converted encoding); otherwise the
  global minimum if it covers at least 1% of all voxels;
* vertebral body: the SPIDER whole-vertebra label anterior to the spinal canal minus a 1-mm margin, where the
  anterior canal edge is taken per superior-inferior index (10th/90th percentile of canal AP indices) and
  interpolated between indices, keeping the largest connected component;
* trabecular region: the body eroded by 2 mm (Euclidean distance in mm);
* T1 reference: the canal label within the superior-inferior extent of the L3 body plus 3 mm, eroded by
  1 mm, restricted to the lower 60% of its baseline-corrected intensities (fluid is dark on T1).
Levels: SPIDER numbers full vertebrae bottom-up; label 1 is taken as L5.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, label

CANAL_LABEL = 100
LEVEL_OF_LABEL = {1: "L5", 2: "L4", 3: "L3", 4: "L2", 5: "L1"}
LEVELS = ["L1", "L2", "L3", "L4"]


@dataclass(frozen=True)
class Axes:
    lr_axis: int
    ap_axis: int
    si_axis: int
    lr_sign: float
    ap_sign: float
    si_sign: float


def infer_anatomical_axes(direction_xyz) -> Axes:
    """numpy axes (z, y, x order) aligned with L/R, A/P and S/I, from a SimpleITK direction matrix (LPS)."""
    d = np.asarray(direction_xyz, dtype=float).reshape(3, 3)
    image_axes = [int(np.argmax(np.abs(d[p, :]))) for p in range(3)]
    if len(set(image_axes)) != 3:
        image_axes = list(max(itertools.permutations(range(3)), key=lambda perm: sum(abs(d[p, perm[p]]) for p in range(3))))
    np_axes = [2 - j for j in image_axes]
    signs = [float(np.sign(d[p, image_axes[p]]) or 1.0) for p in range(3)]
    return Axes(np_axes[0], np_axes[1], np_axes[2], signs[0], signs[1], signs[2])


def _mode(values):
    x = np.rint(np.asarray(values)[np.isfinite(values)]).astype(np.int64)
    lo = int(x.min())
    counts = np.bincount(x - lo)
    j = int(np.argmax(counts))
    return float(lo + j), float(counts[j] / x.size)


def intensity_baseline(raw: np.ndarray, border_vox: int = 2) -> float:
    a = np.asarray(raw)
    b = np.zeros(a.shape, bool)
    for ax, n in enumerate(a.shape):
        w = min(border_vox, max(1, n // 2))
        lo = [slice(None)] * a.ndim; hi = [slice(None)] * a.ndim
        lo[ax] = slice(0, w); hi[ax] = slice(n - w, n)
        b[tuple(lo)] = True; b[tuple(hi)] = True
    raw_min = float(np.nanmin(a))
    mode, frac = _mode(a[b])
    if frac >= 0.05 and abs(mode - raw_min) <= 2.0:
        return mode
    if float(np.mean(a == raw_min)) >= 0.01:
        return raw_min
    return float("nan")


def _largest_component(mask):
    lab, n = label(mask)
    if n <= 1:
        return mask.astype(bool)
    counts = np.bincount(lab.ravel()); counts[0] = 0
    return lab == counts.argmax()


def vertebral_body(vertebra, canal, axes: Axes, spacing_zyx, margin_mm=1.0, min_voxels=100):
    v, c = vertebra.astype(bool), canal.astype(bool)
    if not v.any() or not c.any():
        return np.zeros_like(v)
    si, ap = axes.si_axis, axes.ap_axis
    si_idx = np.where(v)[si]
    lo, hi = int(si_idx.min()), int(si_idx.max())
    ap2 = ap - (1 if si < ap else 0)
    margin = max(0, int(round(margin_mm / spacing_zyx[ap])))
    q = 0.10 if axes.ap_sign > 0 else 0.90
    th = {}
    for s in range(lo, hi + 1):
        sl = [slice(None)] * 3; sl[si] = s
        cs = c[tuple(sl)]
        if cs.any():
            th[s] = float(np.quantile(np.where(cs)[ap2], q))
    if not th:
        return np.zeros_like(v)
    ks = np.array(sorted(th), float); kt = np.array([th[int(s)] for s in ks])
    out = np.zeros_like(v)
    for s in range(lo, hi + 1):
        t = float(np.interp(s, ks, kt))
        sl = [slice(None)] * 3; sl[si] = s
        vs = v[tuple(sl)]
        coords = np.indices(vs.shape)[ap2]
        keep = coords <= (t - margin) if axes.ap_sign > 0 else coords >= (t + margin)
        out[tuple(sl)] = vs & keep
    out = _largest_component(out)
    return out if out.sum() >= min_voxels else np.zeros_like(v)


def eroded(mask, spacing_zyx, mm, min_voxels=100):
    roi = distance_transform_edt(mask.astype(bool), sampling=spacing_zyx) >= float(mm)
    return roi if roi.sum() >= min_voxels else np.zeros_like(roi)


def t1_canal_reference(corrected, canal, l3_body, axes: Axes, spacing_zyx, si_margin_mm=3.0, erosion_mm=1.0,
                       lower_fraction=0.60, min_voxels=50):
    """Returns (reference median, reference mask)."""
    si = axes.si_axis
    idx = np.where(l3_body)[si]
    m = int(round(si_margin_mm / spacing_zyx[si]))
    sl = [slice(None)] * 3; sl[si] = slice(max(0, int(idx.min()) - m), min(corrected.shape[si], int(idx.max()) + m + 1))
    cand = np.zeros_like(canal, bool); cand[tuple(sl)] = canal.astype(bool)[tuple(sl)]
    if erosion_mm > 0:
        cand = distance_transform_edt(cand, sampling=spacing_zyx) >= float(erosion_mm)
    vals = corrected[cand]
    if vals.size == 0:
        return float("nan"), np.zeros_like(cand)
    q = float(np.quantile(vals, lower_fraction))
    fluid = cand & (corrected <= q)
    fv = corrected[fluid]
    if fv.size < min_voxels or np.median(fv) <= 0:
        return float("nan"), fluid
    return float(np.median(fv)), fluid
