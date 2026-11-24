#!/usr/bin/env python
"""Intensity-ceiling clipping in SPIDER and its effect on region estimates (simulation within SPIDER).

Part A  prevalence of ceiling pile-up by stored encoding (T1 vertebral vROI, T2 canal).
Part B  T1 simulation: unclipped vertebral vROIs (ground truth) are clipped at the ROI quantile that
        puts a fraction f of voxels at the ceiling; estimators are compared with their own unclipped value.
Part C  T2 simulation: same, on the eroded canal (CSF reference) of the unclipped 'other'-encoding series.
Part D  real clipped T1 vertebrae: naive vs censoring-aware estimates (descriptive only).

Estimators: median, mean, 5% trimmed mean (naive); right-censored normal and log-normal MLE (Tobit);
normal fit to two observed lower quantiles.  Outputs go to results/.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import optimize, stats
from scipy.ndimage import distance_transform_edt

from paths import SPIDER as ROOT, RESULTS, T1_REGIONS
OUT = RESULTS
OUT.mkdir(parents=True, exist_ok=True)
F_GRID = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


# ---------------------------------------------------------------- estimators
def tobit(x_obs: np.ndarray, n_cens: int, c: float, log: bool = False) -> float:
    """Right-censored normal (or log-normal) MLE; returns the estimated median."""
    if log:
        x_obs = np.log(np.clip(x_obs, 1e-6, None)); c = np.log(c)
    n_u = x_obs.size
    if n_cens == 0:
        m = float(np.median(x_obs))
        return float(np.exp(m)) if log else m
    s1, s2 = x_obs.sum(), (x_obs ** 2).sum()

    def nll(p):
        mu, ls = p; s = np.exp(ls)
        ss = (s2 - 2 * mu * s1 + n_u * mu ** 2) / s ** 2
        return 0.5 * ss + n_u * ls - n_cens * stats.norm.logsf((c - mu) / s)

    q = np.quantile(x_obs, [0.1, 0.5]) if n_u else np.array([c, c])
    p0 = [max(q[1], np.median(x_obs)), np.log(max(np.std(x_obs), 1e-3))]
    r = optimize.minimize(nll, p0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
    mu = float(r.x[0])
    return float(np.exp(mu)) if log else mu


def quantile_normal(x_clipped: np.ndarray, f: float) -> float:
    """Normal fit through two empirical quantiles that lie below the ceiling."""
    pa = 0.10
    pb = min(0.50, 0.9 * (1 - f))
    if pb <= pa + 0.02:
        pb = pa + 0.02
    qa, qb = np.quantile(x_clipped, [pa, pb])
    za, zb = stats.norm.ppf([pa, pb])
    sd = (qb - qa) / (zb - za)
    return float(qb - zb * sd)


def trimmed(x, k=0.05):
    return float(stats.trim_mean(x, k))


def estimates(xc: np.ndarray, c: float, f: float) -> dict:
    cens = xc >= c
    obs = xc[~cens]
    return {
        "median": float(np.median(xc)),
        "mean": float(np.mean(xc)),
        "trimmed_mean": trimmed(xc),
        "tobit_normal": tobit(obs, int(cens.sum()), c, log=False),
        "tobit_lognormal": tobit(obs, int(cens.sum()), c, log=True),
        "quantile_normal": quantile_normal(xc, f),
    }


def simulate(x: np.ndarray) -> list[dict]:
    truth = {"median": float(np.median(x)), "mean": float(np.mean(x)), "trimmed_mean": trimmed(x)}
    rows = []
    for f in F_GRID:
        c = float(np.quantile(x, 1 - f))
        xc = np.minimum(x, c)
        f_real = float((x >= c).mean())
        e = estimates(xc, c, f)
        r = {"f_target": f, "f_real": f_real, "skew": float(stats.skew(x)), "n_vox": int(x.size)}
        for k, v in e.items():
            ref = truth.get(k, truth["median"])           # censoring-aware estimators target the median
            r[f"err_{k}"] = v / ref - 1
        rows.append(r)
    return rows


def summarise(df: pd.DataFrame, label: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("err_")]
    g = df.groupby("f_target")
    s = pd.concat({c[4:]: g[c].agg(lambda v: f"{100*np.median(v):+.1f} [{100*np.quantile(v,.25):+.1f}, {100*np.quantile(v,.75):+.1f}]")
                   for c in cols}, axis=1)
    s.insert(0, "n", g.size())
    s.insert(1, "f_real_median", g.f_real.median().round(3))
    s.to_csv(OUT / f"summary_{label}.csv")
    return s


# ---------------------------------------------------------------- Part A
def part_a():
    v = pd.read_csv(T1_REGIONS / "t1_vertebrae.csv")
    v["enc"] = np.where(v.intensity_baseline == -1000, "-1000_3096", "other")
    v["regime"] = pd.cut(v.vertebral_high_clip_fraction, [-1, 0, .005, .05, .5, 1],
                         labels=["0", "0-0.5%", "0.5-5%", "5-50%", ">50%"])
    t1 = pd.crosstab(v.enc, v.regime)
    m = pd.read_csv(RESULTS / "t2_canal_ceiling.csv")
    t = m[m.kind == "t2"].drop_duplicates("series").copy()
    t["regime"] = pd.cut(t.csf_clip_hi, [-1, 0, .005, .05, .5, 1], labels=["0", "0-0.5%", "0.5-5%", "5-50%", ">50%"])
    t2 = pd.crosstab(t.enc, t.regime)
    with open(OUT / "partA_prevalence.txt", "w") as fh:
        fh.write("T1 vertebral vROI, fraction of voxels at stored ceiling (rows = encoding)\n")
        fh.write(t1.to_string() + "\n\nT2 whole canal label, fraction at stored ceiling\n" + t2.to_string() + "\n")
    print(open(OUT / "partA_prevalence.txt").read())
    return v, t


# ---------------------------------------------------------------- Part B / D
def part_b_d(v: pd.DataFrame):
    rows, real = [], []
    cache = {}
    for r in v.itertuples():
        pid, lev = r.patient_id, r.anatomic_level
        mp = T1_REGIONS / "masks" / f"{pid}_{lev}_vroi.mha"
        if not mp.exists():
            continue
        if pid not in cache:
            cache.clear()
            raw = sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT / "images" / f"{pid}_t1.mha"))).astype(np.float64)
            cache[pid] = raw
        raw = cache[pid]
        roi = sitk.GetArrayFromImage(sitk.ReadImage(str(mp))).astype(bool)
        if roi.shape != raw.shape:
            continue
        xr = raw[roi]
        x = xr - r.intensity_baseline
        hi = raw.max()
        f_obs = float((xr == hi).mean())
        if f_obs == 0:
            for s in simulate(x):
                rows.append({"patient_id": pid, "level": lev, **s})
        elif f_obs > 0.05:
            c = float(hi - r.intensity_baseline)
            e = estimates(x, c, f_obs)
            real.append({"patient_id": pid, "level": lev, "f_obs": f_obs, "ref": r.reference_signal,
                         **{k: val for k, val in e.items()}})
    B = pd.DataFrame(rows); B.to_csv(OUT / "partB_t1_sim_vertebra.csv", index=False)
    D = pd.DataFrame(real)
    # within-patient context: the patient's unclipped levels (median signal) for comparison
    allv = v.set_index(["patient_id", "anatomic_level"])
    ctx = []
    for r in D.itertuples():
        others = v[(v.patient_id == r.patient_id) & (v.anatomic_level != r.level) & (v.vertebral_high_clip_fraction == 0)]
        ctx.append(others.signal_v.median() if len(others) else np.nan)
    D["other_levels_median"] = ctx
    for k in ["median", "mean", "tobit_normal", "tobit_lognormal", "quantile_normal"]:
        D[f"{k}_vs_others"] = D[k] / D.other_levels_median - 1
    D.to_csv(OUT / "partD_t1_real_clipped.csv", index=False)
    return B, D


# ---------------------------------------------------------------- Part C
def part_c(t: pd.DataFrame):
    rows = []
    for s in t[t.enc == "other"].series:
        img = sitk.ReadImage(str(ROOT / "images" / f"{s}.mha"))
        raw = sitk.GetArrayFromImage(img).astype(np.float64)
        lab = sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT / "masks" / f"{s}.mha")))
        canal = lab == 100
        er = distance_transform_edt(canal, sampling=img.GetSpacing()[::-1]) >= 1.0
        if er.sum() < 50:
            continue
        x = raw[er] - min(raw.min(), 0)
        for r in simulate(x):
            rows.append({"series": s, "raw_min": float(raw.min()), "raw_max": float(raw.max()), **r})
    C = pd.DataFrame(rows); C.to_csv(OUT / "partC_t2_csf_sim.csv", index=False)
    return C


if __name__ == "__main__":
    v, t = part_a()
    B, D = part_b_d(v)
    print("\nPart B  T1 vertebral vROI simulation: % error, median [IQR] across vertebrae")
    print(summarise(B, "t1_vertebra").to_string())
    C = part_c(t)
    print("\nPart C  T2 canal (CSF) simulation: % error, median [IQR] across series")
    print(summarise(C, "t2_csf").to_string())
    print("\nPart D  real clipped T1 vertebrae (>5% at ceiling)")
    print(D.round(3).to_string())
