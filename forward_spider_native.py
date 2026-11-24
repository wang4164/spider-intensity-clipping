#!/usr/bin/env python
"""Forward simulation on SPIDER's native (unconverted) volumes: apply the inferred conversion and measure the bias.

Native subset (spider_encoding_probe.csv): 23 T1, 22 T2, 28 T2 SPACE volumes (Siemens, values never clipped).
Inferred conversion, in baseline-corrected units:  y = clip(round(k (x - p5)), 0, 4096),  k = 4096 / (p95 - p5),
p5/p95 over the whole stored volume.  The linear map without clipping, l = k (x - p5), separates the two error
sources: offset (l vs native) and clipping (y vs l).

T1: vertebral trabecular regions L1-L4 and the L3 canal-fluid reference written by build_t1_regions.py
    (results/t1_regions/masks); vertebral score = median of level medians / reference median.
T2 and SPACE: CSF = canal label (100) eroded 1 mm; discs = labels 201-209 eroded 1 mm; ratio = disc median / CSF median.
Observed clipping in the converted volumes (same ROI definitions) is printed for comparison.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).parent))
from run_clipping_pilot import estimates  # noqa: E402

from paths import SPIDER as ROOT, RESULTS, T1_REGIONS
RES = RESULTS
LEVELS = ["L1", "L2", "L3", "L4"]
REG = ([-1, 0, .005, .05, .5, 1], ["0", "0-0.5%", "0.5-5%", "5-50%", ">50%"])


def convert(x, p5, p95):
    k = 4096.0 / (p95 - p5)
    lin = k * (x - p5)
    return lin, np.clip(np.round(lin), 0, 4096)


def est_all(vals_clipped, f):
    if f == 0:
        m = float(np.median(vals_clipped))
        return {"median": m, "mean": float(np.mean(vals_clipped)), "tobit_normal": m, "quantile_normal": m}
    e = estimates(vals_clipped, 4096.0, f)
    return {k: e[k] for k in ["median", "mean", "tobit_normal", "quantile_normal"]}


def t1(native):
    v = pd.read_csv(T1_REGIONS / "t1_vertebrae.csv")
    vrows, prows = [], []
    for pid in sorted(native):
        img = sitk.ReadImage(str(ROOT / "images" / f"{pid}_t1.mha"))
        x = sitk.GetArrayFromImage(img).astype(np.float64)
        b = float(v.loc[v.patient_id == pid, "intensity_baseline"].iloc[0])
        p5, p95 = np.quantile(x, [0.05, 0.95])
        lin, y = convert(x, p5, p95)
        def m(name):
            p = T1_REGIONS / "masks" / f"{pid}_{name}.mha"
            return sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(bool) if p.exists() else None
        ref = m("L3_canal_reference")
        if ref is None or not ref.any():
            continue
        R_nat, R_lin, R_y = np.median(x[ref] - b), np.median(lin[ref]), np.median(y[ref])
        lev = {}
        for L in LEVELS:
            roi = m(f"{L}_vroi")
            if roi is None or not roi.any():
                continue
            yv = y[roi]; f = float((yv >= 4096).mean())
            e = est_all(yv, f)
            S_nat, S_lin = float(np.median(x[roi] - b)), float(np.median(lin[roi]))
            mean_lin = float(np.mean(lin[roi]))
            lev[L] = {"nat": S_nat, "lin": S_lin, **{f"y_{k}": val for k, val in e.items()}}
            vrows.append({"pid": pid, "level": L, "f_clipped": f, "p5_over_signal": p5 / max(S_nat + b, 1e-9),
                          **{f"clip_err_{k}": e[k] / (mean_lin if k == "mean" else S_lin) - 1 for k in e}})
        if len(lev) < 3:
            continue
        S = lambda key: float(np.median([d[key] for d in lev.values()]))
        vbq_nat, vbq_lin = S("nat") / R_nat, S("lin") / R_lin
        r = {"pid": pid, "n_levels": len(lev), "p5": p5, "p95": p95, "ref_native": R_nat, "p5_over_ref": (p5 - b) / R_nat,
             "vbq_native": vbq_nat, "offset_err": vbq_lin / vbq_nat - 1}
        for k in ["median", "mean", "quantile_normal"]:
            vq = S(f"y_{k}") / R_y
            r[f"vbq_conv_{k}"] = vq; r[f"total_err_{k}"] = vq / vbq_nat - 1; r[f"clip_err_{k}"] = vq / vbq_lin - 1
        prows.append(r)
    return pd.DataFrame(vrows), pd.DataFrame(prows)


def t2(files, kind):
    rows = []
    for fn in files:
        img = sitk.ReadImage(str(ROOT / "images" / f"{fn}.mha"))
        x = sitk.GetArrayFromImage(img).astype(np.float64)
        lab = sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT / "masks" / f"{fn}.mha")))
        sp = img.GetSpacing()[::-1]
        canal = lab == 100
        csf = distance_transform_edt(canal, sampling=sp) >= 1.0
        if csf.sum() < 50:
            continue
        p5, p95 = np.quantile(x, [0.05, 0.95])
        lin, y = convert(x, p5, p95)
        f_canal = float((y[canal] >= 4096).mean())
        yc = y[csf]; f = float((yc >= 4096).mean())
        e = est_all(yc, f)
        C_nat, C_lin = float(np.median(x[csf])), float(np.median(lin[csf]))
        r = {"series": fn, "kind": kind, "f_canal_whole": f_canal, "f_csf_roi": f, "p5_over_csf": p5 / C_nat,
             "vol_frac_ceiling": float((y >= 4096).mean()), "vol_occupancy": float(np.unique(y).size / 4097)}
        for k in e:
            r[f"csf_clip_err_{k}"] = e[k] / (float(np.mean(lin[csf])) if k == "mean" else C_lin) - 1
        ratios = []
        for L in [int(z) for z in np.unique(lab) if 200 < z < 300]:
            d = distance_transform_edt(lab == L, sampling=sp) >= 1.0
            if d.sum() < 20:
                continue
            D_nat, D_lin, D_y = np.median(x[d]), np.median(lin[d]), np.median(y[d])
            ratios.append({"nat": D_nat / C_nat, "lin": D_lin / C_lin, "conv": D_y / e["median"],
                           "conv_qn": D_y / e["quantile_normal"], "disc_clip": float((y[d] >= 4096).mean())})
        if ratios:
            R = pd.DataFrame(ratios)
            r.update(n_discs=len(R), ratio_offset_err=float((R.lin / R.nat - 1).median()),
                     ratio_clip_err=float((R.conv / R.lin - 1).median()), ratio_total_err=float((R.conv / R.nat - 1).median()),
                     ratio_total_err_qn=float((R.conv_qn / R.nat - 1).median()), disc_clip_max=float(R.disc_clip.max()))
        rows.append(r)
    return pd.DataFrame(rows)


def q(s):
    return f"{100*s.median():+.1f} [{100*s.quantile(.25):+.1f}, {100*s.quantile(.75):+.1f}] (range {100*s.min():+.1f} to {100*s.max():+.1f})"


if __name__ == "__main__":
    P = pd.read_csv(RES / "spider_encoding_probe.csv")
    nat = P[~((P["min"] == -1000) & (P["max"] == 3096))].copy()
    nat["kind"] = nat.file.str.split("_", n=1).str[1]
    V, PT = t1({int(f.split("_")[0]) for f in nat[nat.kind == "t1"].file})
    V.to_csv(RES / "forward_native_t1_vertebra.csv", index=False); PT.to_csv(RES / "forward_native_t1_patient.csv", index=False)
    S = pd.concat([t2(nat[nat.kind == "t2"].file, "t2"), t2(nat[nat.kind == "t2_SPACE"].file, "t2_SPACE")])
    S.to_csv(RES / "forward_native_t2.csv", index=False)

    print(f"== T1 native forward: {len(V)} vertebrae, {len(PT)} patients")
    print("vertebral vROI clipping regimes (forward):", pd.cut(V.f_clipped, *REG).value_counts().sort_index().to_dict())
    print("observed converted T1 (n=692):            {'0': 450, '0-0.5%': 119, '0.5-5%': 78, '5-50%': 41, '>50%': 4}")
    for k in ["median", "mean", "tobit_normal", "quantile_normal"]:
        print(f"  vertebral signal clip-only error, {k}: {q(V[f'clip_err_{k}'])}")
    print("patient VBQ: offset-only error", q(PT.offset_err))
    for k in ["median", "mean", "quantile_normal"]:
        print(f"  VBQ clip-only ({k}): {q(PT[f'clip_err_{k}'])}   total ({k}): {q(PT[f'total_err_{k}'])}")
    print("  p5 relative to reference median:", PT.p5_over_ref.describe().round(3).to_dict())
    for kind, d in S.groupby("kind"):
        print(f"\n== {kind} native forward: {len(d)} series")
        print("  reproduction: ceiling", d.vol_frac_ceiling.describe().round(4)[["min", "50%", "max"]].to_dict(), "occupancy median", round(d.vol_occupancy.median(), 3))
        print("  whole-canal clipping regimes (forward):", pd.cut(d.f_canal_whole, *REG).value_counts().sort_index().to_dict())
        print("  eroded CSF ROI regimes (forward):     ", pd.cut(d.f_csf_roi, *REG).value_counts().sort_index().to_dict())
        for k in ["median", "mean", "tobit_normal", "quantile_normal"]:
            print(f"  CSF clip-only error, {k}: {q(d[f'csf_clip_err_{k}'])}")
        print("  disc/CSF ratio: offset", q(d.ratio_offset_err), "| clip", q(d.ratio_clip_err), "| total", q(d.ratio_total_err),
              "| total with quantile-normal CSF", q(d.ratio_total_err_qn))
    print("\nobserved converted T2 whole canal (n=188): {'0': 0, '0-0.5%': 3, '0.5-5%': 19, '5-50%': 133, '>50%': 33}")
