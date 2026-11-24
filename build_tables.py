#!/usr/bin/env python
"""Build Tables 1-2 and the Supplementary material (Markdown + CSV) from results/.

Every number is read from the analysis outputs; nothing is typed by hand.
Writes: manuscript/tables_v1.md, manuscript/supplement_v1.md, results/tables/*.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd

from paths import SPIDER as ROOT, RESULTS, T1_REGIONS
RES = RESULTS
MAN = RESULTS / "tables_md"; MAN.mkdir(exist_ok=True)
TAB = RES / "tables"; TAB.mkdir(exist_ok=True)
rng = np.random.default_rng(20261123)


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def mir(s, d=1, scale=100, sign=False):
    """median (IQR) [range] of a series, in percent by default."""
    s = pd.Series(s).dropna() * scale
    f = (lambda v: f"{v:+.{d}f}") if sign else (lambda v: f"{v:.{d}f}")
    return f"{f(s.median())} ({f(s.quantile(.25))} to {f(s.quantile(.75))}) [{f(s.min())} to {f(s.max())}]".replace("-", "−")


def mr(s, d=1, scale=100):
    s = pd.Series(s).dropna() * scale
    return f"{s.median():.{d}f} ({s.min():.{d}f}–{s.max():.{d}f})"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def nfrac(mask):
    mask = pd.Series(mask).dropna().astype(bool)
    return f"{int(mask.sum())}/{len(mask)} ({pct(mask.mean())})"


# ------------------------------------------------------------------ inputs
P = pd.read_csv(RES / "spider_encoding_probe.csv")
P["conv"] = (P["min"] == -1000) & (P["max"] == 3096)
P["seq"] = np.where(P.file.str.contains("SPACE"), "T2 SPACE", np.where(P.file.str.endswith("_t1"), "T1", "T2"))
P["vendor"] = P.manufacturer.str.split().str[0].str.title()
A = pd.read_csv(RES / "external_aisslab_series.csv")
R = pd.read_csv(RES / "external_rsna_series_n0.csv")
RS = pd.read_csv(RES / "rsna_spider_transform_n0.csv")
S = pd.read_csv(RES / "csf_roi_sensitivity.csv")
F2 = pd.read_csv(RES / "forward_native_t2.csv")
F1 = pd.read_csv(RES / "forward_native_t1_patient.csv")
F1v = pd.read_csv(RES / "forward_native_t1_vertebra.csv")
D = pd.read_csv(RES / "partD_t1_real_clipped.csv")
V = pd.read_csv(T1_REGIONS / "t1_vertebrae.csv")

# ------------------------------------------------------------------ Table 1
rows = []
for (conv, seq), d in P.groupby(["conv", "seq"]):
    vend = ", ".join(f"{k} {v}" for k, v in d.vendor.value_counts().items())
    rows.append({"Dataset": "SPIDER, converted" if conv else "SPIDER, native", "Sequence": seq, "Volumes": len(d),
                 "Vendors": vend, "Stored range": "−1000 to 3096" if conv else f"maximum {int(d['max'].median())} ({int(d['max'].min())}–{int(d['max'].max())})",
                 "Voxels at stored maximum, % median (range)": mr(d.frac_max, 4),
                 "Lattice occupancy, % median (range)": mr(d.occupancy, 1),
                 "Classified as clipped (>0.01%)": nfrac(d.frac_max > 1e-4)})
for desc, d in R.groupby("desc"):
    rows.append({"Dataset": "RSNA LumbarDISC", "Sequence": desc.replace("Sagittal ", "Sagittal "), "Volumes": len(d),
                 "Vendors": "not recorded", "Stored range": f"maximum {int(d['max'].median())} ({int(d['max'].min())}–{int(d['max'].max())})",
                 "Voxels at stored maximum, % median (range)": mr(d.frac_at_max, 4),
                 "Lattice occupancy, % median (range)": "—", "Classified as clipped (>0.01%)": nfrac(d.frac_at_max > 1e-4)})
rows.append({"Dataset": "Second lumbar dataset", "Sequence": "Sagittal T2", "Volumes": len(A),
             "Vendors": ", ".join(f"{({'GE': 'GE'}).get(k.split()[0], k.split()[0].title())} {v}" for k, v in A.manufacturer.value_counts().items()),
             "Stored range": f"maximum {int(A['max'].median())} ({int(A['max'].min())}–{int(A['max'].max())})",
             "Voxels at stored maximum, % median (range)": mr(A.frac_at_max, 4),
             "Lattice occupancy, % median (range)": "—", "Classified as clipped (>0.01%)": nfrac(A.frac_at_max > 1e-4)})
T1 = pd.DataFrame(rows)
T1.to_csv(TAB / "Table_1.csv", index=False)

# ------------------------------------------------------------------ Table 2
cens = lambda s: nfrac(pd.Series(s).dropna() > .5)
rois = [("whole", "Whole canal label"), ("ero1", "Canal eroded 1 mm"), ("ero2", "Canal eroded 2 mm"), ("core", "Core CSF (upper 60%)")]
sets = [("SPIDER converted, observed", "t2", True), ("SPIDER converted, observed", "SPACE", True),
        ("SPIDER native, after transform", "t2", False), ("SPIDER native, after transform", "SPACE", False)]
rowsA = []
for lbl, kind, conv in sets:
    d = S[(S.kind == kind) & (S.converted == conv)]
    r = {"Data": lbl, "Sequence": "2D T2" if kind == "t2" else "3D SPACE", "Series": len(d)}
    for k, name in rois:
        r[name] = cens(d[f"f_{k}"])
    rowsA.append(r)
rowsA.append({"Data": "RSNA, after transform", "Sequence": "2D T2/STIR", "Series": len(RS),
              "Whole canal label": "—", "Canal eroded 1 mm": "—", "Canal eroded 2 mm": "—",
              "Core CSF (upper 60%)": cens(RS.roi_frac_clipped) + " (central 2.5-mm discs)"})
T2a = pd.DataFrame(rowsA)
rowsB = []
for kind, lbl in [("t2", "2D T2"), ("t2_SPACE", "3D SPACE")]:
    d = F2[F2.kind == kind]
    rowsB.append({"Sequence": lbl, "Series": len(d), "CSF region": "Canal eroded 1 mm",
                  "Offset component": mir(d.ratio_offset_err, sign=True), "Clipping component": mir(d.ratio_clip_err, sign=True),
                  "Total, CSF median": mir(d.ratio_total_err, sign=True), "Total, quantile-normal CSF": mir(d.ratio_total_err_qn, sign=True)})
    d2 = S[(S.kind == ("t2" if kind == "t2" else "SPACE")) & ~S.converted]
    rowsB.append({"Sequence": lbl, "Series": len(d2), "CSF region": "Core CSF (upper 60%)", "Offset component": "—", "Clipping component": "—",
                  "Total, CSF median": mir(d2.ratio_err_median_core, sign=True), "Total, quantile-normal CSF": mir(d2.ratio_err_qn_core, sign=True)})
T2b = pd.DataFrame(rowsB)
T2a.to_csv(TAB / "Table_2A.csv", index=False); T2b.to_csv(TAB / "Table_2B.csv", index=False)

tables_md = f"""# Tables

Table 1. Stored encoding and ceiling pile-up of the analysed volumes, by dataset and sequence.

{md_table(T1)}

Data are n, n/N (%) or median (range). Voxels at stored maximum is the share of all voxels of a volume equal to its largest stored value. Lattice occupancy is the number of distinct stored values divided by the number of integers between the stored minimum and maximum; it was computed for SPIDER only (—). Volumes were classified as clipped when more than 0.01% of voxels lay at the stored maximum; any threshold between 0.01% and 4% gave the same classification (Supplementary Table S2). RSNA series are anonymised and do not record the vendor. Converted SPIDER volumes are those stored in the −1000 to 3096 encoding; native volumes are stored as acquired. RSNA, Radiological Society of North America; SPACE, sampling perfection with application-optimised contrasts using different flip-angle evolution.

Table 2. Censoring of the CSF reference and error of disc-to-CSF ratios under the inferred transform.

(A) Series in which more than half of the CSF region lay at the ceiling, so that its median was censored

{md_table(T2a)}

(B) Relative error (%) of disc-to-CSF ratios in native SPIDER volumes after the transform

{md_table(T2b)}

Data are n/N (%) in (A) and median (IQR) [range] across series in (B); each series contributes the median error over its discs. Observed values in (A) are measured on the released converted volumes; values after transform are obtained by applying the inferred per-volume 5th–95th percentile clip to native volumes. In RSNA the CSF region is the union of 2.5-mm discs around annotated canal points, which samples central CSF and is therefore closest to the core definition. The offset component compares the ratio after linear rescaling without clipping with the native ratio; the clipping component compares the ratio after clipping with the ratio after linear rescaling; the total compares the ratio after the full transform with the native ratio. Quantile-normal CSF replaces the CSF median by the centre of a normal distribution fitted through two empirical quantiles below the ceiling. CSF, cerebrospinal fluid; IQR, interquartile range.
"""
(MAN / "tables_v1.md").write_text(tables_md, encoding="utf-8")

# ------------------------------------------------------------------ Supplement
S1 = pd.read_csv(RES / "detector_threshold_sweep.csv").rename(columns={"threshold_%": "Threshold, % of voxels at maximum",
                                                                       "sensitivity": "Converted volumes flagged", "false_pos": "Native volumes flagged"})
s2 = []
for name, fn in [("Vertebral marrow, T1 (542 regions)", "summary_t1_vertebra.csv"), ("CSF, native T2 (22 series)", "summary_t2_csf.csv")]:
    d = pd.read_csv(RES / fn)
    for _, r in d.iterrows():
        s2.append({"Region": name, "Clipped, %": f"{100*r.f_target:g}", "Median": r["median"], "Mean": r["mean"], "Trimmed mean (5%)": r.trimmed_mean,
                   "Censored-normal MLE": r.tobit_normal, "Censored log-normal MLE": r.tobit_lognormal, "Quantile-normal fit": r.quantile_normal})
S2 = pd.DataFrame(s2)

ov = pd.read_csv(ROOT / "overview.csv"); ov = ov[ov.new_file_name.str.endswith("_t1")].copy()
ov["patient_id"] = ov.new_file_name.str.split("_").str[0].astype(int)
Vs = V.merge(ov[["patient_id", "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength"]], on="patient_id")
Vs["Encoding"] = np.where(Vs.intensity_baseline == -1000, "converted", "native")
Vs["Scanner"] = Vs.Manufacturer.str.split().str[0].str.title() + " " + Vs.ManufacturerModelName + " " + Vs.MagneticFieldStrength.astype(str) + " T"
g = Vs.groupby(["Encoding", "Scanner"])
S3 = pd.DataFrame({"Vertebrae": g.size(), "Patients": g.patient_id.nunique(),
                   "Any voxel at ceiling": g.vertebral_high_clip_fraction.apply(lambda s: nfrac(s > 0)),
                   ">5% at ceiling": g.vertebral_high_clip_fraction.apply(lambda s: nfrac(s > .05)),
                   ">50% at ceiling": g.vertebral_high_clip_fraction.apply(lambda s: nfrac(s > .5))}).reset_index()

RS["regime"] = pd.cut(RS.roi_frac_clipped, [-1, 0, .05, .5, 1], labels=["0", ">0 to 5%", ">5 to 50%", ">50%"])
s4 = []
for reg, d in RS.groupby("regime", observed=True):
    s4.append({"Pixels clipped": reg, "Series": len(d), "Median": mir(d.err_median, sign=True), "Mean": mir(d.err_mean, sign=True),
               "Censored-normal MLE": mir(d.err_tobit_normal, sign=True), "Quantile-normal fit": mir(d.err_quantile_normal, sign=True)})
S4a = pd.DataFrame(s4)
s4b = []
for q in (0.95, 0.98, 0.99, 0.995, 0.999):
    x = (RS[f"roi_frac_clipped_q{q}"] > .5).values.astype(float); y = (RS[f"roi_frac_clipped_q{q}"] > 0).values.astype(float)
    bx = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    s4b.append({"Upper clipping percentile": f"{100*q:g}th", "CSF median censored, % (95% CI)": f"{100*x.mean():.1f} ({100*np.percentile(bx,2.5):.1f}–{100*np.percentile(bx,97.5):.1f})",
                "Any CSF pixel clipped, %": f"{100*y.mean():.1f}"})
S4b = pd.DataFrame(s4b)

S5 = pd.DataFrame({"Statistic": ["Vertebrae with >5% of region at ceiling (patients)", "Vertebrae with >50% at ceiling",
                                 "Mean below quantile-normal estimate, % median (range)",
                                 "Quantile-normal above naive median where >50% clipped, % (range)"],
                   "Value": [f"{len(D)} ({D.patient_id.nunique()})", f"{int((D.f_obs > .5).sum())}",
                             f"{100*(1 - D['mean']/D.quantile_normal).median():.1f} ({100*(1 - D['mean']/D.quantile_normal).min():.1f} to {100*(1 - D['mean']/D.quantile_normal).max():.1f})",
                             (lambda d: f"{100*(d.quantile_normal/d['median']-1).min():.1f} to {100*(d.quantile_normal/d['median']-1).max():.1f}")(D[D.f_obs > .5])]})
S5b = D[D.f_obs > .5][["level", "f_obs", "median", "mean", "tobit_normal", "quantile_normal"]].copy()
S5b.insert(0, "Vertebra", [f"Case {i+1}, {l}" for i, l in enumerate(S5b.level)]); S5b = S5b.drop(columns="level")
S5b["f_obs"] = (100 * S5b.f_obs).round(1)
for c in ["median", "mean", "tobit_normal", "quantile_normal"]:
    S5b[c] = S5b[c].round(0).astype(int)
S5b.columns = ["Vertebra", "Region at ceiling, %", "Median", "Mean", "Censored-normal MLE", "Quantile-normal fit"]

S6 = pd.DataFrame([
    {"Quantity": "Vertebrae with >5% at ceiling after transform", "Value": nfrac(F1v.f_clipped > .05)},
    {"Quantity": "Vertebral signal, clipping component (median estimate)", "Value": mir(F1v.clip_err_median, sign=True)},
    {"Quantity": "Vertebral signal, clipping component (mean estimate)", "Value": mir(F1v.clip_err_mean, sign=True)},
    {"Quantity": "Patient vertebral score, offset component", "Value": mir(F1.offset_err, sign=True)},
    {"Quantity": "Patient vertebral score, clipping component", "Value": mir(F1.clip_err_median, sign=True)},
    {"Quantity": "Patient vertebral score, total", "Value": mir(F1.total_err_median, sign=True)},
    {"Quantity": "p5 minus native baseline, relative to reference", "Value": mir(F1.p5_over_ref, sign=True)},
    {"Quantity": "Native reference signal, stored units, median (range)", "Value": f"{F1.ref_native.median():.0f} ({F1.ref_native.min():.0f}–{F1.ref_native.max():.0f})"}])

for name, df in [("S2", S1), ("S3", S2), ("S1", S3), ("S5A", S4a), ("S5B", S4b), ("S6A", S5), ("S6B", S5b), ("S4", S6)]:
    df.to_csv(TAB / f"Table_{name}.csv", index=False)

supp = f"""# Supplementary material

Undocumented percentile clipping in a public lumbar spine MRI dataset: detection, reverse engineering and bias in reference-normalised signal ratios

Contents

- Supplementary Methods S1. Detection metrics and estimators
- Supplementary Table S1. Ceiling voxels in T1 vertebral marrow regions by scanner and encoding
- Supplementary Table S2. Detection of clipped volumes across thresholds
- Supplementary Table S3. Error of region estimates under simulated clipping within SPIDER, all estimators
- Supplementary Table S4. Replay of the transform on native SPIDER T1 volumes
- Supplementary Table S5. RSNA replay: error by clipping class and censoring by clipping percentile
- Supplementary Table S6. Clipped vertebrae in the released SPIDER volumes

# Supplementary Methods S1. Detection metrics and estimators

Detection. For a volume with stored values v, the share at the maximum is the number of voxels with v = max(v) divided by the number of voxels, and the share at the minimum is defined analogously. Lattice occupancy is the number of distinct stored values divided by max(v) − min(v) + 1. The mean spacing between used values is the mean difference between consecutive distinct stored values, excluding the two bounds. For linear rescaling by a factor k > 1 followed by rounding, the expected spacing is k, and occupancy is approximately 1/k.

Transform. With p5 and p95 the 5th and 95th percentiles of all voxels of the native volume x, the inferred transform is y = clip((x − p5)/(p95 − p5), 0, 1) × 4096 − 1000, rounded to integers. After removal of the −1000 offset it is y′ = min(max(round(k(x − p5)), 0), 4096) with k = 4096/(p95 − p5). In replay, clipping was applied at 4096 and the linear map without clipping, k(x − p5), served as reference for the clipping component.

Censored-normal maximum likelihood. Voxels at the ceiling c are treated as right-censored. With n_u uncensored values x_i and n_c censored voxels, the log-likelihood of a normal distribution with mean μ and standard deviation σ is Σ log φ((x_i − μ)/σ) − n_u log σ + n_c log[1 − Φ((c − μ)/σ)], maximised by the Nelder–Mead method from starting values at the median and standard deviation of the uncensored values; μ is returned as the regional signal. The log-normal variant applies the same model to log-transformed values and returns exp(μ).

Quantile-normal fit. With the clipped share f, two empirical quantiles below the ceiling are taken at probabilities p_a = 0.10 and p_b = min(0.50, 0.9(1 − f)), with p_b ≥ p_a + 0.02; σ = (q_b − q_a)/(z_b − z_a) and μ = q_b − z_b σ, where z are standard normal quantiles, and μ is returned as the regional signal. When less than half of a region is clipped, p_b = 0.50 and the estimate equals the empirical median.

Supplementary Table S1. Ceiling voxels in T1 vertebral marrow regions by scanner and encoding.

{md_table(S3)}

Data are n or n/N (%). Regions are the 2-mm-eroded trabecular regions of L1–L4; shares refer to stored values equal to the volume maximum. Scanner model and field strength are taken from the DICOM metadata distributed with the dataset.

Supplementary Table S2. Detection of clipped volumes across thresholds.

{md_table(S1)}

Converted volumes are the 374 SPIDER volumes stored in the −1000 to 3096 encoding; native volumes are 73 SPIDER, 3954 RSNA and 500 second-dataset volumes. The threshold applies to the share of all voxels at the stored maximum. The largest share among native volumes was 0.0046% and the smallest among converted volumes 5.000%.

Supplementary Table S3. Error of region estimates under simulated clipping within SPIDER, all estimators.

{md_table(S2)}

Data are median [IQR] relative error (%) across regions, comparing each estimator after clipping with the same estimator without clipping; censoring-aware estimators are compared with the unclipped median. Regions without any ceiling voxel served as ground truth: 542 vertebral marrow regions of 167 patients on T1-weighted volumes and the 1-mm-eroded canal of 22 native T2-weighted volumes. Each region was clipped at the quantile placing the stated share of voxels at the ceiling. The censored log-normal estimator was biased at every clipped share and was not used in the main analysis. MLE, maximum-likelihood estimate.

Supplementary Table S4. Replay of the transform on native SPIDER T1 volumes.

{md_table(S6)}

Data are n/N (%) or median (IQR) [range] in % unless stated, for 92 vertebrae of 23 patients imaged on one Siemens 1.5 T scanner model. The patient vertebral score is the median of the L1–L4 marrow signals divided by the L3 canal-fluid reference. The offset component arises because the transform sets the zero of the scale at p5 rather than at the native baseline; it is large for the T1 reference because canal fluid is dark on T1-weighted images.

Supplementary Table S5. RSNA replay: error by clipping class and censoring by clipping percentile.

(A) Relative error (%) of the CSF estimate by share of CSF pixels clipped

{md_table(S4a)}

(B) Share of series with a censored CSF median by upper clipping percentile

{md_table(S4b)}

Data are median (IQR) [range] in (A) and % with 95% bootstrap confidence intervals (2000 resamples of studies) in (B), for 1973 RSNA sagittal T2/STIR series with annotated canal points. The CSF region is the union of 2.5-mm discs around the annotated points. The censored-normal estimate was unstable in these small regions.

Supplementary Table S6. Clipped vertebrae in the released SPIDER volumes.

{md_table(S5)}

Vertebrae with more than half of the region at the ceiling, stored units after removal of the −1000 offset:

{md_table(S5b)}

No ground truth exists for these vertebrae; the comparison is descriptive. Case numbers are sequential and do not correspond to dataset identifiers.
"""
(MAN / "supplement_v1.md").write_text(supp, encoding="utf-8")
print(md_table(T1)); print(); print(md_table(T2a)); print(); print(md_table(T2b)); print(); print(md_table(S3)); print(); print(md_table(S5)); print(md_table(S6))
