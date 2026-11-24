#!/usr/bin/env python
"""v2 tables (adds Table 3) and the reordered v2 supplement. Runs build_tables.py first and reuses its objects.
Writes manuscript/tables_v2.md, manuscript/supplement_v2.md and results/tables/Table_3*.csv, Table_v2_*.csv."""
from __future__ import annotations
import re
import runpy
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

G = runpy.run_path(str(Path(__file__).with_name("build_tables.py")), run_name="build_tables")
RES, MAN, TAB = G["RES"], G["MAN"], G["TAB"]
md_table, mir = G["md_table"], G["mir"]
rng = np.random.default_rng(20261124)


def unicode_minus(text):
    """typographic minus for negative numbers in table rows only"""
    nl = chr(10)
    return nl.join(re.sub(r"(?<=[\s\[(,])-(?=\d)", "−", l) if l.startswith("|") else l for l in text.split(nl))


DI = pd.read_csv(RES / "ratio_incidence_discs.csv")
DI = DI[DI.roi != "core_reselected"].copy()
ROI_NAME = {"whole": "Whole canal label", "ero1": "Canal eroded 1 mm", "ero2": "Canal eroded 2 mm", "core": "Core CSF (upper 60%)"}


def boot_share(d, flag, B=2000):
    pids = d.pid.unique(); grp = {p: x[flag].values for p, x in d.groupby("pid")}
    bs = [np.concatenate([grp[p] for p in rng.choice(pids, len(pids))]).mean() for _ in range(B)]
    return f"{100*d[flag].mean():.0f} ({100*np.percentile(bs, 2.5):.0f}–{100*np.percentile(bs, 97.5):.0f})"


t3a = []
for (kind, roi), d in DI.groupby(["kind", "roi"], sort=False):
    d = d.copy(); r = {"Sequence": kind, "CSF region": ROI_NAME[roi], "Discs (patients)": f"{len(d)} ({d.pid.nunique()})"}
    for t in (0.10, 0.20, 0.30):
        d["f"] = (d.err_total.abs() > t).astype(float); r[f"Error >{int(100*t)}%"] = boot_share(d, "f")
    d["f"] = (d.err_total_qn.abs() > 0.10).astype(float); r["Error >10%, quantile-normal CSF"] = boot_share(d, "f")
    t3a.append(r)
T3a = pd.DataFrame(t3a)
t3b = []
for roi in ["whole", "ero1", "core"]:
    d = DI[DI.roi == roi].copy()
    d["level"] = np.select([d.csf_clipped == 0, d.csf_clipped <= 0.5], ["1: none clipped", "2: ≤50% clipped"], "3: >50% clipped")
    d["bad"] = d.err_total.abs() > 0.10; d["off"] = d.bad & (d.err_offset.abs() > 0.10)
    for lvl, g_ in d.groupby("level"):
        t3b.append({"CSF region": ROI_NAME[roi], "QC level (released image)": lvl, "Discs": len(g_),
                    "Error >10%": f"{int(g_.bad.sum())}/{len(g_)} ({100*g_.bad.mean():.0f})",
                    "Of these, offset >10%": f"{int(g_.off.sum())}/{int(g_.bad.sum())}"})
T3b = pd.DataFrame(t3b)
T3a.to_csv(TAB / "Table_3A.csv", index=False); T3b.to_csv(TAB / "Table_3B.csv", index=False)

# external per-disc replay on RSNA (rsna_disc_replay.py): incidence and level-3 screening by series type
RD = pd.read_csv(RES / "rsna_disc_replay.csv")
RD = RD[RD.t2_type.isin(["plain_T2", "fat_suppressed"])].copy()
RD["err_total"] = RD.R2 / RD.R0 - 1; RD["err_offset"] = RD.R1 / RD.R0 - 1; RD["err_clip"] = RD.R2 / RD.R1 - 1
RD["bad"] = RD.err_total.abs() > 0.10; RD["l3"] = RD.csf_clipped > 0.5; RD["off"] = RD.bad & (RD.err_offset.abs() > 0.10)
RD_GROUPS = [("All", RD), ("Plain T2", RD[RD.t2_type == "plain_T2"]), ("Fat-suppressed", RD[RD.t2_type == "fat_suppressed"])]


def boot_study(d, num, den=None, B=2000):
    """share num/den (den None = all rows) with a 95% CI from bootstrap resamples of studies"""
    agg = d.assign(_n=1).groupby("study").agg(a=(num, "sum"), b=(den or "_n", "sum"))
    a, b = agg.a.values.astype(float), agg.b.values.astype(float)
    idx = rng.integers(0, len(a), (B, len(a)))
    bs = a[idx].sum(1) / b[idx].sum(1)
    return f"{100*a.sum()/b.sum():.0f} ({100*np.percentile(bs, 2.5):.0f}–{100*np.percentile(bs, 97.5):.0f})"


t3c = []
for name, d in RD_GROUPS:
    d = d.copy(); r = {"Series type": name, "Discs (studies)": f"{len(d)} ({d.study.nunique()})"}
    for t in (0.10, 0.20, 0.30):
        d["f"] = d.err_total.abs() > t; r[f"Error >{int(100*t)}%"] = boot_study(d, "f")
    d["tp"] = d.bad & d.l3; d["tn"] = ~d.bad & ~d.l3; d["nb"] = ~d.bad; d["fn"] = d.bad & ~d.l3
    r["Sensitivity of level 3"] = boot_study(d, "tp", "bad"); r["Specificity of level 3"] = boot_study(d, "tn", "nb")
    r["Missed errors >10%"] = boot_study(d, "fn")
    t3c.append(r)
T3c = pd.DataFrame(t3c); T3c.to_csv(TAB / "Table_3C.csv", index=False)

mi = lambda s: f"{100*s.median():+.1f} ({100*s.quantile(.25):+.1f} to {100*s.quantile(.75):+.1f})".replace("-", "−")
s10a, s10b = [], []
for name, d in RD_GROUPS:
    s10a.append({"Series type": name, "Offset component": mi(d.err_offset), "Clipping component": mi(d.err_clip),
                 "Total": mi(d.err_total), "Offset >10%, %": f"{100*(d.err_offset.abs() > .1).mean():.0f}",
                 "Clipping >10%, %": f"{100*(d.err_clip.abs() > .1).mean():.0f}",
                 "CSF median censored, %": f"{100*d.l3.mean():.0f}", "Disc median censored, %": f"{100*(d.disc_clipped > .5).mean():.0f}"})
    if name == "All":
        continue
    lev = np.select([d.csf_clipped == 0, d.csf_clipped <= 0.5], ["1: none clipped", "2: ≤50% clipped"], "3: >50% clipped")
    for lvl, g_ in d.groupby(lev):
        s10b.append({"Series type": name, "QC level (released image)": lvl, "Discs": len(g_),
                     "Error >10%": f"{int(g_.bad.sum())}/{len(g_)} ({100*g_.bad.mean():.0f})",
                     "Of these, offset >10%": f"{int(g_.off.sum())}/{int(g_.bad.sum())}"})
S10A, S10B = pd.DataFrame(s10a), pd.DataFrame(s10b)
S10C = pd.DataFrame([{"Level": lv, "Discs": len(g_), "Error >10%, %": f"{100*g_.bad.mean():.0f}",
                      "Missed errors >10%, %": f"{100*(g_.bad & ~g_.l3).mean():.0f}"} for lv, g_ in RD.groupby("level")])
S10A.to_csv(TAB / "Table_v2_S10A.csv", index=False); S10B.to_csv(TAB / "Table_v2_S10B.csv", index=False)

comp = DI.assign(err_clip=(1 + DI.err_total) / (1 + DI.err_offset) - 1)
s6 = []
for (kind, roi), d in comp.groupby(["kind", "roi"], sort=False):
    mi = lambda s: f"{100*s.median():+.1f} ({100*s.quantile(.25):+.1f} to {100*s.quantile(.75):+.1f})".replace("-", "−")
    s6.append({"Sequence": kind, "CSF region": ROI_NAME[roi], "Offset component": mi(d.err_offset),
               "Clipping component": mi(d.err_clip), "Total": mi(d.err_total),
               "Offset >10%, %": f"{100*(d.err_offset.abs() > .1).mean():.0f}", "Clipping >10%, %": f"{100*(d.err_clip.abs() > .1).mean():.0f}"})
S6A = pd.DataFrame(s6)
s6b = []
for kind in ["2D T2", "SPACE"]:
    d = DI[(DI.roi == "ero1") & (DI.kind == kind)].dropna(subset=["pfirrmann"])
    r = {"Sequence": kind}
    for gr, x in d.groupby("pfirrmann"):
        r[f"Grade {int(gr)}"] = f"{100*x.err_offset.median():+.1f} (n={len(x)})".replace("-", "−")
    r["Spearman ρ with grade"] = f"{d[['err_offset', 'pfirrmann']].corr(method='spearman').iloc[0, 1]:.2f}".replace("-", "−")
    s6b.append(r)
S6B = pd.DataFrame(s6b)
S6A.to_csv(TAB / "Table_v2_S6A.csv", index=False); S6B.to_csv(TAB / "Table_v2_S6B.csv", index=False)
CT = pd.read_csv(RES / "tables" / "Table_candidate_transforms.csv")

tables_v2 = G["tables_md"].replace("(Supplementary Table S2)", "(Supplementary Table S3)") + f"""
Table 3. Incidence of disc-to-CSF ratio errors in native SPIDER volumes after the inferred transform, and their detection by a clipping-based quality-control level.

(A) Discs with an absolute ratio error above each threshold (Error >x%), % (95% CI)

{md_table(T3a)}

(B) Discs with an absolute ratio error above 10% by quality-control level

{md_table(T3b)}

(C) External replay on RSNA discs: discs with an absolute ratio error above each threshold, and level 3 as a screen for an error above 10%, % (95% CI)

{md_table(T3c)}

Data are % of discs with 95% confidence intervals from 2000 bootstrap resamples of patients in (A) and of studies in (C), and n/N (%) in (B). (A) and (B) use native SPIDER volumes; (C) uses 1906 RSNA studies whose T2/STIR series could be classified as plain or fat-suppressed T2 from the images, with discs located by SpineNetV2 and the CSF region around the annotated canal points. Sensitivity is the share of discs with an error above 10% that level 3 flagged, specificity the share of discs without such an error that it did not flag, and missed errors are discs with an error above 10% outside level 3, as a share of all discs. The error compares the disc-to-CSF ratio after the transform with the native ratio, both from region medians; CSF regions are fixed from native coordinates, and re-selecting the core region on the transformed image gave identical voxels because the transform preserves intensity order. The quality-control level is the share of CSF-region voxels at the ceiling, which can be measured on released data. The offset error, which arises because the transform sets the zero of the scale at the 5th percentile, cannot be measured on released data; the last column of (B) counts discs whose offset component alone exceeded 10%. QC, quality control.
"""
(MAN / "tables_v2.md").write_text(unicode_minus(tables_v2), encoding="utf-8")

S1, S2, S3, S4a, S4b, S5, S5b, S6 = (G[k] for k in ["S1", "S2", "S3", "S4a", "S4b", "S5", "S5b", "S6"])
_ga = pd.read_csv(RES / "tables" / "Table_grade_association.csv")
_ga["CSF region"] = _ga["CSF region"].map(ROI_NAME)
GA = _ga.rename(columns={"rho native": "ρ native", "rho transformed": "ρ after transform",
                         "slope native (%/grade)": "Within-patient slope native, %/grade",
                         "slope transformed (%/grade)": "Within-patient slope after transform, %/grade",
                         "slope change (log, 95% CI)": "Slope change, log (95% CI)",
                         "G5/G2 native": "G5/G2 native", "G5/G2 transformed": "G5/G2 after transform",
                         "within-patient order flips": "Within-patient order flips"})
for _c, _fmt in [("ρ native", "{:.3f}"), ("ρ after transform", "{:.3f}"), ("Within-patient slope native, %/grade", "{:.1f}"),
                 ("Within-patient slope after transform, %/grade", "{:.1f}"), ("G5/G2 native", "{:.3f}"), ("G5/G2 after transform", "{:.3f}")]:
    GA[_c] = GA[_c].astype(float).map(_fmt.format)
GA = GA.astype(str).apply(lambda col: col.str.replace("-", "−", regex=False))
_rs = G["RS"].merge(pd.read_csv(RES / "rsna_t2_group.csv"), on="series")
_rl = []
for grp, lab in [("T2", "Described as T2"), ("unlabelled", "No description")]:
    d = _rs[_rs.grp == grp]
    row = {"Series label": lab, "Series": len(d)}
    for q, name in [(None, "p95 (inferred transform)"), (0.995, "p99.5"), (0.999, "p99.9")]:
        x = ((d.roi_frac_clipped if q is None else d[f"roi_frac_clipped_q{q}"]) > .5).values.astype(float)
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        row[f"Censored at {name}, %"] = f"{100*x.mean():.1f} ({100*np.percentile(b, 2.5):.1f}–{100*np.percentile(b, 97.5):.1f})"
    row["Median error where censored, %"] = f"{100*d.loc[d.roi_frac_clipped > .5, 'err_median'].median():.1f}".replace("-", "−")
    _rl.append(row)
RL = pd.DataFrame(_rl)
sections = [
    ("Supplementary Table S1. Ceiling voxels in T1 vertebral marrow regions by scanner and encoding.", md_table(S3),
     "Data are n or n/N (%). Regions are the 2-mm-eroded trabecular regions of L1–L4 of 196 T1 volumes (784 regions); shares refer to stored values equal to the volume maximum. Scanner model and field strength are taken from the DICOM metadata distributed with the dataset."),
    ("Supplementary Table S2. Ten candidate intensity transforms applied to the 73 native SPIDER volumes, compared with the converted release.", md_table(CT),
     "Each candidate maps native values linearly between two bounds onto integers 0–4096 (−1000 to 3096 after the offset), clipping values outside the bounds. Percentile bounds are computed over all voxels of each volume; the DICOM window uses the stored window centre and width. The observed envelope is a ceiling share between 5.00% and 5.12% and a floor share of at least 5.00%; the smallest floor share in the release was 5.000%, which a lower bound above the 5th percentile cannot produce. Floor shares and occupancy depend on the amount of background and on the native intensity range, and differ between the native subset and the converted release, which came from other scanners."),
    ("Supplementary Table S3. Detection of clipped volumes across thresholds.", md_table(S1),
     "Converted volumes are the 374 SPIDER volumes stored in the −1000 to 3096 encoding; native volumes are 73 SPIDER, 3954 RSNA and 500 second-dataset volumes. The threshold applies to the share of all voxels at the stored maximum. The largest share among native volumes was 0.0046% and the smallest among converted volumes 5.000%."),
    ("Supplementary Table S4. Error of region estimates under simulated clipping within SPIDER, all estimators.", md_table(S2),
     "Data are median [IQR] relative error (%) across regions, comparing each estimator after clipping with the same estimator without clipping; censoring-aware estimators are compared with the unclipped median. Regions without any ceiling voxel served as ground truth: 542 vertebral marrow regions of 167 patients on T1-weighted volumes and the 1-mm-eroded canal of 22 native T2-weighted volumes. The censored log-normal estimator was biased at every clipped share and was not used in the main analysis. MLE, maximum-likelihood estimate."),
    ("Supplementary Table S5. Replay of the transform on native SPIDER T1 volumes.", md_table(S6),
     "Data are n/N (%) or median (IQR) [range] in % unless stated, for 92 vertebrae of 23 patients imaged on one Siemens 1.5 T scanner model. The patient vertebral score is the median of the L1–L4 marrow signals divided by the L3 canal-fluid reference. The offset component arises because the transform sets the zero of the scale at p5 rather than at the native baseline; it is large for the T1 reference because canal fluid is dark on T1-weighted images."),
    ("Supplementary Table S6. Components of the disc-to-CSF ratio error and the offset component by Pfirrmann grade.",
     "(A) Components by sequence and CSF region, % median (IQR); shares with an absolute component above 10%\n\n" + md_table(S6A)
     + "\n\n(B) Offset component by Pfirrmann grade, canal eroded 1 mm, % median\n\n" + md_table(S6B),
     "Per-disc errors in 22 native 2D T2 and 28 native SPACE volumes after the inferred transform. The clipping component is the ratio error after the full transform relative to the ratio after linear rescaling without clipping. Pfirrmann grades are those distributed with the dataset; grade 1 discs in 2D T2 come from few patients."),
    ("Supplementary Table S7. Association between Pfirrmann grade and the disc-to-CSF ratio before and after the inferred transform.",
     md_table(GA),
     "Native SPIDER T2 and SPACE volumes, discs with a Pfirrmann grade. ρ, Spearman correlation between ratio and grade across discs. The within-patient slope is the change in the ratio per grade from a regression of the log ratio on grade with patient fixed effects, expressed in %; its change after the transform is given on the log scale with a 95% confidence interval from 2000 bootstrap resamples of patients. Because the CSF reference is constant within a patient, the within-patient slope does not depend on the CSF region. G5/G2 is the ratio of the median ratio of grade 5 discs to that of grade 2 discs across patients. Order flips count pairs of discs within the same patient whose ratio order reversed after the transform."),
    ("Supplementary Table S8. RSNA replay: error by clipping class, censoring by clipping percentile and by series label.",
     "(A) Relative error (%) of the CSF estimate by share of CSF pixels clipped\n\n" + md_table(S4a)
     + "\n\n(B) Share of series with a censored CSF median by upper clipping percentile\n\n" + md_table(S4b)
     + "\n\n(C) Share of series with a censored CSF median by the series description retained in the DICOM header\n\n" + md_table(RL),
     "Data are median (IQR) [range] in (A) and % with 95% bootstrap confidence intervals (2000 resamples of studies) in (B) and (C), for 1973 RSNA sagittal T2/STIR series with annotated canal points. The CSF region is the union of 2.5-mm discs around the annotated points. The anonymised headers retain no inversion time; 1594 series are described as T2, one as STIR, and 378 carry no description and are likely to include the fat-suppressed series, in which CSF is the brightest tissue. The censored-normal estimate was unstable in these small regions."),
    ("Supplementary Table S9. External replay on RSNA discs: error components, censoring and quality-control levels.",
     "(A) Components of the disc-to-CSF ratio error by series type, % median (IQR), and shares of discs\n\n" + md_table(S10A)
     + "\n\n(B) Discs with an absolute ratio error above 10% by quality-control level\n\n" + md_table(S10B)
     + "\n\n(C) Error and missed errors by disc level, both series types\n\n" + md_table(S10C),
     "Per-disc errors in 7811 discs of 1906 RSNA studies after the inferred transform, with regions fixed in native coordinates. Discs were located by SpineNetV2 and shrunk by 40% towards their centroid; the CSF region is the union of 2.5-mm discs around the annotated canal points. Series type was classified from the images: the within-image percentile, on the mid-sagittal T2/STIR slice, of the pixels that were brightest on the matched T1-weighted slice (mostly fat) was at least 90 in plain T2 and below 87 in fat-suppressed series, thresholds set by visual review of 24 studies across the score range; 65 studies in between and 2 that could not be classified were excluded. A disc median is censored when more than half of the disc region lies at the ceiling. The detector located L5/S1 less often than other levels. QC, quality control."),
    ("Supplementary Table S10. Clipped vertebrae in the released SPIDER volumes.",
     md_table(S5) + "\n\nVertebrae with more than half of the region at the ceiling, stored units after removal of the −1000 offset:\n\n" + md_table(S5b),
     "No ground truth exists for these vertebrae; the comparison is descriptive. Case numbers are sequential and do not correspond to dataset identifiers."),
]
supp = G["supp"]
meth = supp[supp.index("# Supplementary Methods S1."):supp.index("\nSupplementary Table S1.")].rstrip()
meth = meth.replace("# Supplementary Methods S1. Detection metrics and estimators",
                    "# Supplementary Methods S1. Detection metrics, estimators, candidate transforms and quality-control levels")
meth += """

Candidate transforms. Ten candidate mappings were applied to each native SPIDER volume: linear rescaling between the 5th and 95th, 2nd and 98th, 1st and 99th, and 0.5th and 99.5th percentiles of all voxels, with clipping outside the bounds; four variants that kept one bound at the 5th or 95th percentile and moved the other (minimum to 95th, 1st to 95th, 10th to 95th and 5th to 99th percentile); linear rescaling between the minimum and maximum without clipping; and the DICOM display window from the window centre minus half the width to the centre plus half the width. Besides the per-volume envelope, a candidate had to be able to produce the smallest floor share observed in the release (5.000%), which excludes lower bounds above the 5th percentile. Each was mapped to integers 0–4096, and the shares at the ceiling and floor and the lattice occupancy were compared with the converted release.

Per-disc ratio errors and quality-control levels. For every disc of the 50 native T2 and SPACE volumes and every CSF region, the native ratio, the ratio after linear rescaling without clipping and the ratio after the full transform were computed from region medians, with region coordinates fixed from the native image. The incidence of absolute errors above 10%, 20% and 30% is reported with 95% confidence intervals from 2000 bootstrap resamples of patients, and robustness by leaving out one patient at a time. The quality-control level was defined from the share of CSF-region voxels at the ceiling: level 1, none; level 2, up to 50%; level 3, more than 50%. Its sensitivity and specificity for an absolute ratio error above 10% were computed against the replay."""
contents = "\n".join(f"- {t.rstrip('.')}" for t, _, _ in sections)
supp_v2 = ("# Supplementary material\n\nHidden intensity transformations in lumbar MRI: consequences for reference-normalised biomarkers and degeneration-dependent measurement bias\n\nContents\n\n"
           "- Supplementary Methods S1. Detection metrics, estimators, candidate transforms and quality-control levels\n" + contents + "\n\n"
           + meth + "\n\n" + "\n\n".join(f"{t}\n\n{body}\n\n{note}" for t, body, note in sections) + "\n")
(MAN / "supplement_v2.md").write_text(unicode_minus(supp_v2), encoding="utf-8")
print(md_table(T3a)); print(md_table(T3b)); print(md_table(S6B))
