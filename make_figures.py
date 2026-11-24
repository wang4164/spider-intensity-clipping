#!/usr/bin/env python
"""Main figures of the paper (reads results/; writes results/figures/).

House figure style (strike-house-style §7): Arial, white background, left/bottom axes only, no gridlines,
no shaded bands, panel titles "(A)  Title" left-aligned bold, Okabe-Ito colours with one meaning each:
  median = blue #0072B2, mean = orange #E69F00, quantile-normal fit = green #009E73, censored-normal MLE = pink #CC79A7,
  grey = reference / native.
Elsevier widths: double column 190 mm.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import SimpleITK as sitk

from paths import SPIDER as ROOT, RESULTS, T1_REGIONS
RES = RESULTS
FIG = RES / "figures"; FIG.mkdir(exist_ok=True)
from spine_regions import infer_anatomical_axes  # noqa: E402

BLUE, ORANGE, GREEN, PINK = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
GREY, DARK, LIGHT = "#969696", "#333333", "#D9D9D9"
EST = [("median", "Median", BLUE), ("mean", "Mean", ORANGE),
       ("quantile_normal", "Quantile-normal fit", GREEN), ("tobit_normal", "Censored-normal MLE", PINK)]
W = 190 / 25.4

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": DARK,
    "text.color": DARK, "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
    "pdf.fonttype": 42, "savefig.dpi": 600})


def title(ax, letter, text):
    ax.set_title(f"({letter})  {text}", loc="left", fontweight="bold", pad=6)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.tif", bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


# ------------------------------------------------------------------ Figure 1: workflow
def figure1():
    fig, ax = plt.subplots(figsize=(W, 2.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    def box(x, y, w, h, head, body):
        # pad=0 so that the drawn outline coincides with (x, y, w, h) and arrows end exactly on the edge
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc="white", ec=DARK, lw=0.8, zorder=2))
        ax.text(x + w / 2, y + h - 1.8, head, ha="center", va="top", fontweight="bold", fontsize=8, zorder=3)
        ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=7, linespacing=1.35, zorder=3)
        return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}

    def arrow(p0, p1, dashed=False):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=0.8, color=DARK,
                                     linestyle=(0, (3, 2)) if dashed else "-", shrinkA=3, shrinkB=3, zorder=1))

    a = box(1, 22, 22, 15, "Public release", "SPIDER lumbar MRI\n447 volumes, 218 patients\n374 in −1000..3096 encoding\n73 native (Siemens)")
    b = box(1, 3, 22, 15, "Native controls", "RSNA 2024: 3954 series\nSecond dataset: 500 series\nSPIDER native: 73 volumes")
    c = box(28, 11.5, 20, 17, "1  Detection", "Share of voxels at the\nstored maximum;\ninteger-lattice occupancy\n374/374 vs 0/4527")
    d = box(53, 11.5, 20, 17, "2  Inference", "Per-volume clip at the\n5th and 95th percentiles,\nlinear rescale to 4096\nlevels (5.0% at ceiling)")
    e = box(78, 22, 21, 15, "3  Forward replay", "Transform applied to\nnative SPIDER and RSNA;\nmarrow, CSF and disc\nregions")
    f = box(78, 3, 21, 15, "4  Estimation", "Median, mean,\nquantile-normal fit,\ncensored-normal MLE;\nratio bias")
    arrow(a["r"], (28, 23)); arrow(b["r"], (28, 17))
    arrow(c["r"], d["l"])
    arrow((73, 23), e["l"]); arrow((73, 17), f["l"], dashed=True)
    arrow(e["b"], f["t"])
    save(fig, "Figure_1")


# ------------------------------------------------------------------ Figure 2: evidence for the transform
def oriented_mid_sagittal(arr, img, lab=None):
    ax3 = infer_anatomical_axes(np.array(img.GetDirection()).reshape(3, 3))
    if lab is not None:
        idx = int(np.argmax((lab == 100).sum(axis=tuple(i for i in range(3) if i != ax3.lr_axis))))
    else:
        idx = arr.shape[ax3.lr_axis] // 2
    def take(a):
        s = np.take(a, idx, axis=ax3.lr_axis)
        rem = [i for i in range(3) if i != ax3.lr_axis]
        if rem != [ax3.si_axis, ax3.ap_axis]:
            s = s.T
        if ax3.si_sign > 0:
            s = s[::-1, :]
        if ax3.ap_sign < 0:
            s = s[:, ::-1]
        return s
    sp = img.GetSpacing()[::-1]
    return take(arr), (take(lab) if lab is not None else None), sp[ax3.si_axis], sp[ax3.ap_axis]


def figure2():
    P = pd.read_csv(RES / "spider_encoding_probe.csv")
    P["conv"] = (P["min"] == -1000) & (P["max"] == 3096)
    A = pd.read_csv(RES / "external_aisslab_series.csv"); R = pd.read_csv(RES / "external_rsna_series_n0.csv")
    ex = "171_t2"
    img = sitk.ReadImage(str(ROOT / "images" / f"{ex}.mha")); arr = sitk.GetArrayFromImage(img)
    lab = sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT / "masks" / f"{ex}.mha")))

    fig = plt.figure(figsize=(W, 5.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 0.85], hspace=0.55, wspace=0.42)

    # (A) histogram of stored values with the two spikes
    ax = fig.add_subplot(gs[0, 0:2])
    v = arr.ravel()
    inner = v[(v > -1000) & (v < 3096)]
    ax.hist(inner, bins=np.arange(-1000, 3097, 64), color=GREY, lw=0)
    for x0, col in [(-1000, ORANGE), (3096, ORANGE)]:
        ax.bar(x0, (v == x0).sum(), width=40, color=col, lw=0)
    ax.set_yscale("log"); ax.set_xlim(-1150, 3250); ax.set_ylim(top=(v == -1000).sum() * 40)
    ax.set_xlabel("Stored voxel value"); ax.set_ylabel("Voxels per bin (log scale)")
    for x0, txt, ha, dx in [(-1000, f"floor −1000: {100*(v == -1000).mean():.1f}% of voxels", "left", 60),
                             (3096, f"ceiling 3096: {100*(v == 3096).mean():.1f}% of voxels", "right", -60)]:
        ax.text(0.02 if ha == "left" else 0.98, 0.93, txt, transform=ax.transAxes, ha=ha, va="top", fontsize=7, color=DARK)
    ins = ax.inset_axes([0.36, 0.60, 0.28, 0.22])
    u, c = np.unique(v[(v >= 1000) & (v < 1080)], return_counts=True)
    ins.vlines(u, 0, c, color=DARK, lw=0.8); ins.set_xlim(1000, 1080)
    ins.set_title("used values 1000–1080", fontsize=6.5, pad=2); ins.tick_params(labelsize=6)
    ins.set_yticks([])
    title(ax, "A", f"Stored values of one converted volume (patient {ex.split('_')[0]}, T2)")

    # (B) share at ceiling, converted vs native
    ax = fig.add_subplot(gs[0, 2])
    groups = [("SPIDER\nconverted", P[P.conv].frac_max, ORANGE), ("SPIDER\nnative", P[~P.conv].frac_max, GREY),
              ("RSNA", R.frac_at_max, GREY), ("AISSLab", A.frac_at_max, GREY)]
    rng = np.random.default_rng(1)
    for i, (name, s, col) in enumerate(groups):
        y = np.clip(s.values, 1e-8, None) * 100
        ax.scatter(i + rng.uniform(-0.25, 0.25, len(y)), y, s=3, color=col, alpha=0.5, lw=0, rasterized=True)
    ax.axhline(0.01, color=DARK, lw=0.6, ls="--")
    ax.text(3.45, 0.013, "0.01%", fontsize=6.5, ha="right", va="bottom")
    ax.set_yscale("log"); ax.set_xticks(range(4))
    ax.set_xticklabels(["SPIDER\nconverted", "SPIDER\nnative", "RSNA", "Second\ndataset"], fontsize=6.5)
    ax.set_ylabel("Voxels at the stored maximum (%)"); ax.set_xlim(-0.6, 3.6)
    title(ax, "B", "Ceiling pile-up")

    # (C) lattice spacing vs window width
    ax = fig.add_subplot(gs[1, 0])
    c = P[P.conv & P.pred_gap.notna()]
    ax.scatter(c.pred_gap, c.gap_mean, s=6, color=ORANGE, alpha=0.7, lw=0)
    r = np.corrcoef(c.pred_gap, c.gap_mean)[0, 1]
    ax.text(0.04, 0.95, f"r = {r:.2f}, n = {len(c)}", transform=ax.transAxes, va="top", fontsize=7)
    ax.set_xlabel("4096 / DICOM window width"); ax.set_ylabel("Mean spacing between used values")
    title(ax, "C", "Values are stretched")

    # (D) share at ceiling vs floor per converted volume
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(100 * c.frac_min, 100 * c.frac_max, s=6, color=ORANGE, alpha=0.7, lw=0)
    ax.set_ylim(4.9, 5.2); ax.set_xlabel("Voxels at the floor (%)"); ax.set_ylabel("Voxels at the ceiling (%)")
    ax.text(0.97, 0.95, "every converted volume:\n5.00–5.12% at the ceiling", transform=ax.transAxes, ha="right", va="top", fontsize=7)
    title(ax, "D", "A fixed 5% at the ceiling")

    # (E) image with saturated voxels
    ax = fig.add_subplot(gs[1, 2])
    s2, l2, ssi, sap = oriented_mid_sagittal(arr.astype(float), img, lab)
    ax.imshow(s2, cmap="gray", aspect=ssi / sap, vmin=-1000, vmax=3096, interpolation="nearest")
    sat = np.ma.masked_where(s2 < 3096, np.ones_like(s2))
    ax.imshow(sat, cmap=matplotlib.colors.ListedColormap([ORANGE]), aspect=ssi / sap, alpha=0.85, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    title(ax, "E", "Saturated voxels")
    save(fig, "Figure_2")


# ------------------------------------------------------------------ Figure 3: simulation within SPIDER
def err_panel(ax, df, letter, text, ylim):
    for (key, name, col), dx in zip(EST, (-1.5, -0.5, 0.5, 1.5)):     # small horizontal dodge so coincident estimators stay visible
        g = df.groupby("f_target")[f"err_{key}"]
        med, lo, hi = 100 * g.median(), 100 * g.quantile(.25), 100 * g.quantile(.75)
        x = 100 * med.index.values + dx
        ax.errorbar(x, med, yerr=[med - lo, hi - med], color=col, lw=1.2, marker="o", ms=3, capsize=1.5, elinewidth=0.6, label=name)
    ax.axhline(0, color=GREY, lw=0.6, ls="--"); ax.axvline(50, color=LIGHT, lw=0.8)
    ax.text(51, ylim[0] + 2, "half of ROI clipped", fontsize=6.5, color=GREY, va="bottom")
    ax.set_ylim(*ylim); ax.set_xlim(-3, 84)
    ax.set_xlabel("Voxels clipped in the ROI (%)"); ax.set_ylabel("Error of the ROI estimate (%)")
    title(ax, letter, text)


def figure3():
    B = pd.read_csv(RES / "partB_t1_sim_vertebra.csv"); C = pd.read_csv(RES / "partC_t2_csf_sim.csv")
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"wspace": 0.3})
    err_panel(axs[0], B, "A", f"Vertebral marrow, T1 (n = {B[['patient_id','level']].drop_duplicates().shape[0]} ROIs)", (-22, 6))
    err_panel(axs[1], C, "B", f"CSF, T2 (n = {C.series.nunique()} series)", (-50, 6))
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    save(fig, "Figure_3")


# ------------------------------------------------------------------ Figure 4: replay on native data and CSF ROI choice
def figure4():
    R = pd.read_csv(RES / "rsna_spider_transform_n0.csv")
    S = pd.read_csv(RES / "csf_roi_sensitivity.csv")
    rng = np.random.default_rng(20261123)
    fig = plt.figure(figsize=(W, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.3)

    # (A) error vs fraction clipped, RSNA replay
    ax = fig.add_subplot(gs[0, 0])
    edges = np.array([0, 1e-9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.0001])
    fc = 100 * R.roi_frac_clipped
    bins = pd.cut(fc, edges, right=False, include_lowest=True)
    centres = [0] + [5 + 10 * i for i in range(10)]
    for (key, name, col), dx in zip([("median", "Median", BLUE), ("mean", "Mean", ORANGE)], (-1.2, 1.2)):
        g = (100 * R[f"err_{key}"]).groupby(bins, observed=False)
        med, lo, hi = g.median().values, g.quantile(.25).values, g.quantile(.75).values
        ok = ~np.isnan(med)
        xs = np.array(centres)[ok] + dx
        ax.errorbar(xs, med[ok], yerr=[med[ok] - lo[ok], hi[ok] - med[ok]], color=col, lw=1.2, marker="o", ms=3,
                    capsize=1.5, elinewidth=0.6, label=name)
    n_per = bins.value_counts(sort=False).values
    ax.axvline(50, color=LIGHT, lw=0.8); ax.axhline(0, color=GREY, lw=0.6, ls="--")
    ax.set_xlim(-4, 102); ax.set_ylim(-82, 6)
    ax.set_xlabel("CSF ROI pixels clipped (%)"); ax.set_ylabel("Error of the CSF estimate (%)")
    ax.legend(frameon=False, loc="lower left")
    title(ax, "A", f"RSNA T2 after the transform (n = {len(R)})")

    # (B) censored share by clipping percentile, bootstrap CI
    ax = fig.add_subplot(gs[0, 1])
    qs = [0.95, 0.98, 0.99, 0.995, 0.999]
    est, lo, hi = [], [], []
    for q in qs:
        x = (R[f"roi_frac_clipped_q{q}"] > .5).values.astype(float)
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        est.append(100 * x.mean()); lo.append(100 * np.percentile(bs, 2.5)); hi.append(100 * np.percentile(bs, 97.5))
    xpos = np.arange(len(qs)); est, lo, hi = map(np.array, (est, lo, hi))
    ax.errorbar(xpos, est, yerr=[est - lo, hi - est], color=BLUE, marker="s", ms=4, lw=1.2, capsize=2, elinewidth=0.8)
    for xp, e in zip(xpos, est):
        ax.text(xp + 0.12, e + 1.5, f"{e:.1f}%", ha="left", va="bottom", fontsize=7)
    ax.set_xticks(xpos); ax.set_xticklabels([f"p{100*q:g}" for q in qs])
    ax.set_xlim(-0.3, len(qs) - 0.4); ax.set_ylim(0, 68); ax.set_xlabel("Upper clipping percentile"); ax.set_ylabel("Series with a censored CSF median (%)")
    title(ax, "B", "Where the clip is placed")

    # (C) censored share by CSF ROI definition, observed SPIDER release
    ax = fig.add_subplot(gs[1, 0])
    rois = [("whole", "Whole\ncanal"), ("ero1", "Eroded\n1 mm"), ("ero2", "Eroded\n2 mm"), ("core", "Pure CSF\ncore")]
    sets = [("2D T2", S[(S.kind == "t2") & S.converted], BLUE), ("3D SPACE", S[(S.kind == "SPACE") & S.converted], ORANGE)]
    wbar = 0.36
    for j, (name, d, col) in enumerate(sets):
        vals = [100 * (d[f"f_{k}"].dropna() > .5).mean() for k, _ in rois]
        xs = np.arange(len(rois)) + (j - 0.5) * wbar
        ax.bar(xs, vals, width=wbar - 0.03, color=col, label=f"{name} (n = {len(d)})")
        for xb, vb in zip(xs, vals):
            ax.text(xb, vb + 1.5, f"{vb:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(rois))); ax.set_xticklabels([r[1] for r in rois])
    ax.set_ylim(0, 105); ax.set_ylabel("Series with a censored CSF median (%)")
    ax.legend(frameon=False, loc="upper left")
    title(ax, "C", "Released SPIDER, by CSF region")

    # (D) disc/CSF ratio total error by ROI, native forward 2D T2
    ax = fig.add_subplot(gs[1, 1])
    d = S[(S.kind == "t2") & ~S.converted]
    for i, (k, lbl) in enumerate(rois):
        for j, (col_key, col, off) in enumerate([(f"ratio_err_median_{k}", BLUE, -0.15), (f"ratio_err_qn_{k}", GREEN, 0.15)]):
            y = 100 * d[col_key].dropna().values
            ax.scatter(i + off + rng.uniform(-0.06, 0.06, len(y)), y, s=8, color=col, alpha=0.75, lw=0,
                       label=(["CSF median", "Quantile-normal CSF"][j] if i == 0 else None))
            ax.hlines(np.median(y), i + off - 0.11, i + off + 0.11, color=DARK, lw=1.2)
    ax.axhline(0, color=GREY, lw=0.6, ls="--")
    ax.set_xticks(range(len(rois))); ax.set_xticklabels([r[1] for r in rois])
    ax.set_ylabel("Error of the disc/CSF ratio (%)")
    ax.legend(frameon=False, loc="upper left")
    title(ax, "D", f"Disc/CSF ratio, native T2 (n = {len(d)})")
    save(fig, "Figure_5")


# ------------------------------------------------------------------ Figure 5: degeneration-dependent offset and the screening blind spot
def figure5():
    D = pd.read_csv(RES / "ratio_incidence_discs.csv")
    rng = np.random.default_rng(5)
    fig = plt.figure(figsize=(W, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1.05], wspace=0.42)

    # (A) offset component by Pfirrmann grade, eroded 1 mm canal
    ax = fig.add_subplot(gs[0, 0])
    d = D[(D.roi == "ero1")].dropna(subset=["pfirrmann"])
    for (kind, col), dx in zip([("2D T2", BLUE), ("SPACE", ORANGE)], (-0.17, 0.17)):
        for g in range(1, 6):
            y = 100 * d[(d.kind == kind) & (d.pfirrmann == g)].err_offset.values
            ax.scatter(g + dx + rng.uniform(-0.07, 0.07, len(y)), y, s=5, color=col, alpha=0.45, lw=0,
                       label=kind if g == 1 else None)
            ax.hlines(np.median(y), g + dx - 0.12, g + dx + 0.12, color=DARK, lw=1.2)
    ax.axhline(0, color=GREY, lw=0.6, ls="--")
    ax.set_xticks(range(1, 6)); ax.set_xlabel("Pfirrmann grade"); ax.set_ylabel("Offset component of ratio error (%)")
    ax.set_ylim(-36, 4)
    ax.legend(frameon=False, loc="lower left", markerscale=2)
    title(ax, "A", "Offset error by grade")

    # (B) offset component against the native ratio (2D, eroded canal), coloured by grade: darker discs lose more
    ax = fig.add_subplot(gs[0, 1])
    e = D[(D.roi == "ero1") & (D.kind == "2D T2")].dropna(subset=["pfirrmann"])
    shades = {1: "#C6DBEF", 2: "#9ECAE1", 3: "#6BAED6", 4: "#3182BD", 5: "#08519C"}
    for g in range(1, 6):
        s = e[e.pfirrmann == g]
        ax.scatter(s.ratio_native, 100 * s.err_offset, s=9, color=shades[g], lw=0.3, edgecolor="white", label=f"{g}")
    ax.axhline(0, color=GREY, lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.4]); ax.set_xticklabels(["0.05", "0.1", "0.2", "0.4"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Native disc-to-CSF ratio (log scale)"); ax.set_ylabel("Offset component (%)")
    ax.legend(title="Grade", frameon=False, loc="lower right", fontsize=6, title_fontsize=6.5, markerscale=1.3, ncol=1)
    title(ax, "B", "Offset vs disc signal")

    # (C) discs with |error| > 10% by QC level, split into offset-driven and clipping-driven
    ax = fig.add_subplot(gs[0, 2])
    labels, off, clip, x = [], [], [], []
    for gi, (roi, name) in enumerate([("whole", "Whole\ncanal"), ("ero1", "Eroded\n1 mm"), ("core", "Core\nCSF")]):
        e = D[D.roi == roi]
        lvl3 = e.csf_clipped > 0.5
        bad = e.err_total.abs() > 0.10
        offd = bad & (e.err_offset.abs() > 0.10)
        for j, (flag, tag) in enumerate([(~lvl3, "L2"), (lvl3, "L3")]):
            labels.append(tag); x.append(gi * 2.6 + j)
            off.append(100 * (offd & flag).sum() / flag.sum()); clip.append(100 * ((bad & ~offd) & flag).sum() / flag.sum())
        ax.text(gi * 2.6 + 0.5, -11, name, ha="center", va="top", fontsize=6.5, transform=ax.transData, clip_on=False)
    ax.bar(x, off, color=GREY, width=0.8, label="Offset-driven")
    ax.bar(x, clip, bottom=off, color=ORANGE, width=0.8, label="Clipping-driven")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Discs with |ratio error| >10% (%)"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    title(ax, "C", "What screening sees")
    save(fig, "Figure_4")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for w in which:
        {"1": figure1, "2": figure2, "3": figure3, "4": figure4, "5": figure5}[w]()
        print("Figure", w, "done")
