"""Probe SPIDER stored encodings: value-lattice spacing vs the DICOM window (overview.csv)."""
import numpy as np, pandas as pd, SimpleITK as sitk
from pathlib import Path
from paths import SPIDER as ROOT, RESULTS, T1_REGIONS
ov = pd.read_csv(ROOT / "overview.csv")
rows = []
for r in ov.itertuples():
    f = ROOT / "images" / f"{r.new_file_name}.mha"
    if not f.exists():
        continue
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(f)))
    u = np.unique(a)
    inner = u[(u > u.min()) & (u < u.max())]
    gaps = np.diff(inner)
    def num(x):
        try: return float(str(x).strip("[]").replace('"', '').split("\\")[0].split(",")[0])
        except Exception: return np.nan
    wc, ww = num(r.WindowCenter), num(r.WindowWidth)
    rows.append({"file": r.new_file_name, "dtype": str(a.dtype), "min": float(u.min()), "max": float(u.max()),
                 "n_distinct": int(u.size), "occupancy": float(u.size / (u.max() - u.min() + 1)),
                 "gap_median": float(np.median(gaps)) if gaps.size else np.nan, "gap_mean": float(gaps.mean()) if gaps.size else np.nan,
                 "frac_min": float((a == u.min()).mean()), "frac_max": float((a == u.max()).mean()),
                 "WC": wc, "WW": ww, "pred_gap": 4096 / ww if ww and ww > 0 else np.nan,
                 "manufacturer": r.Manufacturer, "field": r.MagneticFieldStrength, "desc": r.SeriesDescription})
D = pd.DataFrame(rows); D.to_csv(RESULTS / "spider_encoding_probe.csv", index=False)
D["enc"] = np.where((D["min"] == -1000) & (D["max"] == 3096), "-1000_3096", "other")
print(D.groupby("enc")[["occupancy", "gap_mean", "pred_gap", "frac_min", "frac_max", "WW"]].describe().T.round(3).to_string())
print(D[D.enc == "other"][["file", "dtype", "min", "max", "occupancy", "WC", "WW", "manufacturer"]].to_string())
m = D[(D.enc == "-1000_3096") & D.pred_gap.notna()]
print("corr gap_mean vs 4096/WW:", np.corrcoef(m.gap_mean, m.pred_gap)[0, 1].round(3), " n=", len(m))
print(m[["file", "occupancy", "gap_mean", "pred_gap", "WC", "WW", "frac_min", "frac_max"]].head(15).round(3).to_string())
