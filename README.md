# Hidden intensity transformations in lumbar MRI: analysis code

This repository holds the code for the paper:

> Hidden intensity transformations in lumbar MRI: consequences for reference-normalised biomarkers and degeneration-dependent measurement bias. [Authors]. Biomedical Signal Processing and Control (submitted).

The code does four things:

- It detects ceiling pile-up in stored MRI volumes.
- It identifies the per-volume 5th–95th percentile clip in the SPIDER release by replaying ten candidate transforms on native volumes.
- It quantifies how that clip changes region estimates and disc-to-CSF ratios. This uses simulation within SPIDER, forward replay on native SPIDER volumes, and replay on the RSNA 2024 lumbar series.
- It evaluates a clipping-based screening rule.

No image data are included. All three datasets are public and must be downloaded from their original sources.

## Data

| Dataset | Source | Used for |
|---|---|---|
| SPIDER lumbar spine MRI (images, masks, `overview.csv`, `radiological_gradings.csv`) | van der Graaf et al., Sci. Data 11 (2024) 264, https://doi.org/10.1038/s41597-024-03090-w | Detection, transform identification, simulation, forward replay |
| RSNA 2024 Lumbar Spine Degenerative Classification (`train_images/`, `train_series_descriptions.csv`, `train_label_coordinates.csv`) | Richards et al., Radiol. Artif. Intell. (2026), https://doi.org/10.1148/ryai.250480 | Detector negatives, replay of the transform, per-disc external replay |
| LSS MRI AISSLab (DICOM) | Mendeley Data, https://doi.org/10.17632/rgb77xm3jf.4 | Detector negatives |

Tell the code where the data are with environment variables. The defaults are folders under `./data`; see `paths.py`.

```
SPIDER_DIR=/path/to/SPIDER          # contains images/, masks/, overview.csv, radiological_gradings.csv
RSNA_DIR=/path/to/rsna-2024         # contains train_images/, train_series_descriptions.csv, train_label_coordinates.csv
AISSLAB_DIR=/path/to/AISSLab/DICOM
SPINENET_DIR=/path/to/SpineNet      # clone of https://github.com/rwindsor1/SpineNet (per-disc replay only)
RESULTS_DIR=./results               # optional
```

## Environment

The analysis used Python 3.12, with the package versions listed in `requirements.txt`.

```
pip install -r requirements.txt
```

The per-disc RSNA replay (`rsna_disc_replay.py`) needs SpineNetV2 (Windsor et al., arXiv:2205.01683), used unchanged at commit `5e96ac5`, with PyTorch and a CUDA GPU. Install it in a separate environment, following its own instructions.

## Running the analysis

Run the scripts from the repository folder in the order below. Each one writes to `results/`.

| Step | Script | Output | Paper |
|---|---|---|---|
| 1 | `probe_encoding.py` | `spider_encoding_probe.csv` | Table 1, Fig. 2 |
| 2 | `build_t1_regions.py` | `t1_regions/` (T1 marrow regions, L3 canal-fluid reference) | Methods 2.4 |
| 3 | `t2_canal_ceiling.py` | `t2_canal_ceiling.csv` | Results 3.1 |
| 4 | `run_clipping_pilot.py` | simulation within SPIDER (Parts A–D) | Fig. 3, Tables S1, S4, S10 |
| 5 | `candidate_transforms.py` | ten candidate transforms | Table S2 |
| 6 | `audit_external.py 0` | native RSNA and AISSLab series | Table 1, Table S3 |
| 7 | `detector_sweep.py` | detection across thresholds | Table S3 |
| 8 | `forward_spider_native.py` | replay on native SPIDER volumes | Table 2, Table S5 |
| 9 | `csf_roi_sensitivity.py` | four CSF region definitions | Table 2, Fig. 5C–D |
| 10 | `ratio_incidence.py`, `analyse_incidence.py` | per-disc ratio errors, QC levels | Table 3A–B, Fig. 4C, Table S6 |
| 11 | `grade_association.py` | ratio and Pfirrmann grade | Fig. 4A–B, Table S7 |
| 12 | `simulate_rsna.py 0` | transform replayed on RSNA series | Fig. 5A–B, Table S8 |
| 13 | `rsna_series_labels.py` | series descriptions from DICOM headers | Table S8C |
| 14 | `rsna_t2_type_screen.py` | plain T2 vs fat-suppressed classification | Methods 2.6 |
| 15 | `rsna_disc_replay.py` (SpineNet environment), `analyse_rsna_disc.py` | per-disc external replay | Table 3C, Table S9 |
| 16 | `build_tables.py`, `build_tables_v2.py`, `make_figures.py` | all tables and figures | |

`rsna_disc_replay.py` saves its progress every 50 series and resumes where it stopped. `--batch N` stops after N new series. The full run took about 80 minutes on one GPU.

`spine_regions.py` holds the T1 region definitions: intensity baseline, vertebral body, 2-mm-eroded trabecular region and L3 canal-fluid reference. `paths.py` holds the data locations.

## Licence

The code is released under the MIT licence (see `LICENSE`). The datasets remain under their own terms.

## Citation

If you use this code, please cite the paper above and this repository (see `CITATION.cff`).
