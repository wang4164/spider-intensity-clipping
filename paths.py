"""Locations of the input datasets and of the output folder.

Set them with environment variables (recommended) or edit the defaults below.

  SPIDER_DIR   SPIDER release: images/, masks/, overview.csv, radiological_gradings.csv
  RSNA_DIR     RSNA 2024 Lumbar Spine Degenerative Classification data: train_images/,
               train_series_descriptions.csv, train_label_coordinates.csv
  AISSLAB_DIR  LSS MRI AISSLab dataset, DICOM folder
  SPINENET_DIR clone of https://github.com/rwindsor1/SpineNet (only for rsna_disc_replay.py)
  RESULTS_DIR  where all outputs are written (default: ./results)
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPIDER = Path(os.environ.get("SPIDER_DIR", HERE / "data" / "SPIDER"))
RSNA = Path(os.environ.get("RSNA_DIR", HERE / "data" / "RSNA"))
AISS = Path(os.environ.get("AISSLAB_DIR", HERE / "data" / "AISSLab" / "DICOM"))
SPINENET = Path(os.environ.get("SPINENET_DIR", HERE / "external" / "SpineNet"))
RESULTS = Path(os.environ.get("RESULTS_DIR", HERE / "results"))
T1_REGIONS = RESULTS / "t1_regions"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
for _p in (RESULTS, T1_REGIONS, TABLES, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)
