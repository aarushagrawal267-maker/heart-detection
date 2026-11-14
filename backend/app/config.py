"""Central configuration: paths and pipeline constants."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("QRS_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("QRS_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw" / "mitdb" / "1.0.0"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.environ.get("QRS_MODELS_DIR", ROOT / "models"))
CONFIGS_DIR = ROOT / "configs"
DB_PATH = Path(os.environ.get("QRS_DB_PATH", DATA_DIR / "qrs_atlas.db"))

DATASET_NAME = "MIT-BIH Arrhythmia Database"
DATASET_VERSION = "1.0.0"
DATASET_URL = "https://physionet.org/content/mitdb/1.0.0/"
DATASET_FILES_URL = "https://physionet.org/files/mitdb/1.0.0/"
DATASET_ZIP_URL = (
    "https://physionet.org/static/published-projects/mitdb/"
    "mit-bih-arrhythmia-database-1.0.0.zip"
)
DATASET_DOI = "https://doi.org/10.13026/C2F305"
DATASET_LICENSE = "Open Data Commons Attribution License v1.0 (ODC-By 1.0)"
DATASET_CITATIONS = [
    "Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. "
    "IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).",
    "Goldberger A, Amaral L, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet: "
    "Components of a new research resource for complex physiologic signals. "
    "Circulation 101(23):e215-e220 (2000).",
]

RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "111", "112",
    "113", "114", "115", "116", "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210", "212", "213", "214", "215",
    "217", "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234",
]
EXPECTED_FS = 360
EXPECTED_SAMPLES = 650000
PRIMARY_LEAD = "MLII"

# ---- preprocessing / segmentation defaults (documented in docs/METHODOLOGY.md) ----
PIPELINE = {
    "version": 1,
    "lead": PRIMARY_LEAD,
    "baseline_median_ms": [200, 600],   # two-stage median filter (de Chazal 2004)
    "lowpass_hz": 45.0,                 # zero-phase Butterworth
    "lowpass_order": 4,
    "store_pre": 128,                   # samples stored before the annotation
    "store_post": 192,                  # samples stored after the annotation (margin for shift augmentation)
    "rr_local_beats": 10,               # +/- beats used for the local RR median
}
DEFAULT_WINDOW = {"pre": 90, "post": 166}   # 256 samples = 0.711 s at 360 Hz


def pipeline_hash(cfg: dict | None = None) -> str:
    cfg = cfg or PIPELINE
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def processed_dir(cfg: dict | None = None) -> Path:
    return PROCESSED_DIR / pipeline_hash(cfg)


def load_json(name: str) -> dict:
    with open(CONFIGS_DIR / name, encoding="utf-8") as fh:
        return json.load(fh)
