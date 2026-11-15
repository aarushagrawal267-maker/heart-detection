"""Dataset acquisition, inventory, AAMI mapping, splits and detector evaluation."""
from __future__ import annotations

import threading

import numpy as np
from fastapi import APIRouter, HTTPException

from .. import config
from ..data import acquire, loader, subjects
from ..labels import aami
from ..segmentation import beats
from ..signal import detect, quality
from ..splits import split as split_mod
from .common import cache_json, require_dataset, require_processed

router = APIRouter(prefix="/api", tags=["dataset"])
_TASK = {"running": False, "kind": None, "stage": "idle", "progress": 0.0, "error": None}


def _run(kind: str) -> None:
    def prog(stage, frac):
        _TASK.update(stage=stage, progress=round(frac, 3))
    try:
        if kind == "download":
            acquire.download(progress=prog)
            if not beats.is_built():
                beats.build(progress=prog)
        else:
            beats.build(progress=prog)
        _TASK.update(stage="done", progress=1.0)
    except Exception as exc:  # noqa: BLE001
        _TASK.update(error=str(exc), stage="failed")
    finally:
        _TASK["running"] = False


def _start(kind: str) -> dict:
    if _TASK["running"]:
        raise HTTPException(409, f"A dataset task is already running ({_TASK['kind']})")
    _TASK.update(running=True, kind=kind, stage="starting", progress=0.0, error=None)
    threading.Thread(target=_run, args=(kind,), daemon=True).start()
    return dict(_TASK)


@router.get("/dataset/status", summary="Dataset download / processing status")
def status() -> dict:
    present = acquire.is_present()
    built = beats.is_built() if present else False
    out = {"present": present, "processed": built, "task": dict(_TASK),
           "name": config.DATASET_NAME, "version": config.DATASET_VERSION,
           "n_records_expected": len(config.RECORDS), "n_subjects": subjects.n_subjects(),
           "pipeline_hash": config.pipeline_hash(),
           "provenance": {k: v for k, v in (acquire.provenance() or {}).items() if k != "sha256"} or None}
    if built:
        _, meta, man = beats.load()
        out["n_segments"] = man["n_segments"]
        out["class_counts"] = {k: int(v) for k, v in meta["aami"].value_counts().items()}
    return out


@router.post("/dataset/download", summary="Download and verify MIT-BIH from PhysioNet")
def download() -> dict:
    return _start("download")


@router.post("/dataset/build", summary="Run preprocessing + segmentation pipeline")
def build() -> dict:
    require_dataset()
    return _start("build")


@router.post("/dataset/verify", summary="Re-verify SHA-256 checksums of local files")
def verify() -> dict:
    require_dataset()
    return acquire.verify()


@router.get("/dataset/provenance", summary="Source, licence, citation and checksums")
def provenance() -> dict:
    p = acquire.provenance()
    base = {"dataset": config.DATASET_NAME, "version": config.DATASET_VERSION,
            "source_url": config.DATASET_URL, "doi": config.DATASET_DOI,
            "license": config.DATASET_LICENSE, "citations": config.DATASET_CITATIONS}
    if p:
        base.update({k: v for k, v in p.items() if k != "sha256"})
        base["n_checksums"] = len(p.get("sha256", {}))
    return base


@router.get("/dataset/mapping", summary="MIT-BIH symbol -> AAMI class mapping")
def mapping() -> dict:
    cfg = aami.mapping_config()
    out = {"classes": cfg["classes"], "non_beat_symbols": cfg["non_beat_symbols"],
           "paced_records": cfg["paced_records"], "paced_records_note": cfg["paced_records_note"],
           "primary_task_classes": cfg["primary_task_classes"], "q_class_note": cfg["q_class_note"]}
    if beats.is_built():
        _, meta, man = beats.load()
        sym_total: dict[str, int] = {}
        for r in man["records"]:
            for s, n in r["symbol_counts"].items():
                sym_total[s] = sym_total.get(s, 0) + n
        out["symbol_counts"] = sym_total
        out["n_boundary_excluded"] = man["n_boundary_excluded"]
    return out


def _inventory() -> list[dict]:
    _, meta, man = beats.load()
    part = split_mod.record_partition()
    stats = {r["record"]: r for r in man["records"]}
    paced = set(aami.paced_records())
    rows = []
    for rec in config.RECORDS:
        h = loader.load_header(rec)
        ch = loader.analysis_channel(rec)
        raw = loader.load_signal(rec)[:, ch]
        adc_min = (-(2 ** (h.adc_res[ch] - 1)) - h.adc_zero[ch]) / h.adc_gain[ch]
        adc_max = ((2 ** (h.adc_res[ch] - 1) - 1) - h.adc_zero[ch]) / h.adc_gain[ch]
        q = quality.record_quality(raw, h.fs, adc_min=adc_min, adc_max=adc_max)["summary"]
        m = meta[meta["record"] == rec]
        st = stats[rec]
        rows.append({
            "record": rec, "subject": subjects.subject_of(rec), "partition": part[rec],
            "fold": int(m["fold"].iloc[0]) if len(m) else -1,
            "leads": h.leads, "analysis_lead": h.leads[ch], "has_mlii": "MLII" in h.leads,
            "fs": h.fs, "n_samples": h.n_samples, "duration_s": h.duration_s,
            "comments": h.comments, "paced": rec in paced,
            "n_annotations": st["n_annotations"], "n_beats": st["n_beats"],
            "n_non_beat": st["n_non_beat"], "n_boundary_excluded": st["n_boundary_excluded"],
            "class_counts": {k: int(v) for k, v in m["aami"].value_counts().items()},
            "symbol_counts": st["symbol_counts"], "quality": q,
        })
    return rows


@router.get("/dataset/records", summary="Recording inventory with class counts and quality")
def records() -> list[dict]:
    require_processed()
    return cache_json("inventory.json", _inventory)


@router.get("/splits", summary="Subject-level partition definition, class counts and leakage checks")
def splits() -> dict:
    require_processed()
    _, meta, _ = beats.load()
    return split_mod.describe(meta)


def _detector_eval() -> dict:
    rows = []
    paced = set(aami.paced_records())
    for rec in config.RECORDS:
        sig = np.asarray(beats.load_filtered(rec))
        det = detect.detect_r_peaks(sig, config.EXPECTED_FS)
        ann = loader.load_annotations(rec)
        ref = ann["sample"][np.array([aami.is_beat(s) for s in ann["symbol"]])]
        m = detect.match_peaks(ref, det, config.EXPECTED_FS)
        rows.append({"record": rec, "paced": rec in paced,
                     **{k: v for k, v in m.items() if k not in ("matched_ref", "matched_det")}})
    use = [r for r in rows if not r["paced"]]
    tp, fp, fn = (sum(r[k] for r in use) for k in ("tp", "fp", "fn"))
    return {"records": rows, "tolerance_ms": 150,
            "overall_non_paced": {"tp": tp, "fp": fp, "fn": fn,
                                  "sensitivity": round(tp / (tp + fn), 5),
                                  "positive_predictivity": round(tp / (tp + fp), 5)},
            "detector": "Pan-Tompkins-style detector implemented in app/signal/detect.py"}


@router.get("/detection/evaluation", summary="R-peak detector vs reference annotations (150 ms)")
def detection_evaluation() -> dict:
    require_processed()
    return cache_json("detector_eval.json", _detector_eval)
