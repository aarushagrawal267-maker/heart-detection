"""Model inference on real recordings (single beats and full records)."""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
import torch

from .. import config
from ..data import loader
from ..db import models as db
from ..evaluation import metrics as M
from ..explain import attribution
from ..labels import aami
from ..segmentation import beats
from ..signal import detect, filters
from ..training import data as D
from ..training import trainer

RESEARCH_NOTICE = ("Research-model output for education and research only. "
                   "Not a medical diagnosis and not for clinical use.")


class NoModel(Exception):
    pass


def active_experiment_id() -> str | None:
    exp_id = db.get_setting("active_experiment_id")
    with db.session() as s:
        if exp_id:
            exp = s.get(db.Experiment, exp_id)
            if (exp and exp.status == "completed" and exp.kind == "cnn" and exp.checkpoint_path
                    and trainer.resolve_path(exp.checkpoint_path).exists()):
                return exp_id
        # fallback: best completed CNN by validation macro-F1 (validation, never test)
        best, best_f1 = None, -1.0
        for exp in s.query(db.Experiment).filter_by(kind="cnn", status="completed").all():
            f1 = (exp.summary or {}).get("val_macro_f1") or -1
            if exp.checkpoint_path and trainer.resolve_path(exp.checkpoint_path).exists() and f1 > best_f1:
                best, best_f1 = exp.id, f1
        return best


@lru_cache(maxsize=4)
def _load(exp_id: str):
    with db.session() as s:
        exp = s.get(db.Experiment, exp_id)
        if exp is None or exp.kind != "cnn" or not exp.checkpoint_path:
            raise NoModel(f"Experiment {exp_id} has no CNN checkpoint")
        path = exp.checkpoint_path
    file = trainer.resolve_path(path)
    if not file.is_file():
        raise NoModel(f"Model file for experiment {exp_id} is missing ({file}). Restore it from "
                      "git (models/<id>/model.pt) or train a new model in the Model Laboratory.")
    try:
        model, scaler, ck = trainer.load_checkpoint(path)
    except Exception as exc:  # noqa: BLE001 - corrupt/incompatible file -> clear message
        raise NoModel(f"Model file for experiment {exp_id} could not be loaded ({file}): "
                      f"{type(exc).__name__}: {exc}") from exc
    if ck.get("pipeline_hash", config.pipeline_hash(ck["pipeline"])) != config.pipeline_hash():
        raise NoModel(f"Experiment {exp_id} was trained with a different preprocessing pipeline "
                      "than the current beat set; retrain it before using it for inference.")
    return model, scaler, ck


def get_model(exp_id: str | None = None):
    exp_id = exp_id or active_experiment_id()
    if not exp_id:
        raise NoModel("No trained CNN is available yet (or its models/<id>/model.pt file is "
                      "missing). Train one in the Model Laboratory.")
    model, scaler, ck = _load(exp_id)
    return exp_id, model, scaler, ck


def _record_rows(record: str):
    segs, meta, _ = beats.load()
    rows = np.flatnonzero((meta["record"] == record).to_numpy())
    return segs, meta, rows


def _partition_note(partition: str) -> str:
    return {
        "train": "This recording belongs to the TRAINING pool. Predictions here are not "
                 "evidence of generalisation to new patients.",
        "test": "This recording belongs to the HELD-OUT test subjects (never used for "
                "training or model selection).",
        "excluded": "This recording is EXCLUDED from training and evaluation (paced rhythm, "
                    "AAMI EC57). Predictions are outside the model's intended scope.",
    }.get(partition, "")


def classify_beat(record: str, beat_index: int, exp_id: str | None = None,
                  explain_method: str | None = "integrated_gradients", log: bool = True) -> dict:
    exp_id, model, scaler, ck = get_model(exp_id)
    cfg = ck["config"]
    classes = cfg["classes"]
    segs, meta, rows = _record_rows(record)
    match = rows[(meta.iloc[rows]["beat_index"] == beat_index).to_numpy()]
    if len(match) == 0:
        raise KeyError(f"Beat {beat_index} not found in record {record}")
    i = int(match[0])
    row = meta.iloc[i]
    pre, post = cfg["window_pre"], cfg["window_post"]
    w = D.windows(segs, np.array([i]), ck["pipeline"], pre, post)
    rr_raw = row[beats.RR_FEATURES].to_numpy(np.float64)[None, :]
    x, rr = D.prepare_inputs(w, rr_raw, scaler)
    t0 = time.perf_counter()
    proba = trainer.predict_proba(model, x, rr)[0]
    latency = (time.perf_counter() - t0) * 1000
    pred = int(proba.argmax())
    ref_cls = row["aami"]
    fs = config.EXPECTED_FS
    out = {
        "record": record, "beat_index": int(beat_index), "sample": int(row["sample"]),
        "time_s": float(row["time_s"]), "lead": row["lead"],
        "reference": {"symbol": row["symbol"], "description": aami.describe(row["symbol"]),
                      "aami": ref_cls, "in_model_classes": ref_cls in classes},
        "prediction": {"class": classes[pred], "confidence": round(float(proba[pred]), 5),
                       "scores": {c: round(float(p), 5) for c, p in zip(classes, proba)}},
        "agrees_with_reference": (classes[pred] == ref_cls) if ref_cls in classes else None,
        "rr": {k: round(float(v), 4) for k, v in zip(beats.RR_FEATURES, rr_raw[0])},
        "waveform_mv": [round(float(v), 4) for v in w[0]],
        "model_input": [round(float(v), 4) for v in x[0, 0]],
        "time_ms": [round((k - pre) / fs * 1000, 2) for k in range(pre + post)],
        "window": {"pre": pre, "post": post, "fs": fs},
        "partition": row["partition"], "partition_note": _partition_note(row["partition"]),
        "experiment_id": exp_id, "latency_ms": round(latency, 3), "notice": RESEARCH_NOTICE,
        "explanation": None,
    }
    if explain_method:
        out["explanation"] = attribution.explain(model, x[0, 0], rr[0], pred, explain_method)
        out["explanation"]["target_class"] = classes[pred]
        out["explanation"]["rr_features"] = beats.RR_FEATURES
        out["explanation"]["caveat"] = ("Attribution maps are analytical visualisations of the "
                                        "model's sensitivity, not proof of clinical reasoning.")
    if log:
        with db.session() as s:
            s.add(db.ClassificationEvent(kind="beat", record=record, beat_index=int(beat_index),
                                         experiment_id=exp_id, predicted=classes[pred],
                                         reference=ref_cls, confidence=float(proba[pred])))
            s.commit()
    return out


def explain_beat(record: str, beat_index: int, method: str, target: str | None = None,
                 exp_id: str | None = None) -> dict:
    exp_id, model, scaler, ck = get_model(exp_id)
    cfg = ck["config"]
    segs, meta, rows = _record_rows(record)
    match = rows[(meta.iloc[rows]["beat_index"] == beat_index).to_numpy()]
    if len(match) == 0:
        raise KeyError(f"Beat {beat_index} not found in record {record}")
    i = int(match[0])
    w = D.windows(segs, np.array([i]), ck["pipeline"], cfg["window_pre"], cfg["window_post"])
    rr_raw = meta.iloc[i][beats.RR_FEATURES].to_numpy(np.float64)[None, :]
    x, rr = D.prepare_inputs(w, rr_raw, scaler)
    if target is None:
        t = int(trainer.predict_proba(model, x, rr)[0].argmax())
    elif target in cfg["classes"]:
        t = cfg["classes"].index(target)
    else:
        raise ValueError(f"target must be one of {cfg['classes']}")
    out = attribution.explain(model, x[0, 0], rr[0], t, method)
    out["target_class"] = cfg["classes"][t]
    out["rr_features"] = beats.RR_FEATURES
    return out


def sanity_check(record: str, beat_index: int, exp_id: str | None = None) -> dict:
    exp_id, model, scaler, ck = get_model(exp_id)
    cfg = ck["config"]
    segs, meta, rows = _record_rows(record)
    match = rows[(meta.iloc[rows]["beat_index"] == beat_index).to_numpy()]
    if len(match) == 0:
        raise KeyError(f"Beat {beat_index} not found in record {record}")
    i = int(match[0])
    w = D.windows(segs, np.array([i]), ck["pipeline"], cfg["window_pre"], cfg["window_post"])
    rr_raw = meta.iloc[i][beats.RR_FEATURES].to_numpy(np.float64)[None, :]
    x, rr = D.prepare_inputs(w, rr_raw, scaler)
    t = int(trainer.predict_proba(model, x, rr)[0].argmax())
    return attribution.randomization_sanity(model, x[0, 0], rr[0], t)


def analyze_record(record: str, mode: str = "reference", exp_id: str | None = None,
                   log: bool = True) -> dict:
    """Classify every beat of a recording.

    mode='reference': beats at the expert annotation positions (principal pipeline).
    mode='detected':  beats found by the project's own R-peak detector; each detection is
                      matched to a reference beat within 150 ms for comparison.
    """
    exp_id, model, scaler, ck = get_model(exp_id)
    cfg = ck["config"]
    classes = cfg["classes"]
    pipeline = ck["pipeline"]
    pre, post = cfg["window_pre"], cfg["window_post"]
    fs = config.EXPECTED_FS
    segs, meta, rows = _record_rows(record)
    rmeta = meta.iloc[rows]
    partition = rmeta["partition"].iloc[0] if len(rmeta) else "excluded"
    t0 = time.perf_counter()
    detection = None
    if mode == "reference":
        w = D.windows(segs, rows, pipeline, pre, post)
        rr_raw = rmeta[beats.RR_FEATURES].to_numpy(np.float64)
        samples = rmeta["sample"].to_numpy()
        ref_cls = rmeta["aami"].tolist()
        ref_sym = rmeta["symbol"].tolist()
        beat_idx = rmeta["beat_index"].tolist()
    elif mode == "detected":
        sig = np.asarray(beats.load_filtered(record, config.pipeline_hash(pipeline)))
        det = detect.detect_r_peaks(sig, fs)
        rr_all = beats.rr_features(det, fs, pipeline["rr_local_beats"])
        cut, valid = beats.extract_segments(sig, det, pre, post)
        samples = det[valid]
        rr_raw = rr_all[valid]
        w = cut
        ann = loader.load_annotations(record)
        is_b = np.array([aami.is_beat(s) for s in ann["symbol"]])
        ref_samples = ann["sample"][is_b]
        ref_syms = np.array(ann["symbol"])[is_b]
        m = detect.match_peaks(ref_samples, det, fs)
        # map each kept detection to the nearest reference beat within tolerance
        tol = int(0.15 * fs)
        pos = np.searchsorted(ref_samples, samples)
        ref_cls, ref_sym, beat_idx = [], [], []
        for s, p in zip(samples, pos):
            best = None
            for q in (p - 1, p):
                if 0 <= q < len(ref_samples) and abs(int(ref_samples[q]) - int(s)) <= tol:
                    if best is None or abs(int(ref_samples[q]) - int(s)) < abs(int(ref_samples[best]) - int(s)):
                        best = q
            if best is None:
                ref_cls.append(None); ref_sym.append(None); beat_idx.append(None)
            else:
                ref_cls.append(aami.to_aami(ref_syms[best])); ref_sym.append(str(ref_syms[best]))
                beat_idx.append(int(best))
        detection = {k: v for k, v in m.items() if k not in ("matched_ref", "matched_det")}
        missed = ref_samples[~m["matched_ref"]]
        detection["missed_reference_samples"] = [int(v) for v in missed[:500]]
    else:
        raise ValueError("mode must be 'reference' or 'detected'")
    x, rr = D.prepare_inputs(w, rr_raw, scaler)
    proba = trainer.predict_proba(model, x, rr)
    elapsed = time.perf_counter() - t0
    pred = proba.argmax(1)
    beats_out = []
    for k in range(len(samples)):
        beats_out.append({
            "beat_index": beat_idx[k], "sample": int(samples[k]),
            "time_s": round(float(samples[k]) / fs, 4),
            "reference": ref_cls[k], "symbol": ref_sym[k],
            "predicted": classes[int(pred[k])], "confidence": round(float(proba[k, pred[k]]), 4),
            "scores": [round(float(p), 4) for p in proba[k]],
        })
    pred_dist = {c: int((pred == i).sum()) for i, c in enumerate(classes)}
    ref_dist: dict[str, int] = {}
    for c in ref_cls:
        key = c if c is not None else "unmatched"
        ref_dist[key] = ref_dist.get(key, 0) + 1
    ci = {c: i for i, c in enumerate(classes)}
    comp = [(ci[r], int(p)) for r, p in zip(ref_cls, pred) if r in ci]
    agreement = None
    if comp:
        yt, yp = map(np.array, zip(*comp))
        agreement = M.compute(yt, yp, classes, curves=False)
    hdr = loader.load_header(record)
    ch = loader.analysis_channel(record)
    out = {
        "record": record, "mode": mode, "lead": hdr.leads[ch], "fs": fs,
        "duration_s": hdr.duration_s, "classes": classes, "experiment_id": exp_id,
        "partition": partition, "partition_note": _partition_note(partition),
        "n_beats": len(beats_out), "beats": beats_out,
        "predicted_distribution": pred_dist, "reference_distribution": ref_dist,
        "agreement": agreement, "detection": detection,
        "inference_s": round(elapsed, 4), "notice": RESEARCH_NOTICE,
    }
    if log:
        with db.session() as s:
            s.add(db.ClassificationEvent(
                kind="record", record=record, experiment_id=exp_id,
                summary={"mode": mode, "n_beats": len(beats_out), "predicted": pred_dist,
                         "agreement_accuracy": agreement["accuracy"] if agreement else None}))
            s.commit()
    return out


def clear_cache() -> None:
    _load.cache_clear()


# kept for API symmetry with filters (used by docs/tests)
zscore = filters.zscore
torch.set_num_threads(max(torch.get_num_threads(), 1))
