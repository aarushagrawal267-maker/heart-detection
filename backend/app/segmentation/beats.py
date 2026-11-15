"""Heartbeat segmentation at expert reference positions and dataset construction.

Every beat annotation becomes one row of metadata; a segment of
`store_pre + store_post` samples (filtered, mV) is stored around the annotation sample.
Models use a configurable sub-window of that stored segment, so different window
lengths can be compared without re-running the pipeline.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import config
from ..data import loader, subjects
from ..labels import aami
from ..signal import filters
from ..splits import split as split_mod

RR_FEATURES = ["rr_prev", "rr_next", "rr_prev_ratio", "rr_next_ratio"]


def rr_features(samples: np.ndarray, fs: int, local_beats: int) -> np.ndarray:
    """RR-interval features from beat positions (seconds and local ratios).

    rr_prev / rr_next: interval to the previous / next beat.
    *_ratio: the interval divided by the median RR of the surrounding +/- `local_beats`
    beats. Only the beat timing of the same recording is used - never labels.
    """
    s = np.asarray(samples, dtype=np.float64)
    n = len(s)
    if n < 2:
        return np.tile(np.array([0.8, 0.8, 1.0, 1.0]), (n, 1))
    rr = np.diff(s) / fs
    rr_prev = np.concatenate([[rr[0]], rr])
    rr_next = np.concatenate([rr, [rr[-1]]])
    local = np.empty(n)
    for i in range(n):
        lo, hi = max(i - local_beats, 0), min(i + local_beats, n - 1)
        local[i] = np.median(rr[lo:hi]) if hi > lo else rr_prev[i]
    local = np.maximum(local, 1e-3)
    return np.stack([rr_prev, rr_next, rr_prev / local, rr_next / local], axis=1)


def extract_segments(x: np.ndarray, samples: np.ndarray, pre: int, post: int):
    """Cut [s-pre, s+post) around each position. Returns (segments, valid_mask)."""
    n = len(x)
    valid = (samples - pre >= 0) & (samples + post <= n)
    idx = samples[valid][:, None] + np.arange(-pre, post)[None, :]
    return x[idx].astype(np.float32), valid


def process_record(record: str, cfg: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """Filter one recording and segment all of its beats."""
    header = loader.load_header(record)
    ch = loader.analysis_channel(record)
    raw = loader.load_signal(record)[:, ch]
    filtered = filters.preprocess(raw, header.fs, cfg)
    ann = loader.load_annotations(record)
    sym = np.array(ann["symbol"])
    is_beat = np.array([aami.is_beat(s) for s in sym])
    beat_samples = ann["sample"][is_beat]
    beat_symbols = sym[is_beat]
    rr = rr_features(beat_samples, header.fs, cfg["rr_local_beats"])
    segs, valid = extract_segments(filtered, beat_samples, cfg["store_pre"], cfg["store_post"])
    meta = pd.DataFrame({
        "record": record,
        "subject": subjects.subject_of(record),
        "beat_index": np.arange(len(beat_samples)),
        "sample": beat_samples,
        "time_s": beat_samples / header.fs,
        "symbol": beat_symbols,
        "aami": [aami.to_aami(s) for s in beat_symbols],
        "lead": header.leads[ch],
    })
    for i, name in enumerate(RR_FEATURES):
        meta[name] = rr[:, i].astype(np.float32)
    n_boundary = int((~valid).sum())
    meta = meta[valid].reset_index(drop=True)
    stats = {
        "record": record, "lead": header.leads[ch], "channel": int(ch),
        "n_annotations": int(len(sym)), "n_beats": int(is_beat.sum()),
        "n_non_beat": int((~is_beat).sum()),
        "n_boundary_excluded": n_boundary,
        "symbol_counts": {str(k): int(v) for k, v in zip(*np.unique(sym, return_counts=True))},
    }
    return filtered, segs, meta, stats


def build(cfg: dict | None = None, progress=None) -> dict:
    """Run the full preprocessing pipeline and write the processed dataset."""
    cfg = cfg or config.PIPELINE
    out = config.processed_dir(cfg)
    (out / "signals").mkdir(parents=True, exist_ok=True)
    partition = split_mod.record_partition()
    folds = None
    all_segs, all_meta, stats = [], [], []
    for i, rec in enumerate(config.RECORDS):
        if progress:
            progress("preprocess", i / len(config.RECORDS))
        filtered, segs, meta, st = process_record(rec, cfg)
        np.save(out / "signals" / f"{rec}.npy", filtered)
        meta["partition"] = partition[rec]
        all_segs.append(segs)
        all_meta.append(meta)
        stats.append(st)
    meta = pd.concat(all_meta, ignore_index=True)
    segs = np.concatenate(all_segs, axis=0)
    folds = split_mod.assign_folds(meta)
    meta["fold"] = meta["record"].map(folds).fillna(-1).astype(int)
    np.save(out / "segments.npy", segs)
    meta.to_parquet(out / "beats.parquet", index=False)
    manifest = {
        "pipeline": cfg, "pipeline_hash": config.pipeline_hash(cfg),
        "dataset_version": config.DATASET_VERSION,
        "n_segments": int(len(meta)), "segment_length": int(segs.shape[1]),
        "records": stats, "folds": folds,
        "n_boundary_excluded": int(sum(s["n_boundary_excluded"] for s in stats)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    clear_cache()
    if progress:
        progress("done", 1.0)
    return manifest


def is_built(cfg: dict | None = None) -> bool:
    out = config.processed_dir(cfg)
    return (out / "manifest.json").exists() and (out / "segments.npy").exists()


@lru_cache(maxsize=2)
def _load(hash_: str):
    out = config.PROCESSED_DIR / hash_
    meta = pd.read_parquet(out / "beats.parquet")
    segs = np.load(out / "segments.npy", mmap_mode="r")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    return segs, meta, manifest


def load(cfg: dict | None = None):
    """Return (segments memmap, metadata DataFrame, manifest)."""
    return _load(config.pipeline_hash(cfg))


def clear_cache() -> None:
    _load.cache_clear()
    load_filtered.cache_clear()


@lru_cache(maxsize=6)
def load_filtered(record: str, hash_: str | None = None) -> np.ndarray:
    hash_ = hash_ or config.pipeline_hash()
    return np.load(config.PROCESSED_DIR / hash_ / "signals" / f"{record}.npy", mmap_mode="r")


def window_slice(cfg: dict, pre: int, post: int) -> slice:
    """Slice of the stored segment corresponding to a model window (pre, post)."""
    if pre > cfg["store_pre"] or post > cfg["store_post"] or pre < 1 or post < 1:
        raise ValueError(
            f"Window ({pre},{post}) exceeds stored segment ({cfg['store_pre']},{cfg['store_post']})")
    return slice(cfg["store_pre"] - pre, cfg["store_pre"] + post)
