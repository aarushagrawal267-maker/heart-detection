"""Waveform, annotation, beat and quality access for individual recordings."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from .. import config
from ..data import loader, subjects
from ..labels import aami
from ..segmentation import beats
from ..signal import quality
from ..splits import split as split_mod
from .common import cache_json, require_dataset, require_processed

router = APIRouter(prefix="/api/records", tags=["records"])


def _check(record: str) -> None:
    require_dataset()
    if record not in config.RECORDS:
        raise HTTPException(404, f"Unknown record {record}")


def _channel(record: str, channel: str | None) -> int:
    h = loader.load_header(record)
    if channel is None:
        return loader.analysis_channel(record)
    if channel in h.leads:
        return h.leads.index(channel)
    if channel.isdigit() and int(channel) < len(h.leads):
        return int(channel)
    raise HTTPException(400, f"Channel {channel} not in {h.leads}")


def minmax_decimate(x: np.ndarray, buckets: int) -> tuple[np.ndarray, np.ndarray]:
    """Min/max envelope decimation that preserves QRS peaks for display."""
    n = len(x)
    if n <= buckets * 2:
        return np.arange(n), x
    edges = np.linspace(0, n, buckets + 1).astype(int)
    idx = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = x[a:b]
        i_min, i_max = a + int(np.argmin(seg)), a + int(np.argmax(seg))
        idx.extend(sorted((i_min, i_max)))
    idx = np.asarray(idx)
    return idx, x[idx]


@router.get("", summary="List recordings")
def list_records() -> list[dict]:
    require_dataset()
    part = split_mod.record_partition()
    out = []
    for r in config.RECORDS:
        h = loader.load_header(r)
        out.append({"record": r, "leads": h.leads, "fs": h.fs, "duration_s": h.duration_s,
                    "subject": subjects.subject_of(r), "partition": part[r],
                    "paced": r in aami.paced_records(),
                    "analysis_lead": h.leads[loader.analysis_channel(r)]})
    return out


@router.get("/{record}", summary="Recording header and metadata")
def record_info(record: str) -> dict:
    _check(record)
    h = loader.load_header(record)
    d = h.to_dict()
    d.update({"subject": subjects.subject_of(record),
              "partition": split_mod.record_partition()[record],
              "paced": record in aami.paced_records(),
              "analysis_lead": h.leads[loader.analysis_channel(record)]})
    return d


@router.get("/{record}/signal", summary="Waveform window (mV), raw or preprocessed")
def signal(record: str, start: float = 0.0, end: float = 10.0, channel: str | None = None,
           processed: bool = False, max_points: int = Query(6000, ge=100, le=200000)) -> dict:
    _check(record)
    h = loader.load_header(record)
    ch = _channel(record, channel)
    a = max(int(start * h.fs), 0)
    b = min(int(end * h.fs), h.n_samples)
    if b <= a:
        raise HTTPException(400, "end must be greater than start")
    if processed:
        require_processed()
        if ch != loader.analysis_channel(record):
            raise HTTPException(400, "Preprocessed signal is stored for the analysis lead only")
        x = np.asarray(beats.load_filtered(record)[a:b], dtype=np.float32)
    else:
        x = loader.load_signal(record)[a:b, ch]
    idx, y = minmax_decimate(x, max_points // 2)
    return {"record": record, "lead": h.leads[ch], "fs": h.fs, "start_sample": a, "end_sample": b,
            "processed": processed, "decimated": len(idx) < len(x),
            "t": [round((a + int(i)) / h.fs, 5) for i in idx],
            "y": [round(float(v), 4) for v in y]}


@router.get("/{record}/overview", summary="Whole-record min/max envelope for navigation")
def overview(record: str, channel: str | None = None, points: int = Query(1500, ge=100, le=10000)) -> dict:
    _check(record)
    h = loader.load_header(record)
    ch = _channel(record, channel)
    x = loader.load_signal(record)[:, ch]
    edges = np.linspace(0, len(x), points + 1).astype(int)
    mins = [float(x[a:b].min()) for a, b in zip(edges[:-1], edges[1:])]
    maxs = [float(x[a:b].max()) for a, b in zip(edges[:-1], edges[1:])]
    return {"record": record, "lead": h.leads[ch], "duration_s": h.duration_s,
            "t": [round(float(e) / h.fs, 3) for e in edges[:-1]],
            "min": [round(v, 3) for v in mins], "max": [round(v, 3) for v in maxs]}


@router.get("/{record}/annotations", summary="Reference annotations in a time window")
def annotations(record: str, start: float = 0.0, end: float | None = None) -> dict:
    _check(record)
    h = loader.load_header(record)
    ann = loader.load_annotations(record)
    end = end if end is not None else h.duration_s
    s = ann["sample"]
    m = (s >= start * h.fs) & (s < end * h.fs)
    items = []
    for i in np.flatnonzero(m):
        sym = ann["symbol"][i]
        items.append({"sample": int(s[i]), "time_s": round(float(s[i]) / h.fs, 4), "symbol": sym,
                      "aami": aami.to_aami(sym), "is_beat": aami.is_beat(sym),
                      "aux": ann["aux"][i] or None})
    # rhythm episodes from '+' annotations (aux field such as "(N", "(AFIB")
    rhythms = []
    cur = None
    for i, sym in enumerate(ann["symbol"]):
        if sym == "+" and ann["aux"][i]:
            if cur:
                cur["end_s"] = round(float(s[i]) / h.fs, 3)
                rhythms.append(cur)
            cur = {"rhythm": ann["aux"][i].lstrip("("), "start_s": round(float(s[i]) / h.fs, 3)}
    if cur:
        cur["end_s"] = round(h.duration_s, 3)
        rhythms.append(cur)
    rhythms = [r for r in rhythms if r["end_s"] > start and r["start_s"] < end]
    return {"record": record, "annotations": items, "rhythms": rhythms}


@router.get("/{record}/beats", summary="Segmented reference beats (metadata) in a window")
def beat_list(record: str, start: float = 0.0, end: float | None = None) -> dict:
    _check(record)
    require_processed()
    _, meta, _ = beats.load()
    m = meta[meta["record"] == record]
    if end is not None:
        m = m[(m["time_s"] >= start) & (m["time_s"] < end)]
    elif start:
        m = m[m["time_s"] >= start]
    cols = ["beat_index", "sample", "time_s", "symbol", "aami"] + beats.RR_FEATURES
    return {"record": record, "n": int(len(m)),
            "beats": [{k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else
                           (int(v) if isinstance(v, (np.integer,)) else v))
                       for k, v in zip(cols, row)} for row in m[cols].itertuples(index=False)]}


@router.get("/{record}/beats/{beat_index}", summary="One beat segment (raw and preprocessed)")
def beat_detail(record: str, beat_index: int, pre: int = config.DEFAULT_WINDOW["pre"],
                post: int = config.DEFAULT_WINDOW["post"]) -> dict:
    _check(record)
    require_processed()
    segs, meta, _ = beats.load()
    rows = np.flatnonzero(((meta["record"] == record) & (meta["beat_index"] == beat_index)).to_numpy())
    if len(rows) == 0:
        raise HTTPException(404, "Beat not found (may be excluded at a recording boundary)")
    i = int(rows[0])
    row = meta.iloc[i]
    try:
        sl = beats.window_slice(config.PIPELINE, pre, post)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    h = loader.load_header(record)
    ch = loader.analysis_channel(record)
    s = int(row["sample"])
    raw = loader.load_signal(record)[s - pre:s + post, ch]
    return {"record": record, "beat_index": beat_index, "sample": s, "time_s": float(row["time_s"]),
            "symbol": row["symbol"], "description": aami.describe(row["symbol"]),
            "aami": row["aami"], "partition": row["partition"], "lead": h.leads[ch],
            "rr": {k: round(float(row[k]), 4) for k in beats.RR_FEATURES},
            "time_ms": [round((k - pre) / h.fs * 1000, 2) for k in range(pre + post)],
            "raw_mv": [round(float(v), 4) for v in raw],
            "filtered_mv": [round(float(v), 4) for v in np.asarray(segs[i, sl])]}


@router.get("/{record}/quality", summary="Windowed signal-quality profile")
def record_quality(record: str, channel: str | None = None, window_s: float = 10.0) -> dict:
    _check(record)
    h = loader.load_header(record)
    ch = _channel(record, channel)

    def build():
        x = loader.load_signal(record)[:, ch]
        adc_min = (-(2 ** (h.adc_res[ch] - 1)) - h.adc_zero[ch]) / h.adc_gain[ch]
        adc_max = ((2 ** (h.adc_res[ch] - 1) - 1) - h.adc_zero[ch]) / h.adc_gain[ch]
        q = quality.record_quality(x, h.fs, window_s, adc_min, adc_max)
        ann = loader.load_annotations(record)
        noise = [{"time_s": round(float(s) / h.fs, 3), "aux": a}
                 for s, sym, a in zip(ann["sample"], ann["symbol"], ann["aux"]) if sym == "~"]
        q["noise_annotations"] = noise
        q["lead"] = h.leads[ch]
        return q
    if beats.is_built():
        return cache_json(f"quality_{record}_{ch}_{window_s}.json", build)
    return build()
