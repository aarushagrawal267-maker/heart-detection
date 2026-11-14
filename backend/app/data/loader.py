"""WFDB loading: signals in physical units, headers and reference annotations."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import wfdb

from .. import config


@dataclass
class RecordHeader:
    record: str
    fs: int
    n_samples: int
    leads: list[str]
    units: list[str]
    adc_gain: list[float]
    adc_res: list[int]
    adc_zero: list[int]
    comments: list[str]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    def lead_index(self, lead: str) -> int | None:
        return self.leads.index(lead) if lead in self.leads else None

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["duration_s"] = self.duration_s
        return d


def _path(record: str) -> str:
    if record not in config.RECORDS:
        raise KeyError(f"Unknown record: {record}")
    return str(config.RAW_DIR / record)


@lru_cache(maxsize=64)
def load_header(record: str) -> RecordHeader:
    h = wfdb.rdheader(_path(record))
    return RecordHeader(
        record=record, fs=int(h.fs), n_samples=int(h.sig_len), leads=list(h.sig_name),
        units=list(h.units), adc_gain=[float(g) for g in h.adc_gain],
        adc_res=[int(r) for r in h.adc_res], adc_zero=[int(z) for z in h.adc_zero],
        comments=list(h.comments or []),
    )


@lru_cache(maxsize=8)
def load_signal(record: str) -> np.ndarray:
    """Return the full recording, shape (n_samples, n_channels), in mV (float32)."""
    rec = wfdb.rdrecord(_path(record), physical=True)
    return np.asarray(rec.p_signal, dtype=np.float32)


@lru_cache(maxsize=64)
def load_annotations(record: str) -> dict:
    """Reference annotations: sample index, symbol and rhythm (aux) strings."""
    ann = wfdb.rdann(_path(record), "atr")
    return {
        "sample": np.asarray(ann.sample, dtype=np.int64),
        "symbol": list(ann.symbol),
        "aux": [a.rstrip("\x00") if a else "" for a in ann.aux_note],
    }


def primary_channel(record: str) -> int | None:
    """Index of the MLII lead, identified by header name (never by position)."""
    return load_header(record).lead_index(config.PRIMARY_LEAD)


def analysis_channel(record: str) -> int:
    """Channel used for analysis: MLII when present, otherwise the first channel.

    Only the paced records 102 and 104 lack MLII; they are excluded from training
    and evaluation, but stay browsable using their first channel (V5).
    """
    ch = primary_channel(record)
    return 0 if ch is None else ch
