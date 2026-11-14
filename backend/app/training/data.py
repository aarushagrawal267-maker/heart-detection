"""Training data assembly: partition selection, window extraction, scaling, augmentation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..segmentation import beats
from ..signal import filters


@dataclass
class RRScaler:
    """Standardises RR features; fitted on TRAINING rows only."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, rr: np.ndarray) -> "RRScaler":
        return cls(rr.mean(axis=0), rr.std(axis=0) + 1e-6)

    def transform(self, rr: np.ndarray) -> np.ndarray:
        return ((rr - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "RRScaler":
        return cls(np.asarray(d["mean"], dtype=np.float64), np.asarray(d["std"], dtype=np.float64))


def select(meta, classes: list[str], partition: str, folds: list[int] | None = None) -> np.ndarray:
    """Row indices for a partition (and optionally CV folds) restricted to `classes`."""
    m = (meta["partition"] == partition) & meta["aami"].isin(classes)
    if folds is not None:
        m &= meta["fold"].isin(folds)
    return np.flatnonzero(m.to_numpy())


def windows(segs: np.ndarray, idx: np.ndarray, pipeline: dict, pre: int, post: int,
            shift: int = 0) -> np.ndarray:
    sl = beats.window_slice(pipeline, pre, post)
    start, stop = sl.start + shift, sl.stop + shift
    return np.asarray(segs[idx, start:stop], dtype=np.float32)


def prepare_inputs(segs_window: np.ndarray, rr: np.ndarray, scaler: RRScaler):
    x = filters.zscore(segs_window).astype(np.float32)[:, None, :]
    return x, scaler.transform(rr)


class BeatDataset(torch.utils.data.Dataset):
    """Holds the training pool of one partition in memory (about 50 MB for ~45k beats)."""

    def __init__(self, segs, meta, idx, classes, pipeline, pre, post, scaler: RRScaler,
                 augment: bool = False, max_shift: int = 5, seed: int = 0):
        self.idx = idx
        self.pre, self.post = pre, post
        self.augment = augment
        lo = -(pipeline["store_pre"] - pre)
        hi = pipeline["store_post"] - post
        self.shift_range = (max(lo, -max_shift), min(hi, max_shift))
        # keep the full stored segment so random shifts use real neighbouring samples
        self.full = np.asarray(segs[idx], dtype=np.float32)
        self.base = pipeline["store_pre"] - pre
        self.length = pre + post
        rr = meta.iloc[idx][beats.RR_FEATURES].to_numpy(np.float64)
        self.rr = scaler.transform(rr)
        cls_index = {c: i for i, c in enumerate(classes)}
        self.y = meta.iloc[idx]["aami"].map(cls_index).to_numpy(np.int64)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        shift = 0
        if self.augment and self.shift_range[1] > self.shift_range[0]:
            shift = int(self.rng.integers(self.shift_range[0], self.shift_range[1] + 1))
        s = self.base + shift
        x = self.full[i, s:s + self.length]
        x = (x - x.mean()) / (x.std() + 1e-6)
        if self.augment:
            x = x * self.rng.uniform(0.85, 1.15) + self.rng.normal(0, 0.05)
            x = x + self.rng.normal(0, 0.03, size=x.shape)
        return (torch.from_numpy(x.astype(np.float32))[None, :],
                torch.from_numpy(self.rr[i]), torch.tensor(self.y[i]))
