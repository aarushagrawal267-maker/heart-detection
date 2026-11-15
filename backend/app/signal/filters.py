"""Conservative ECG filtering.

1. Baseline wander: two cascaded median filters (200 ms then 600 ms) estimate the
   baseline, which is subtracted (de Chazal et al., 2004). Median filtering follows slow
   drift without distorting the QRS complex or the ST segment.
2. Power-line (60 Hz) and high-frequency noise: zero-phase Butterworth low-pass.
   Zero-phase filtering (forward-backward) does not shift R-peak positions.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal


def _odd(n: int) -> int:
    n = max(int(round(n)), 1)
    return n if n % 2 == 1 else n + 1


def estimate_baseline(x: np.ndarray, fs: int, widths_ms=(200, 600)) -> np.ndarray:
    base = np.asarray(x, dtype=np.float64)
    for w in widths_ms:
        base = ndimage.median_filter(base, size=_odd(w * fs / 1000.0), mode="nearest")
    return base


def lowpass(x: np.ndarray, fs: int, cutoff_hz: float = 45.0, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def preprocess(x: np.ndarray, fs: int, cfg: dict) -> np.ndarray:
    """Apply the documented pipeline to one channel; returns float32 mV."""
    y = np.asarray(x, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0)
    y = y - estimate_baseline(y, fs, cfg["baseline_median_ms"])
    y = lowpass(y, fs, cfg["lowpass_hz"], cfg["lowpass_order"])
    return y.astype(np.float32)


def zscore(segments: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-segment standardisation (stateless: nothing is fitted on any partition)."""
    mu = segments.mean(axis=-1, keepdims=True)
    sd = segments.std(axis=-1, keepdims=True)
    return (segments - mu) / (sd + eps)
