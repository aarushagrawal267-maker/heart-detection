"""Signal-quality inspection (descriptive only: nothing is silently dropped)."""
from __future__ import annotations

import numpy as np
from scipy import signal


def band_power(f: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> float:
    m = (f >= lo) & (f < hi)
    return float(np.trapezoid(pxx[m], f[m])) if m.any() else 0.0


def window_quality(x: np.ndarray, fs: int, adc_min: float | None = None,
                   adc_max: float | None = None) -> dict:
    """Quality indicators for one window of raw signal (mV)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    f, pxx = signal.welch(x - x.mean(), fs=fs, nperseg=min(n, 4 * fs))
    total = band_power(f, pxx, 0.0, fs / 2) + 1e-12
    baseline = band_power(f, pxx, 0.0, 0.5) / total
    powerline = band_power(f, pxx, 58.0, 62.0) / total
    hf = band_power(f, pxx, 45.0, fs / 2) / total
    flat = float(np.mean(np.abs(np.diff(x)) < 1e-6)) if n > 1 else 0.0
    sat = 0.0
    if adc_min is not None and adc_max is not None:
        sat = float(np.mean((x <= adc_min + 1e-9) | (x >= adc_max - 1e-9)))
    score = 1.0
    score -= min(baseline, 0.6) * 0.6
    score -= min(powerline * 4, 0.3)
    score -= min(hf * 2, 0.3)
    score -= min(flat * 1.0, 0.5) if flat > 0.5 else 0.0
    score -= min(sat * 10, 0.5)
    score = float(max(score, 0.0))
    label = "good" if score >= 0.7 else ("fair" if score >= 0.45 else "poor")
    return {"baseline_ratio": round(baseline, 4), "powerline_ratio": round(powerline, 5),
            "hf_ratio": round(hf, 4), "flat_fraction": round(flat, 4),
            "saturation_fraction": round(sat, 5), "amplitude_range_mv": round(float(np.ptp(x)), 3),
            "score": round(score, 3), "label": label}


def record_quality(x: np.ndarray, fs: int, window_s: float = 10.0,
                   adc_min: float | None = None, adc_max: float | None = None) -> dict:
    """Windowed quality profile for a full channel."""
    w = int(window_s * fs)
    windows = []
    for start in range(0, len(x) - w + 1, w):
        q = window_quality(x[start:start + w], fs, adc_min, adc_max)
        q["start_s"] = round(start / fs, 2)
        windows.append(q)
    scores = np.array([q["score"] for q in windows]) if windows else np.array([0.0])
    labels = [q["label"] for q in windows]
    return {
        "window_s": window_s,
        "windows": windows,
        "summary": {
            "mean_score": round(float(scores.mean()), 3),
            "min_score": round(float(scores.min()), 3),
            "good_fraction": round(labels.count("good") / max(len(labels), 1), 3),
            "fair_fraction": round(labels.count("fair") / max(len(labels), 1), 3),
            "poor_fraction": round(labels.count("poor") / max(len(labels), 1), 3),
            "mean_baseline_ratio": round(float(np.mean([q["baseline_ratio"] for q in windows])), 4) if windows else 0,
            "mean_powerline_ratio": round(float(np.mean([q["powerline_ratio"] for q in windows])), 5) if windows else 0,
            "mean_hf_ratio": round(float(np.mean([q["hf_ratio"] for q in windows])), 4) if windows else 0,
        },
    }
