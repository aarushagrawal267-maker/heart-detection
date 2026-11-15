"""Automatic R-peak detection (separate, independently evaluated component).

A compact Pan-Tompkins-style detector written for this project: band-pass (5-18 Hz),
derivative, squaring, moving-window integration and an adaptive threshold with a
refractory period. It is NOT used for the principal classification experiment, which
relies on the expert reference positions.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def detect_r_peaks(x: np.ndarray, fs: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sos = signal.butter(2, [5.0, 18.0], btype="band", fs=fs, output="sos")
    bp = signal.sosfiltfilt(sos, x)
    d = np.gradient(bp)
    win = max(int(0.12 * fs), 1)
    mwi = np.convolve(d * d, np.ones(win) / win, mode="same")
    refractory = int(0.2 * fs)
    cand, _ = signal.find_peaks(mwi, distance=refractory)
    if len(cand) == 0:
        return np.array([], dtype=np.int64)
    # adaptive threshold from running signal / noise peak estimates
    spk = float(np.percentile(mwi[cand], 90))
    npk = float(np.percentile(mwi[cand], 10))
    peaks = []
    for c in cand:
        thr = npk + 0.25 * (spk - npk)
        if mwi[c] > thr:
            peaks.append(c)
            spk = 0.125 * mwi[c] + 0.875 * spk
        else:
            npk = 0.125 * mwi[c] + 0.875 * npk
    # refine each detection to the largest |band-passed| deflection nearby
    half = int(0.08 * fs)
    out = []
    for p in peaks:
        lo, hi = max(p - half, 0), min(p + half, len(x))
        out.append(lo + int(np.argmax(np.abs(bp[lo:hi]))))
    out = np.unique(np.asarray(out, dtype=np.int64))
    if len(out) > 1:  # enforce refractory after refinement
        keep = [out[0]]
        for p in out[1:]:
            if p - keep[-1] >= refractory:
                keep.append(p)
        out = np.asarray(keep, dtype=np.int64)
    return out


def match_peaks(reference: np.ndarray, detected: np.ndarray, fs: int,
                tolerance_s: float = 0.15) -> dict:
    """Greedy one-to-one matching within the AAMI EC57 150 ms tolerance window."""
    tol = int(tolerance_s * fs)
    ref = np.sort(np.asarray(reference))
    det = np.sort(np.asarray(detected))
    i = j = tp = 0
    errors = []
    matched_ref = np.zeros(len(ref), dtype=bool)
    matched_det = np.zeros(len(det), dtype=bool)
    while i < len(ref) and j < len(det):
        diff = det[j] - ref[i]
        if abs(diff) <= tol:
            tp += 1
            errors.append(diff)
            matched_ref[i] = matched_det[j] = True
            i += 1
            j += 1
        elif diff < 0:
            j += 1
        else:
            i += 1
    fn = len(ref) - tp
    fp = len(det) - tp
    return {
        "n_reference": int(len(ref)), "n_detected": int(len(det)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "sensitivity": round(tp / max(tp + fn, 1), 5),
        "positive_predictivity": round(tp / max(tp + fp, 1), 5),
        "mean_abs_error_ms": round(float(np.mean(np.abs(errors)) / fs * 1000), 2) if errors else None,
        "tolerance_ms": tolerance_s * 1000,
        "matched_ref": matched_ref, "matched_det": matched_det,
    }
