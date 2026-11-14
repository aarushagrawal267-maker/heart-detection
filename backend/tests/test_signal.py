"""Filtering, segmentation alignment, RR features and the R-peak detector."""
import numpy as np
import pytest

from app import config
from app.segmentation import beats
from app.signal import detect, filters

FS = 360


def synthetic_ecg(n_beats=40, rr=0.8, seed=0):
    rng = np.random.default_rng(seed)
    n = int((n_beats + 2) * rr * FS)
    t = np.arange(n) / FS
    x = np.zeros(n)
    peaks = (np.arange(1, n_beats + 1) * rr * FS).astype(int)
    for p in peaks:
        x += 1.2 * np.exp(-0.5 * ((np.arange(n) - p) / 4.0) ** 2)          # QRS
        x += 0.25 * np.exp(-0.5 * ((np.arange(n) - p - 0.25 * FS) / 18.0) ** 2)  # T
        x += 0.12 * np.exp(-0.5 * ((np.arange(n) - p + 0.18 * FS) / 10.0) ** 2)  # P
    drift = 0.6 * np.sin(2 * np.pi * 0.2 * t)
    mains = 0.08 * np.sin(2 * np.pi * 60 * t)
    return x, x + drift + mains + rng.normal(0, 0.01, n), peaks


def test_preprocess_removes_drift_and_mains_but_keeps_r_peaks():
    clean, noisy, peaks = synthetic_ecg()
    y = filters.preprocess(noisy, FS, config.PIPELINE)
    for p in peaks[2:-2]:
        local = p - 10 + int(np.argmax(y[p - 10:p + 10]))
        assert abs(local - p) <= 1, "R-peak shifted by filtering"
        assert y[p] == pytest.approx(clean[p] - np.median(clean), rel=0.15)
    # 60 Hz and drift mostly removed: residual vs clean signal is small
    resid = y[400:-400] - (clean[400:-400] - np.median(clean))
    assert np.std(resid) < 0.06


def test_zscore_is_per_segment_and_stateless():
    a = np.random.default_rng(1).normal(3, 2, (5, 256)).astype(np.float32)
    z = filters.zscore(a)
    assert np.allclose(z.mean(1), 0, atol=1e-5)
    assert np.allclose(z.std(1), 1, atol=1e-3)
    assert np.allclose(filters.zscore(a[:1]), z[:1])


def test_segments_are_centred_on_annotation():
    x = np.zeros(5000, dtype=np.float32)
    pos = np.array([50, 700, 1500, 4950])
    x[pos] = np.arange(1, 5)
    segs, valid = beats.extract_segments(x, pos, 90, 166)
    assert valid.tolist() == [False, True, True, False]  # boundary beats excluded
    assert segs.shape == (2, 256)
    assert (segs.argmax(1) == 90).all()
    assert segs[0, 90] == 2 and segs[1, 90] == 3


def test_window_slice_selects_requested_window():
    cfg = config.PIPELINE
    sl = beats.window_slice(cfg, 90, 166)
    assert sl.stop - sl.start == 256
    assert cfg["store_pre"] - sl.start == 90
    with pytest.raises(ValueError):
        beats.window_slice(cfg, cfg["store_pre"] + 1, 10)


def test_rr_features():
    s = np.array([0, 360, 720, 900, 1440, 1800])  # third beat premature
    rr = beats.rr_features(s, FS, local_beats=10)
    assert rr.shape == (6, 4)
    assert rr[3, 0] == pytest.approx(0.5)          # prev RR of premature beat
    assert rr[3, 1] == pytest.approx(1.5)          # compensatory pause
    assert rr[3, 2] < 0.7 < 1.3 < rr[3, 3]


def test_detector_finds_synthetic_beats():
    _, noisy, peaks = synthetic_ecg(n_beats=60)
    y = filters.preprocess(noisy, FS, config.PIPELINE)
    det = detect.detect_r_peaks(y, FS)
    m = detect.match_peaks(peaks, det, FS)
    assert m["sensitivity"] >= 0.98 and m["positive_predictivity"] >= 0.98
    assert m["mean_abs_error_ms"] < 10


def test_match_peaks_counts():
    m = detect.match_peaks(np.array([100, 500, 900]), np.array([110, 905, 2000]), FS)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 1)
