"""Comparison models: majority class, RR-only logistic regression, random forest.

Features are hand-crafted from the same segments and RR features the CNN sees, so the
comparison isolates the effect of learned convolutional features.
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def morphology_features(segs: np.ndarray, center: int, fs: int = 360) -> np.ndarray:
    """Simple morphology descriptors from filtered (mV) segments.

    - 16 samples of the QRS region (-100 ms..+150 ms) at reduced rate
    - 8 samples of the T-wave region (+150..+450 ms)
    - QRS amplitude stats and an approximate QRS width (samples above 30% of peak energy)
    """
    s = np.asarray(segs, dtype=np.float32)
    q0, q1 = center - int(0.10 * fs), center + int(0.15 * fs)
    t0, t1 = center + int(0.15 * fs), min(center + int(0.45 * fs), s.shape[1])
    qrs = s[:, q0:q1]
    tw = s[:, t0:t1]
    qrs_ds = qrs[:, np.linspace(0, qrs.shape[1] - 1, 16).astype(int)]
    tw_ds = tw[:, np.linspace(0, tw.shape[1] - 1, 8).astype(int)]
    energy = np.abs(np.diff(qrs, axis=1))
    thr = 0.3 * energy.max(axis=1, keepdims=True)
    width = (energy > thr).sum(axis=1, keepdims=True) / fs
    stats = np.stack([qrs.max(1), qrs.min(1), np.ptp(qrs, 1), tw.max(1), tw.min(1),
                      s.std(1)], axis=1)
    return np.concatenate([qrs_ds, tw_ds, width, stats], axis=1)


def build(name: str, seed: int = 0):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg_rr":
        return make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed))
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed)
    raise ValueError(f"Unknown baseline: {name}")


BASELINES = {
    "majority": "Majority class (always predicts N) - shows the accuracy trap",
    "logreg_rr": "Logistic regression on RR-interval features only",
    "random_forest": "Random forest on RR features + hand-crafted morphology features",
}
