"""Evaluation metrics for imbalanced multi-class heartbeat classification."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (auc, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)


def _curve_points(x: np.ndarray, y: np.ndarray, max_points: int = 200) -> list[list[float]]:
    if len(x) > max_points:
        idx = np.unique(np.linspace(0, len(x) - 1, max_points).astype(int))
        x, y = x[idx], y[idx]
    return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(x, y)]


def compute(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str],
            proba: np.ndarray | None = None, curves: bool = True) -> dict:
    """Full metric set. y_true / y_pred are integer class indices into `classes`."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    k = len(classes)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))
    total = cm.sum()
    per_class = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        se = tp / (tp + fn) if tp + fn else None
        ppv = tp / (tp + fp) if tp + fp else None
        sp = tn / (tn + fp) if tn + fp else None
        f1 = (2 * se * ppv / (se + ppv)) if se is not None and ppv and (se + ppv) > 0 else (0.0 if tp + fn else None)
        per_class[c] = {
            "support": int(tp + fn), "predicted": int(tp + fp),
            "precision": None if ppv is None else round(float(ppv), 5),
            "recall": None if se is None else round(float(se), 5),
            "sensitivity": None if se is None else round(float(se), 5),
            "specificity": None if sp is None else round(float(sp), 5),
            "fpr": None if sp is None else round(float(1 - sp), 5),
            "f1": None if f1 is None else round(float(f1), 5),
        }
    present = [i for i in range(k) if cm[i].sum() > 0]
    out = {
        "classes": classes,
        "n": int(total),
        "accuracy": round(float(np.trace(cm) / total), 5) if total else None,
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=present, average="macro",
                                         zero_division=0)), 5) if present else None,
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=present, average="weighted",
                                            zero_division=0)), 5) if present else None,
        "macro_recall": round(float(np.mean([per_class[classes[i]]["recall"] for i in present])), 5) if present else None,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
    if proba is not None:
        roc, pr = {}, {}
        for i, c in enumerate(classes):
            yb = (y_true == i).astype(int)
            if 0 < yb.sum() < len(yb):
                entry_roc = {"auc": round(float(roc_auc_score(yb, proba[:, i])), 5)}
                entry_pr = {"ap": round(float(average_precision_score(yb, proba[:, i])), 5),
                            "prevalence": round(float(yb.mean()), 5)}
                if curves:
                    fpr, tpr, _ = roc_curve(yb, proba[:, i])
                    prec, rec, _ = precision_recall_curve(yb, proba[:, i])
                    entry_roc["points"] = _curve_points(fpr, tpr)
                    entry_pr["points"] = _curve_points(rec[::-1], prec[::-1])
                    entry_pr["auc"] = round(float(auc(rec, prec)), 5)
                roc[c], pr[c] = entry_roc, entry_pr
        out["roc"] = roc
        out["pr"] = pr
        aucs = [v["auc"] for v in roc.values()]
        out["macro_roc_auc"] = round(float(np.mean(aucs)), 5) if aucs else None
    return out


def per_subject(y_true, y_pred, subjects, classes) -> list[dict]:
    rows = []
    subjects = np.asarray(subjects)
    for s in sorted(set(subjects)):
        m = subjects == s
        cm = confusion_matrix(np.asarray(y_true)[m], np.asarray(y_pred)[m], labels=list(range(len(classes))))
        row = {"subject": str(s), "n": int(m.sum()), "accuracy": round(float(np.trace(cm) / m.sum()), 4)}
        for i, c in enumerate(classes):
            sup = cm[i].sum()
            row[f"{c}_support"] = int(sup)
            row[f"{c}_recall"] = round(float(cm[i, i] / sup), 4) if sup else None
        rows.append(row)
    return rows


def bootstrap_ci(y_true, y_pred, subjects, classes, n_boot: int = 500, seed: int = 0) -> dict:
    """Cluster bootstrap over SUBJECTS (patients are the unit of generalisation)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred, subjects = map(np.asarray, (y_true, y_pred, subjects))
    uniq = np.unique(subjects)
    idx_by = {s: np.flatnonzero(subjects == s) for s in uniq}
    stats = {"macro_f1": [], "accuracy": []}
    for c in classes:
        stats[f"{c}_f1"] = []
        stats[f"{c}_recall"] = []
    k = len(classes)
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[s] for s in pick])
        yt, yp = y_true[idx], y_pred[idx]
        cm = confusion_matrix(yt, yp, labels=list(range(k)))
        present = [i for i in range(k) if cm[i].sum() > 0]
        stats["accuracy"].append(np.trace(cm) / cm.sum())
        stats["macro_f1"].append(f1_score(yt, yp, labels=present, average="macro", zero_division=0))
        for i, c in enumerate(classes):
            sup, pred = cm[i].sum(), cm[:, i].sum()
            if sup:
                r = cm[i, i] / sup
                p = cm[i, i] / pred if pred else 0.0
                stats[f"{c}_recall"].append(r)
                stats[f"{c}_f1"].append(2 * p * r / (p + r) if p + r else 0.0)
    out = {}
    for name, vals in stats.items():
        if vals:
            out[name] = [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]
    out["n_boot"] = n_boot
    out["unit"] = "subject"
    return out
