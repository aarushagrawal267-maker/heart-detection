"""Training and evaluation of the project's own models on real MIT-BIH beats.

Model selection uses ONLY validation subjects (a subject-grouped fold of the training
pool). Held-out test subjects are evaluated by a separate, explicitly requested and
logged action (`evaluate_held_out`).
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import torch
from torch import nn

from .. import config
from ..db import models as db
from ..evaluation import metrics as M
from ..models import baselines
from ..models.cnn import ECGNet, count_parameters, describe_architecture
from ..segmentation import beats
from ..splits import split as split_mod
from . import data as D

DEFAULT_CNN_CONFIG = {
    "model": "ECGNet",
    "classes": ["N", "S", "V", "F"],
    "window_pre": config.DEFAULT_WINDOW["pre"],
    "window_post": config.DEFAULT_WINDOW["post"],
    "use_rr": True,
    "channels": [32, 64, 128],
    "dropout": 0.4,
    "epochs": 20,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "loss": "weighted_ce",          # ce | weighted_ce | focal
    "class_weight_power": 0.5,      # weight_c = (n_total / (K n_c)) ** power
    "focal_gamma": 2.0,
    "sampler": "none",              # none | balanced (training partition only)
    "augment": True,
    "early_stopping_patience": 6,
    "val_fold": 2,
    "cross_validate": False,
    "seed": 42,
    "max_train_beats": None,        # optional cap (quick experiments / tests)
}

DEFAULT_BASELINE_CONFIG = {
    "model": "random_forest",
    "classes": ["N", "S", "V", "F"],
    "val_fold": 2,
    "seed": 42,
}


class Cancelled(Exception):
    pass


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None, gamma: float):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        logp = torch.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target[:, None]).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1 - p_t) ** self.gamma) * logp_t
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment() -> dict:
    commit = None
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                                capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        pass
    import sklearn
    return {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "numpy": np.__version__, "sklearn": sklearn.__version__,
        "platform": platform.platform(), "device": str(device()),
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "threads": torch.get_num_threads(), "git_commit": commit,
    }


def class_weights(y: np.ndarray, k: int, power: float) -> np.ndarray:
    counts = np.bincount(y, minlength=k).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = (counts.sum() / (k * counts)) ** power
    return (w / w.mean()).astype(np.float32)


# ---------------------------------------------------------------------------------------
# Progress reporting (written to SQLite so the API / UI can stream it)
# ---------------------------------------------------------------------------------------
class Reporter:
    def __init__(self, exp_id: str | None):
        self.exp_id = exp_id

    def _exp(self, s):
        return s.get(db.Experiment, self.exp_id)

    def update(self, **fields) -> None:
        if not self.exp_id:
            return
        with db.session() as s:
            exp = self._exp(s)
            if exp is None:
                return
            for k, v in fields.items():
                setattr(exp, k, v)
            s.commit()

    def check_cancel(self) -> None:
        if not self.exp_id:
            return
        with db.session() as s:
            exp = self._exp(s)
            if exp is not None and exp.status == "cancelling":
                raise Cancelled()

    def epoch(self, fold: int, epoch: int, row: dict) -> None:
        if not self.exp_id:
            return
        with db.session() as s:
            s.add(db.EpochMetric(experiment_id=self.exp_id, fold=fold, epoch=epoch, **row))
            s.commit()


# ---------------------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------------------
def build_model(cfg: dict) -> ECGNet:
    return ECGNet(n_classes=len(cfg["classes"]), n_rr=len(beats.RR_FEATURES),
                  use_rr=cfg["use_rr"], channels=tuple(cfg["channels"]), dropout=cfg["dropout"])


@torch.no_grad()
def predict_proba(model: ECGNet, x: np.ndarray, rr: np.ndarray, batch: int = 2048) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(dev)
        rb = torch.from_numpy(rr[i:i + batch]).to(dev)
        out.append(torch.softmax(model(xb, rb), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.head[-1].out_features))


def _inputs(segs, meta, idx, cfg, pipeline, scaler):
    w = D.windows(segs, idx, pipeline, cfg["window_pre"], cfg["window_post"])
    rr = meta.iloc[idx][beats.RR_FEATURES].to_numpy(np.float64)
    return D.prepare_inputs(w, rr, scaler)


def _labels(meta, idx, classes):
    ci = {c: i for i, c in enumerate(classes)}
    return np.array(meta.iloc[idx]["aami"].map(ci).to_numpy(np.int64))


def train_cnn(cfg: dict, segs, meta, pipeline, train_idx, val_idx, reporter: Reporter,
              fold: int = -1, progress_span=(0.0, 1.0)):
    """Train one CNN; returns (model, scaler, history, best_epoch, best_val_metrics)."""
    reporter.check_cancel()
    set_seed(cfg["seed"])
    dev = device()
    classes = cfg["classes"]
    k = len(classes)
    rr_train = meta.iloc[train_idx][beats.RR_FEATURES].to_numpy(np.float64)
    scaler = D.RRScaler.fit(rr_train)          # fitted on training subjects only
    ds = D.BeatDataset(segs, meta, train_idx, classes, pipeline, cfg["window_pre"],
                       cfg["window_post"], scaler, augment=cfg["augment"], seed=cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    if cfg["sampler"] == "balanced":
        counts = np.bincount(ds.y, minlength=k).astype(np.float64)
        sw = 1.0 / np.maximum(counts[ds.y], 1)
        sampler = torch.utils.data.WeightedRandomSampler(torch.from_numpy(sw), len(ds),
                                                         replacement=True, generator=gen)
        loader = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler)
    else:
        loader = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                                             generator=gen)
    xv, rv = _inputs(segs, meta, val_idx, cfg, pipeline, scaler)
    yv = np.array(_labels(meta, val_idx, classes))

    model = build_model(cfg).to(dev)
    w = None
    if cfg["loss"] in ("weighted_ce", "focal") and cfg["sampler"] != "balanced":
        w = torch.from_numpy(class_weights(ds.y, k, cfg["class_weight_power"])).to(dev)
    criterion = FocalLoss(w, cfg["focal_gamma"]) if cfg["loss"] == "focal" else nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], epochs=cfg["epochs"],
                                                steps_per_epoch=max(len(loader), 1))
    best_f1, best_state, best_epoch, best_metrics, patience = -1.0, None, 0, None, 0
    history = []
    p0, p1 = progress_span
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        tot, n, tp_pred, tp_true = 0.0, 0, [], []
        for xb, rb, yb in loader:
            reporter.check_cancel() if n % (cfg["batch_size"] * 40) == 0 else None
            xb, rb, yb = xb.to(dev), rb.to(dev), yb.to(dev)
            opt.zero_grad()
            logits = model(xb, rb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)
            n += len(yb)
            tp_pred.append(logits.argmax(1).cpu().numpy())
            tp_true.append(yb.cpu().numpy())
        train_loss = tot / max(n, 1)
        tr = M.compute(np.concatenate(tp_true), np.concatenate(tp_pred), classes, curves=False)
        pv = predict_proba(model, xv, rv)
        with torch.no_grad():
            vl = float(nn.functional.nll_loss(torch.log(torch.from_numpy(pv) + 1e-9),
                                              torch.from_numpy(yv),
                                              weight=None if w is None else w.cpu()))
        vm = M.compute(yv, pv.argmax(1), classes, curves=False)
        row = {"train_loss": round(train_loss, 5), "val_loss": round(vl, 5),
               "train_macro_f1": tr["macro_f1"], "val_macro_f1": vm["macro_f1"],
               "val_accuracy": vm["accuracy"],
               "val_per_class_f1": {c: vm["per_class"][c]["f1"] for c in classes},
               "lr": float(sched.get_last_lr()[0]), "duration_s": round(time.time() - t0, 2)}
        history.append({"epoch": epoch, **row})
        reporter.epoch(fold, epoch, row)
        reporter.update(current_epoch=epoch,
                        progress=p0 + (p1 - p0) * epoch / cfg["epochs"],
                        message=f"fold {fold} epoch {epoch}/{cfg['epochs']}: "
                                f"val macro-F1 {vm['macro_f1']:.3f}" if fold >= 0 else
                        f"epoch {epoch}/{cfg['epochs']}: val macro-F1 {vm['macro_f1']:.3f}")
        reporter.check_cancel()
        if vm["macro_f1"] > best_f1:
            best_f1, best_epoch, best_metrics, patience = vm["macro_f1"], epoch, vm, 0
            best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]:
                break
    model.load_state_dict(best_state)
    return model, scaler, history, best_epoch, best_metrics


def save_checkpoint(exp_id: str, model: ECGNet, scaler: D.RRScaler, cfg: dict, pipeline: dict) -> str:
    out = config.MODELS_DIR / exp_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model.pt"
    torch.save({
        "format": "qrs-atlas-ecgnet-v1",
        "state_dict": model.state_dict(), "config": cfg, "pipeline": pipeline,
        "pipeline_hash": config.pipeline_hash(pipeline), "classes": cfg["classes"],
        "rr_features": beats.RR_FEATURES, "rr_scaler": scaler.to_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, path)
    (out / "config.json").write_text(json.dumps({"config": cfg, "pipeline": pipeline}, indent=2))
    return str(path.relative_to(config.ROOT)) if path.is_relative_to(config.ROOT) else str(path)


def resolve_path(p: str):
    from pathlib import Path
    q = Path(p)
    return q if q.is_absolute() else config.ROOT / q


def load_checkpoint(path: str):
    ck = torch.load(resolve_path(path), map_location="cpu", weights_only=False)
    model = build_model(ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, D.RRScaler.from_dict(ck["rr_scaler"]), ck


def _store_eval(exp_id, partition, mets, per_subj, records, inference=None, boot=None):
    with db.session() as s:
        s.query(db.Evaluation).filter_by(experiment_id=exp_id, partition=partition).delete()
        s.add(db.Evaluation(experiment_id=exp_id, partition=partition, metrics=mets,
                            per_subject=per_subj, records=records, inference=inference or {},
                            bootstrap=boot))
        s.commit()


def evaluate_partition(exp_id: str | None, model, scaler, cfg, pipeline, segs, meta, idx,
                       partition: str, bootstrap: bool = False) -> dict:
    x, rr = _inputs(segs, meta, idx, cfg, pipeline, scaler)
    y = _labels(meta, idx, cfg["classes"])
    model = model.cpu()
    t0 = time.perf_counter()
    proba = predict_proba(model, x, rr)
    dt = time.perf_counter() - t0
    # single-beat latency (batch of one), median of 50 calls
    lat = []
    with torch.no_grad():
        xb, rb = torch.from_numpy(x[:1]), torch.from_numpy(rr[:1])
        for _ in range(50):
            t1 = time.perf_counter()
            model(xb, rb)
            lat.append(time.perf_counter() - t1)
    pred = proba.argmax(1)
    subj = meta.iloc[idx]["subject"].to_numpy()
    mets = M.compute(y, pred, cfg["classes"], proba)
    per_subj = M.per_subject(y, pred, subj, cfg["classes"])
    boot = M.bootstrap_ci(y, pred, subj, cfg["classes"]) if bootstrap else None
    inference = {"n_beats": int(len(idx)), "batch_total_s": round(dt, 4),
                 "per_beat_ms_batched": round(dt / max(len(idx), 1) * 1000, 4),
                 "single_beat_latency_ms_median": round(float(np.median(lat)) * 1000, 3),
                 "device": "cpu"}
    records = sorted(meta.iloc[idx]["record"].unique().tolist())
    if exp_id:
        _store_eval(exp_id, partition, mets, per_subj, records, inference, boot)
    return {"metrics": mets, "per_subject": per_subj, "inference": inference, "bootstrap": boot}


def run_cnn(exp_id: str | None, cfg: dict, reporter: Reporter) -> dict:
    cfg = {**DEFAULT_CNN_CONFIG, **cfg}
    pipeline = config.PIPELINE
    segs, meta, _ = beats.load(pipeline)
    n_folds = split_mod.split_config()["n_folds"]
    folds_all = list(range(n_folds))
    cv = None
    if cfg["cross_validate"]:
        cv_rows = []
        for f in folds_all:
            tr = D.select(meta, cfg["classes"], "train", [g for g in folds_all if g != f])
            va = D.select(meta, cfg["classes"], "train", [f])
            tr = _cap(tr, cfg)
            _, _, hist, be, bm = train_cnn(cfg, segs, meta, pipeline, tr, va, reporter, fold=f,
                                           progress_span=(f / (n_folds + 1), (f + 1) / (n_folds + 1)))
            cv_rows.append({"fold": f, "best_epoch": be, "val_macro_f1": bm["macro_f1"],
                            "per_class_f1": {c: bm["per_class"][c]["f1"] for c in cfg["classes"]},
                            "records": sorted(meta.iloc[va]["record"].unique().tolist())})
        vals = [r["val_macro_f1"] for r in cv_rows]
        cv = {"folds": cv_rows, "mean_macro_f1": round(float(np.mean(vals)), 4),
              "std_macro_f1": round(float(np.std(vals)), 4)}
        reporter.update(cv_results=cv)
        span = (n_folds / (n_folds + 1), 1.0)
    else:
        span = (0.0, 0.95)
    vf = cfg["val_fold"]
    tr = _cap(D.select(meta, cfg["classes"], "train", [g for g in folds_all if g != vf]), cfg)
    va = D.select(meta, cfg["classes"], "train", [vf])
    reporter.update(split=split_record(meta, tr, va))
    model, scaler, hist, best_epoch, best = train_cnn(cfg, segs, meta, pipeline, tr, va, reporter,
                                                       progress_span=span)
    reporter.update(message="evaluating on training and validation subjects")
    train_eval = evaluate_partition(exp_id, model, scaler, cfg, pipeline, segs, meta, tr, "train")
    val_eval = evaluate_partition(exp_id, model, scaler, cfg, pipeline, segs, meta, va, "validation",
                                  bootstrap=True)
    path = save_checkpoint(exp_id or "scratch", model, scaler, cfg, pipeline)
    length = cfg["window_pre"] + cfg["window_post"]
    return {
        "checkpoint_path": path, "n_params": count_parameters(model),
        "architecture": describe_architecture(model, length),
        "summary": {"best_epoch": best_epoch, "epochs_run": len(hist),
                    "val_macro_f1": val_eval["metrics"]["macro_f1"],
                    "val_accuracy": val_eval["metrics"]["accuracy"],
                    "val_per_class_f1": {c: val_eval["metrics"]["per_class"][c]["f1"] for c in cfg["classes"]},
                    "train_macro_f1": train_eval["metrics"]["macro_f1"],
                    "cv_mean_macro_f1": cv["mean_macro_f1"] if cv else None},
    }


def _cap(idx: np.ndarray, cfg: dict) -> np.ndarray:
    cap = cfg.get("max_train_beats")
    if cap and len(idx) > cap:
        rng = np.random.default_rng(cfg["seed"])
        idx = np.sort(rng.choice(idx, size=cap, replace=False))
    return idx


def split_record(meta, tr, va) -> dict:
    sp = split_mod.split_config()
    return {
        "split_name": sp["name"], "split_version": sp["version"],
        "train_records": sorted(meta.iloc[tr]["record"].unique().tolist()),
        "validation_records": sorted(meta.iloc[va]["record"].unique().tolist()),
        "test_records": sp["test"], "excluded_records": sp["excluded"],
        "train_subjects": sorted(meta.iloc[tr]["subject"].unique().tolist()),
        "validation_subjects": sorted(meta.iloc[va]["subject"].unique().tolist()),
        "n_train_beats": int(len(tr)), "n_validation_beats": int(len(va)),
    }


# ---------------------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------------------
def baseline_features(segs, meta, idx, pipeline) -> np.ndarray:
    rr = meta.iloc[idx][beats.RR_FEATURES].to_numpy(np.float64)
    seg = np.asarray(segs[idx], dtype=np.float32)
    morph = baselines.morphology_features(seg, center=pipeline["store_pre"])
    return np.concatenate([rr, morph], axis=1)


def _baseline_X(name, segs, meta, idx, pipeline):
    if name == "logreg_rr" or name == "majority":
        return meta.iloc[idx][beats.RR_FEATURES].to_numpy(np.float64)
    return baseline_features(segs, meta, idx, pipeline)


def run_baseline(exp_id: str | None, cfg: dict, reporter: Reporter) -> dict:
    cfg = {**DEFAULT_BASELINE_CONFIG, **cfg}
    pipeline = config.PIPELINE
    segs, meta, _ = beats.load(pipeline)
    folds_all = list(range(split_mod.split_config()["n_folds"]))
    vf = cfg["val_fold"]
    tr = D.select(meta, cfg["classes"], "train", [g for g in folds_all if g != vf])
    va = D.select(meta, cfg["classes"], "train", [vf])
    reporter.update(split=split_record(meta, tr, va), message="fitting", progress=0.2)
    reporter.check_cancel()
    clf = baselines.build(cfg["model"], cfg["seed"])
    clf.fit(_baseline_X(cfg["model"], segs, meta, tr, pipeline), _labels(meta, tr, cfg["classes"]))
    reporter.update(message="evaluating", progress=0.7)
    out = {}
    for part, idx, boot in (("train", tr, False), ("validation", va, True)):
        out[part] = evaluate_baseline(exp_id, clf, cfg, pipeline, segs, meta, idx, part, boot)
    d = config.MODELS_DIR / (exp_id or "scratch")
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "config": cfg, "pipeline": pipeline}, d / "model.joblib")
    (d / "config.json").write_text(json.dumps({"config": cfg, "pipeline": pipeline}, indent=2))
    p = d / "model.joblib"
    vm = out["validation"]["metrics"]
    return {"checkpoint_path": str(p.relative_to(config.ROOT)) if p.is_relative_to(config.ROOT) else str(p),
            "n_params": None, "architecture": [{"layer": baselines.BASELINES[cfg["model"]]}],
            "summary": {"val_macro_f1": vm["macro_f1"], "val_accuracy": vm["accuracy"],
                        "val_per_class_f1": {c: vm["per_class"][c]["f1"] for c in cfg["classes"]},
                        "train_macro_f1": out["train"]["metrics"]["macro_f1"]}}


def evaluate_baseline(exp_id, clf, cfg, pipeline, segs, meta, idx, partition, bootstrap=False):
    X = _baseline_X(cfg["model"], segs, meta, idx, pipeline)
    y = _labels(meta, idx, cfg["classes"])
    t0 = time.perf_counter()
    proba_raw = clf.predict_proba(X)
    dt = time.perf_counter() - t0
    proba = np.zeros((len(idx), len(cfg["classes"])))
    proba[:, clf.classes_] = proba_raw
    pred = proba.argmax(1)
    subj = meta.iloc[idx]["subject"].to_numpy()
    mets = M.compute(y, pred, cfg["classes"], proba)
    per_subj = M.per_subject(y, pred, subj, cfg["classes"])
    boot = M.bootstrap_ci(y, pred, subj, cfg["classes"]) if bootstrap else None
    inference = {"n_beats": int(len(idx)), "batch_total_s": round(dt, 4),
                 "per_beat_ms_batched": round(dt / max(len(idx), 1) * 1000, 4), "device": "cpu"}
    records = sorted(meta.iloc[idx]["record"].unique().tolist())
    if exp_id:
        _store_eval(exp_id, partition, mets, per_subj, records, inference, boot)
    return {"metrics": mets, "per_subject": per_subj, "inference": inference, "bootstrap": boot}


# ---------------------------------------------------------------------------------------
# Held-out test evaluation (explicit, logged)
# ---------------------------------------------------------------------------------------
def evaluate_held_out(exp_id: str, purpose: str = "") -> dict:
    with db.session() as s:
        exp = s.get(db.Experiment, exp_id)
        if exp is None or exp.status != "completed":
            raise ValueError("Experiment not found or not completed")
        s.add(db.TestAccess(experiment_id=exp_id, purpose=purpose or "held-out evaluation"))
        s.commit()
        kind, path, cfg = exp.kind, exp.checkpoint_path, exp.config
    pipeline = config.PIPELINE
    segs, meta, _ = beats.load(pipeline)
    from .registry import export_experiment
    if kind == "cnn":
        model, scaler, ck = load_checkpoint(path)
        cfg = ck["config"]
        idx = D.select(meta, cfg["classes"], "test")
        res = evaluate_partition(exp_id, model, scaler, cfg, pipeline, segs, meta, idx, "test",
                                 bootstrap=True)
    else:
        bundle = joblib.load(resolve_path(path))
        cfg = bundle["config"]
        idx = D.select(meta, cfg["classes"], "test")
        res = evaluate_baseline(exp_id, bundle["model"], cfg, pipeline, segs, meta, idx, "test",
                                bootstrap=True)
    export_experiment(exp_id)
    return res


# ---------------------------------------------------------------------------------------
# Entry point used by the worker process
# ---------------------------------------------------------------------------------------
def run_experiment(exp_id: str) -> None:
    torch.set_num_threads(max(os.cpu_count() or 1, 1))
    reporter = Reporter(exp_id)
    t0 = time.time()
    # Claim the job atomically: a cancel requested before the worker started must win.
    with db.session() as s:
        exp = s.get(db.Experiment, exp_id)
        kind, cfg = exp.kind, dict(exp.config)
        if exp.status in ("cancelling", "cancelled"):
            exp.status = "cancelled"
            exp.message = "cancelled before start"
            exp.finished_at = datetime.now(timezone.utc)
            s.commit()
            return
        exp.status = "running"
        s.commit()
    reporter.update(started_at=datetime.now(timezone.utc), pid=os.getpid(),
                    environment=environment(), pipeline=config.PIPELINE,
                    pipeline_hash=config.pipeline_hash(),
                    total_epochs=cfg.get("epochs", 0) if kind == "cnn" else 0,
                    message="loading data")
    try:
        if not beats.is_built():
            reporter.update(message="building processed dataset")
            beats.build()
        res = run_cnn(exp_id, cfg, reporter) if kind == "cnn" else run_baseline(exp_id, cfg, reporter)
        reporter.update(status="completed", finished_at=datetime.now(timezone.utc),
                        duration_s=round(time.time() - t0, 1), progress=1.0,
                        message="completed", **res)
        from .registry import export_experiment
        export_experiment(exp_id)
    except Cancelled:
        reporter.update(status="cancelled", finished_at=datetime.now(timezone.utc),
                        duration_s=round(time.time() - t0, 1), message="cancelled by user")
    except Exception as exc:  # noqa: BLE001 - recorded and surfaced in the UI
        import traceback
        reporter.update(status="failed", finished_at=datetime.now(timezone.utc),
                        duration_s=round(time.time() - t0, 1), error=traceback.format_exc(),
                        message=f"failed: {exc}")
        raise
