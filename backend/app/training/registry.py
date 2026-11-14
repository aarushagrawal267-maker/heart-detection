"""Portable experiment records: models/<id>/experiment.json <-> SQLite.

The SQLite database is local state (not committed). Every finished experiment is also
exported as JSON next to its checkpoint, so a fresh installation can import the trained
models, their full metric history and the held-out test access log on startup.
"""
from __future__ import annotations

import json
from datetime import datetime

from .. import config
from ..db import models as db


def _dt(v):
    return datetime.fromisoformat(v) if v else None


def export_experiment(exp_id: str) -> None:
    with db.session() as s:
        e = s.get(db.Experiment, exp_id)
        if e is None or e.status != "completed":
            return
        rec = {
            "format": "qrs-atlas-experiment-v1",
            "experiment": {
                "id": e.id, "name": e.name, "kind": e.kind, "model_name": e.model_name,
                "status": e.status, "config": e.config, "pipeline": e.pipeline,
                "pipeline_hash": e.pipeline_hash, "dataset_version": e.dataset_version,
                "split": e.split, "architecture": e.architecture, "n_params": e.n_params,
                "environment": e.environment, "checkpoint_path": e.checkpoint_path,
                "created_at": e.created_at.isoformat() if e.created_at else None,
                "started_at": e.started_at.isoformat() if e.started_at else None,
                "finished_at": e.finished_at.isoformat() if e.finished_at else None,
                "duration_s": e.duration_s, "current_epoch": e.current_epoch,
                "total_epochs": e.total_epochs, "summary": e.summary, "cv_results": e.cv_results,
                "notes": e.notes,
            },
            "epochs": [{"fold": m.fold, "epoch": m.epoch, "train_loss": m.train_loss,
                        "val_loss": m.val_loss, "train_macro_f1": m.train_macro_f1,
                        "val_macro_f1": m.val_macro_f1, "val_accuracy": m.val_accuracy,
                        "val_per_class_f1": m.val_per_class_f1, "lr": m.lr,
                        "duration_s": m.duration_s} for m in e.epochs],
            "evaluations": [{"partition": ev.partition, "metrics": ev.metrics,
                             "per_subject": ev.per_subject, "bootstrap": ev.bootstrap,
                             "inference": ev.inference, "records": ev.records,
                             "created_at": ev.created_at.isoformat() if ev.created_at else None}
                            for ev in e.evaluations],
            "test_access": [{"timestamp": t.timestamp.isoformat(), "purpose": t.purpose}
                            for t in s.query(db.TestAccess).filter_by(experiment_id=exp_id).all()],
        }
    out = config.MODELS_DIR / exp_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")


def import_all() -> int:
    """Import experiment.json records that are not yet in the database."""
    if not config.MODELS_DIR.exists():
        return 0
    n = 0
    with db.session() as s:
        for path in sorted(config.MODELS_DIR.glob("*/experiment.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            x = rec.get("experiment", {})
            if not x.get("id"):
                continue
            existing = s.get(db.Experiment, x["id"])
            if existing is not None:
                # refresh evaluations added on another machine (e.g. a held-out test run)
                have = {ev.partition for ev in existing.evaluations}
                for ev in rec.get("evaluations", []):
                    if ev["partition"] not in have:
                        s.add(db.Evaluation(experiment_id=x["id"], **{**ev, "created_at": _dt(ev["created_at"])}))
                continue
            exp = db.Experiment(**{**x, "created_at": _dt(x["created_at"]), "started_at": _dt(x["started_at"]),
                                   "finished_at": _dt(x["finished_at"]), "progress": 1.0,
                                   "message": "imported from models/"})
            s.add(exp)
            for m in rec.get("epochs", []):
                s.add(db.EpochMetric(experiment_id=x["id"], **m))
            for ev in rec.get("evaluations", []):
                s.add(db.Evaluation(experiment_id=x["id"], **{**ev, "created_at": _dt(ev["created_at"])}))
            for t in rec.get("test_access", []):
                s.add(db.TestAccess(experiment_id=x["id"], timestamp=_dt(t["timestamp"]), purpose=t["purpose"]))
            n += 1
        s.commit()
    sel = config.MODELS_DIR / "selection.json"
    if sel.exists() and not db.get_setting("active_experiment_id"):
        try:
            chosen = json.loads(sel.read_text(encoding="utf-8"))["selected"]["id"]
            with db.session() as s:
                if s.get(db.Experiment, chosen) is not None:
                    db.set_setting("active_experiment_id", chosen)
        except (KeyError, json.JSONDecodeError):
            pass
    return n
