"""Overview page aggregate and system health."""
from __future__ import annotations

from fastapi import APIRouter

from .. import config
from ..data import acquire, subjects
from ..db import models as db
from ..inference import service
from ..segmentation import beats
from ..training import jobs, trainer
from .common import iso

router = APIRouter(prefix="/api", tags=["overview"])


@router.get("/health", summary="Liveness check")
def health() -> dict:
    return {"status": "ok", "device": str(trainer.device())}


@router.get("/overview", summary="Aggregated figures for the overview page (all from real data)")
def overview() -> dict:
    present = acquire.is_present()
    built = present and beats.is_built()
    out: dict = {
        "dataset": {"present": present, "processed": built, "name": config.DATASET_NAME,
                    "version": config.DATASET_VERSION,
                    "n_records": len(config.RECORDS) if present else 0,
                    "n_subjects": subjects.n_subjects() if present else 0},
        "notice": service.RESEARCH_NOTICE,
    }
    if built:
        _, meta, man = beats.load()
        out["dataset"]["n_segments"] = man["n_segments"]
        out["dataset"]["class_counts"] = {k: int(v) for k, v in meta["aami"].value_counts().items()}
        out["dataset"]["partition_counts"] = {k: int(v) for k, v in meta["partition"].value_counts().items()}
    jobs.reap()
    with db.session() as s:
        exps = s.query(db.Experiment).order_by(db.Experiment.created_at.desc()).all()
        out["experiments"] = {"total": len(exps),
                              "completed": sum(e.status == "completed" for e in exps),
                              "running": sum(e.status in ("running", "cancelling") for e in exps)}
        latest = exps[0] if exps else None
        out["latest_experiment"] = None if latest is None else {
            "id": latest.id, "name": latest.name, "status": latest.status,
            "created_at": iso(latest.created_at), "summary": latest.summary, "progress": latest.progress}
        active = service.active_experiment_id()
        out["active_model"] = None
        if active:
            e = s.get(db.Experiment, active)
            evals = {ev.partition: ev for ev in e.evaluations}
            test = evals.get("test")
            out["active_model"] = {
                "id": e.id, "name": e.name, "summary": e.summary, "n_params": e.n_params,
                "finished_at": iso(e.finished_at), "classes": e.config.get("classes"),
                "validation": evals["validation"].metrics if "validation" in evals else None,
                "test": None if test is None else {
                    k: test.metrics.get(k) for k in ("accuracy", "macro_f1", "weighted_f1", "per_class", "n")},
            }
        events = s.query(db.ClassificationEvent).order_by(db.ClassificationEvent.timestamp.desc()).limit(12).all()
        out["recent_activity"] = [{"timestamp": iso(ev.timestamp), "kind": ev.kind, "record": ev.record,
                                   "beat_index": ev.beat_index, "predicted": ev.predicted,
                                   "reference": ev.reference, "confidence": ev.confidence,
                                   "summary": ev.summary} for ev in events]
    return out
