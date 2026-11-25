"""Experiments, training jobs, evaluation, inference and explainability endpoints."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .. import config
from ..db import models as db
from ..explain.attribution import METHODS
from ..inference import service
from ..models import baselines
from ..models.cnn import ECGNet, count_parameters, describe_architecture
from ..training import jobs, trainer
from .common import iso, require_processed

router = APIRouter(prefix="/api", tags=["models"])


def exp_dict(e: db.Experiment, detail: bool = False) -> dict:
    d = {
        "id": e.id, "name": e.name, "kind": e.kind, "model_name": e.model_name,
        "status": e.status, "created_at": iso(e.created_at), "started_at": iso(e.started_at),
        "finished_at": iso(e.finished_at), "duration_s": e.duration_s,
        "current_epoch": e.current_epoch, "total_epochs": e.total_epochs,
        "progress": e.progress, "message": e.message, "summary": e.summary or {},
        "config": e.config, "n_params": e.n_params, "checkpoint_path": e.checkpoint_path,
        "dataset_version": e.dataset_version, "pipeline_hash": e.pipeline_hash,
        "has_test_evaluation": any(ev.partition == "test" for ev in e.evaluations),
    }
    if detail:
        d.update({"pipeline": e.pipeline, "split": e.split, "architecture": e.architecture,
                  "environment": e.environment, "error": e.error, "notes": e.notes,
                  "cv_results": e.cv_results})
    return d


def _get(s, exp_id: str) -> db.Experiment:
    e = s.get(db.Experiment, exp_id)
    if e is None:
        raise HTTPException(404, "Experiment not found")
    return e


# ------------------------------- training ----------------------------------------------
class ExperimentCreate(BaseModel):
    name: str = Field("", max_length=200)
    kind: Literal["cnn", "baseline"] = "cnn"
    config: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    start: bool = True


@router.get("/training/defaults", summary="Default hyperparameters and options")
def training_defaults() -> dict:
    cfg = trainer.DEFAULT_CNN_CONFIG
    m = ECGNet(n_classes=len(cfg["classes"]))
    return {
        "cnn": cfg, "baseline": trainer.DEFAULT_BASELINE_CONFIG,
        "baselines": baselines.BASELINES,
        "options": {"loss": ["ce", "weighted_ce", "focal"], "sampler": ["none", "balanced"],
                    "classes": [["N", "S", "V", "F"], ["N", "S", "V", "F", "Q"]],
                    "val_fold": list(range(5)),
                    "window_limits": {"max_pre": config.PIPELINE["store_pre"],
                                      "max_post": config.PIPELINE["store_post"]}},
        "architecture": describe_architecture(m, cfg["window_pre"] + cfg["window_post"]),
        "n_params": count_parameters(m),
        "device": str(trainer.device()),
    }


@router.post("/experiments", summary="Create (and optionally start) a training experiment")
def create_experiment(body: ExperimentCreate) -> dict:
    require_processed()
    if os.environ.get("QRS_ALLOW_TRAINING", "true").lower() in ("0", "false", "no"):
        raise HTTPException(403, "Training is disabled on this deployment (QRS_ALLOW_TRAINING=false).")
    cfg = body.config
    if body.kind == "cnn":
        if "classes" in cfg and not set(cfg["classes"]) <= {"N", "S", "V", "F", "Q"}:
            raise HTTPException(400, "classes must be a subset of N,S,V,F,Q")
        if "epochs" in cfg and not (1 <= int(cfg["epochs"]) <= 200):
            raise HTTPException(400, "epochs must be 1..200")
        pre = cfg.get("window_pre", trainer.DEFAULT_CNN_CONFIG["window_pre"])
        post = cfg.get("window_post", trainer.DEFAULT_CNN_CONFIG["window_post"])
        if not (16 <= pre <= config.PIPELINE["store_pre"] and 16 <= post <= config.PIPELINE["store_post"]):
            raise HTTPException(400, "window outside the stored segment")
    elif cfg.get("model", "random_forest") not in baselines.BASELINES:
        raise HTTPException(400, f"model must be one of {list(baselines.BASELINES)}")
    if body.start and jobs.running() is not None:
        raise HTTPException(409, "Another training job is running. Wait or cancel it first.")
    try:
        exp = jobs.create(body.name, body.kind, cfg, body.notes)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if body.start:
        jobs.start(exp.id)
    with db.session() as s:
        return exp_dict(_get(s, exp.id))


@router.post("/experiments/{exp_id}/start", summary="Start a queued experiment")
def start_experiment(exp_id: str) -> dict:
    with db.session() as s:
        e = _get(s, exp_id)
        if e.status != "queued":
            raise HTTPException(409, f"Experiment is {e.status}")
    try:
        jobs.start(exp_id)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"started": exp_id}


@router.post("/experiments/{exp_id}/cancel", summary="Request cancellation of a running job")
def cancel_experiment(exp_id: str) -> dict:
    try:
        jobs.cancel(exp_id)
    except KeyError as exc:
        raise HTTPException(404, "Experiment not found") from exc
    return {"cancelling": exp_id}


@router.get("/training/current", summary="The currently running job, if any")
def current_job() -> dict | None:
    e = jobs.running()
    if e is None:
        return None
    with db.session() as s:
        return exp_dict(_get(s, e.id))


# ------------------------------- experiments -------------------------------------------
@router.get("/experiments", summary="Experiment history")
def list_experiments(kind: str | None = None, status: str | None = None) -> list[dict]:
    jobs.reap()
    with db.session() as s:
        q = s.query(db.Experiment)
        if kind:
            q = q.filter_by(kind=kind)
        if status:
            q = q.filter_by(status=status)
        active = service.active_experiment_id()
        out = []
        for e in q.order_by(db.Experiment.created_at.desc()).all():
            d = exp_dict(e)
            d["active"] = e.id == active
            out.append(d)
        return out


@router.get("/experiments/compare", summary="Side-by-side comparison of experiments")
def compare(ids: str = Query(..., description="Comma-separated experiment ids")) -> list[dict]:
    out = []
    with db.session() as s:
        for i in ids.split(","):
            e = _get(s, i.strip())
            d = exp_dict(e, detail=True)
            d["evaluations"] = {ev.partition: {k: ev.metrics.get(k) for k in
                                               ("accuracy", "macro_f1", "weighted_f1", "macro_roc_auc", "per_class")}
                                for ev in e.evaluations}
            out.append(d)
    return out


@router.get("/experiments/{exp_id}", summary="Experiment detail")
def get_experiment(exp_id: str) -> dict:
    jobs.reap()
    with db.session() as s:
        e = _get(s, exp_id)
        d = exp_dict(e, detail=True)
        d["active"] = e.id == service.active_experiment_id()
        return d


@router.get("/experiments/{exp_id}/epochs", summary="Per-epoch training/validation metrics")
def epochs(exp_id: str) -> list[dict]:
    with db.session() as s:
        e = _get(s, exp_id)
        return [{"fold": m.fold, "epoch": m.epoch, "train_loss": m.train_loss, "val_loss": m.val_loss,
                 "train_macro_f1": m.train_macro_f1, "val_macro_f1": m.val_macro_f1,
                 "val_accuracy": m.val_accuracy, "val_per_class_f1": m.val_per_class_f1,
                 "lr": m.lr, "duration_s": m.duration_s} for m in e.epochs]


@router.get("/experiments/{exp_id}/stream", summary="Server-sent events with live training progress")
async def stream(exp_id: str):
    async def gen():
        last_epochs = -1
        while True:
            jobs.reap()
            with db.session() as s:
                e = s.get(db.Experiment, exp_id)
                if e is None:
                    yield "event: error\ndata: {}\n\n"
                    return
                d = exp_dict(e)
                n = len(e.epochs)
                payload = {"experiment": d}
                if n != last_epochs:
                    payload["epochs"] = [{"fold": m.fold, "epoch": m.epoch, "train_loss": m.train_loss,
                                          "val_loss": m.val_loss, "train_macro_f1": m.train_macro_f1,
                                          "val_macro_f1": m.val_macro_f1, "val_accuracy": m.val_accuracy}
                                         for m in e.epochs]
                    last_epochs = n
            yield f"data: {json.dumps(payload)}\n\n"
            if d["status"] in ("completed", "failed", "cancelled"):
                return
            await asyncio.sleep(1.0)
    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/experiments/{exp_id}/evaluations", summary="Stored evaluations by partition")
def evaluations(exp_id: str) -> dict:
    with db.session() as s:
        e = _get(s, exp_id)
        return {ev.partition: {"metrics": ev.metrics, "per_subject": ev.per_subject,
                               "bootstrap": ev.bootstrap, "inference": ev.inference,
                               "records": ev.records, "created_at": iso(ev.created_at)}
                for ev in e.evaluations}


class TestEvalBody(BaseModel):
    purpose: str = Field("", max_length=500)
    confirm: bool = False


@router.post("/experiments/{exp_id}/evaluate-test",
             summary="Evaluate on held-out test subjects (explicit, logged; never for model selection)")
def evaluate_test(exp_id: str, body: TestEvalBody) -> dict:
    require_processed()
    if not body.confirm:
        raise HTTPException(400, "Set confirm=true. Held-out results must not be used to tune models.")
    try:
        res = trainer.evaluate_held_out(exp_id, body.purpose)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"metrics": res["metrics"], "bootstrap": res["bootstrap"], "inference": res["inference"]}


@router.get("/test-access-log", summary="Every access to held-out test labels")
def test_access_log() -> list[dict]:
    with db.session() as s:
        return [{"experiment_id": r.experiment_id, "timestamp": iso(r.timestamp), "purpose": r.purpose}
                for r in s.query(db.TestAccess).order_by(db.TestAccess.timestamp.desc()).all()]


@router.post("/experiments/{exp_id}/activate", summary="Use this CNN for inference")
def activate(exp_id: str) -> dict:
    with db.session() as s:
        e = _get(s, exp_id)
        if e.kind != "cnn" or e.status != "completed":
            raise HTTPException(400, "Only completed CNN experiments can be activated")
        # inference skips experiments whose checkpoint is gone, so activating one would
        # report success while a different model keeps serving predictions
        if not e.checkpoint_path or not trainer.resolve_path(e.checkpoint_path).is_file():
            raise HTTPException(409, f"Model file for experiment {exp_id} is missing; it cannot be activated")
    db.set_setting("active_experiment_id", exp_id)
    return {"active": exp_id}


@router.get("/experiments/{exp_id}/export", summary="Full experiment record as JSON")
def export(exp_id: str) -> dict:
    d = get_experiment(exp_id)
    d["epochs"] = epochs(exp_id)
    d["evaluations"] = evaluations(exp_id)
    return d


@router.delete("/experiments/{exp_id}", summary="Delete an experiment and its checkpoint")
def delete(exp_id: str) -> dict:
    with db.session() as s:
        e = _get(s, exp_id)
        if e.status in ("running", "cancelling"):
            raise HTTPException(409, "Cancel the job first")
        s.delete(e)
        s.commit()
    d = config.MODELS_DIR / exp_id
    if d.exists() and d.resolve().parent == config.MODELS_DIR.resolve():
        shutil.rmtree(d)
    if db.get_setting("active_experiment_id") == exp_id:
        db.set_setting("active_experiment_id", "")
    service.clear_cache()
    return {"deleted": exp_id}


# ------------------------------- inference ---------------------------------------------
class BeatRequest(BaseModel):
    record: str
    beat_index: int
    experiment_id: str | None = None
    explain: Literal["integrated_gradients", "saliency", "grad_cam", "occlusion"] | None = "integrated_gradients"


class RecordRequest(BaseModel):
    record: str
    mode: Literal["reference", "detected"] = "reference"
    experiment_id: str | None = None


class ExplainRequest(BaseModel):
    record: str
    beat_index: int
    method: Literal["integrated_gradients", "saliency", "grad_cam", "occlusion"] = "integrated_gradients"
    target: str | None = None
    experiment_id: str | None = None


def _wrap(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except service.NoModel as exc:
        raise HTTPException(409, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/models/active", summary="The CNN currently used for inference")
def active_model() -> dict | None:
    exp_id = service.active_experiment_id()
    if not exp_id:
        return None
    return get_experiment(exp_id)


@router.post("/inference/beat", summary="Classify one reference beat with the CNN")
def infer_beat(body: BeatRequest) -> dict:
    require_processed()
    if body.record not in config.RECORDS:
        raise HTTPException(404, "Unknown record")
    return _wrap(service.classify_beat, body.record, body.beat_index, body.experiment_id, body.explain)


@router.post("/inference/record", summary="Classify every beat of a recording")
def infer_record(body: RecordRequest) -> dict:
    require_processed()
    if body.record not in config.RECORDS:
        raise HTTPException(404, "Unknown record")
    return _wrap(service.analyze_record, body.record, body.mode, body.experiment_id)


@router.post("/explain", summary="Attribution map for one beat")
def explain(body: ExplainRequest) -> dict:
    require_processed()
    return _wrap(service.explain_beat, body.record, body.beat_index, body.method, body.target,
                 body.experiment_id)


@router.get("/explain/methods", summary="Available explanation methods")
def explain_methods() -> dict:
    return METHODS


@router.get("/explain/sanity", summary="Model-parameter randomisation sanity check")
def explain_sanity(record: str, beat_index: int, experiment_id: str | None = None) -> dict:
    require_processed()
    return _wrap(service.sanity_check, record, beat_index, experiment_id)
