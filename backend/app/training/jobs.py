"""Experiment creation and the single-slot training job runner."""
from __future__ import annotations

import os
import secrets
import signal
import subprocess
import sys
from datetime import datetime, timezone

from .. import config
from ..db import models as db
from . import trainer

_PROCS: dict[str, subprocess.Popen] = {}


def new_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)


def create(name: str, kind: str, cfg: dict, notes: str = "") -> db.Experiment:
    base = trainer.DEFAULT_CNN_CONFIG if kind == "cnn" else trainer.DEFAULT_BASELINE_CONFIG
    unknown = set(cfg) - set(base)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    full = {**base, **cfg}
    exp = db.Experiment(id=new_id(), name=name or f"{full['model']} experiment", kind=kind,
                        model_name=full["model"], status="queued", config=full,
                        pipeline=config.PIPELINE, pipeline_hash=config.pipeline_hash(),
                        total_epochs=full.get("epochs", 0) if kind == "cnn" else 0, notes=notes)
    with db.session() as s:
        s.add(exp)
        s.commit()
    return exp


def running() -> db.Experiment | None:
    reap()
    with db.session() as s:
        return s.query(db.Experiment).filter(
            db.Experiment.status.in_(["running", "cancelling", "queued"])).filter(
            db.Experiment.pid.isnot(None)).first()


def start(exp_id: str) -> None:
    """Launch the worker process for an experiment (one job at a time)."""
    if running() is not None:
        raise RuntimeError("Another training job is already running")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(config.ROOT / "backend") + os.pathsep + env.get("PYTHONPATH", "")
    log_dir = config.MODELS_DIR / exp_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / "train.log", "w", encoding="utf-8")  # noqa: SIM115 - owned by child
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    proc = subprocess.Popen([sys.executable, "-m", "app.training.worker", exp_id],
                            cwd=str(config.ROOT / "backend"), env=env, stdout=log,
                            stderr=subprocess.STDOUT, **kwargs)
    _PROCS[exp_id] = proc
    with db.session() as s:
        exp = s.get(db.Experiment, exp_id)
        exp.pid = proc.pid
        exp.message = "starting worker"
        s.commit()


def cancel(exp_id: str) -> None:
    with db.session() as s:
        exp = s.get(db.Experiment, exp_id)
        if exp is None:
            raise KeyError(exp_id)
        if exp.status == "queued" and exp.pid is None and exp.id not in _PROCS:
            exp.status = "cancelled"          # never started: nothing to stop
            exp.message = "cancelled before start"
            s.commit()
        elif exp.status in ("running", "queued"):
            exp.status = "cancelling"
            s.commit()


def reap() -> None:
    """Mark experiments whose worker died without reporting as failed."""
    with db.session() as s:
        for exp in s.query(db.Experiment).filter(
                db.Experiment.status.in_(["running", "cancelling", "queued"])).all():
            if exp.pid is None:
                continue
            proc = _PROCS.get(exp.id)
            alive = proc.poll() is None if proc is not None else _pid_alive(exp.pid)
            if not alive:
                exp.status = "cancelled" if exp.status == "cancelling" else "failed"
                exp.error = exp.error or "worker process exited unexpectedly (see train.log)"
                exp.finished_at = datetime.now(timezone.utc)
                exp.pid = None
        s.commit()


def _pid_alive(pid: int) -> bool:
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
            return str(pid) in out.stdout
        os.kill(pid, 0)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def kill(exp_id: str) -> None:
    proc = _PROCS.get(exp_id)
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
