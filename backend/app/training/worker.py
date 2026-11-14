"""Training worker process: `python -m app.training.worker <experiment_id>`.

Training runs in its own process so the API stays responsive; progress and metrics are
written to SQLite and streamed to the UI.
"""
from __future__ import annotations

import sys

from .trainer import run_experiment

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m app.training.worker <experiment_id>")
    run_experiment(sys.argv[1])
