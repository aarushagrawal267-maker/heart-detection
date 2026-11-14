"""Shared helpers for API routers."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException

from .. import config
from ..data import acquire
from ..segmentation import beats


def require_dataset() -> None:
    if not acquire.is_present():
        raise HTTPException(409, "MIT-BIH dataset is not downloaded yet. Use POST /api/dataset/download.")


def require_processed() -> None:
    require_dataset()
    if not beats.is_built():
        raise HTTPException(409, "Processed beat dataset not built yet. Use POST /api/dataset/build.")


def iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def cache_json(name: str, builder):
    """Cache a derived JSON artefact inside the processed directory for this pipeline."""
    path: Path = config.processed_dir() / "cache" / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value
