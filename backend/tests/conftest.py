import os

import pytest

from app import config
from app.data import acquire
from app.db import models as db
from app.segmentation import beats


def dataset_ready() -> bool:
    return acquire.is_present() and beats.is_built()


requires_dataset = pytest.mark.skipif(not dataset_ready(), reason="MIT-BIH dataset not downloaded/processed")


@pytest.fixture()
def temp_store(tmp_path, monkeypatch):
    """Isolated SQLite database and model directory (also inherited by worker processes)."""
    dbp = tmp_path / "test.db"
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(config, "DB_PATH", dbp)
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setenv("QRS_DB_PATH", str(dbp))
    monkeypatch.setenv("QRS_MODELS_DIR", str(models))
    db.reset_engine()
    from app.inference import service
    service.clear_cache()
    yield tmp_path
    db.reset_engine()
    service.clear_cache()
