"""QRS Atlas API - ECG arrhythmia research platform (education and research use only)."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import dataset, models_api, overview, records
from .db import models as db

DESCRIPTION = """
Patient-independent ECG heartbeat classification on the MIT-BIH Arrhythmia Database
with a 1D convolutional neural network trained from scratch in this project.

**Research and education only. Not a medical device. Outputs are not diagnoses.**
"""

app = FastAPI(title="QRS Atlas API", version="1.0.0", description=DESCRIPTION,
              openapi_tags=[{"name": "overview"}, {"name": "dataset"}, {"name": "records"},
                            {"name": "models"}])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:4173", "http://127.0.0.1:4173"],
    allow_methods=["*"], allow_headers=["*"],
)
app.include_router(overview.router)
app.include_router(dataset.router)
app.include_router(records.router)
app.include_router(models_api.router)


@app.on_event("startup")
def _startup() -> None:
    db.engine()
    from .training.registry import import_all
    import_all()
