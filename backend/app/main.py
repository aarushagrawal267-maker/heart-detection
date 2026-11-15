"""QRS Atlas API - ECG arrhythmia research platform (education and research use only)."""
from __future__ import annotations

import base64
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .api import dataset, models_api, overview, records
from .db import models as db

DESCRIPTION = """
Patient-independent ECG heartbeat classification on the MIT-BIH Arrhythmia Database
with a 1D convolutional neural network trained from scratch in this project.

**Research and education only. Not a medical device. Outputs are not diagnoses.**
"""


class BasicAuthMiddleware:
    """HTTP Basic authentication for private deployments (pure ASGI, so SSE streams work).

    Enabled only when QRS_AUTH_PASSWORD is set. /api/health stays open for the hosting
    platform's health checks; it reveals nothing but "ok".
    """

    OPEN_PATHS = {"/api/health"}

    def __init__(self, app, username: str, password: str):
        self.app = app
        self.expected = base64.b64encode(f"{username}:{password}".encode()).decode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") in self.OPEN_PATHS:
            return await self.app(scope, receive, send)
        header = dict(scope.get("headers") or []).get(b"authorization", b"").decode()
        scheme, _, token = header.partition(" ")
        if scheme.lower() == "basic" and secrets.compare_digest(token.strip(), self.expected):
            return await self.app(scope, receive, send)
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"www-authenticate", b'Basic realm="QRS Atlas", charset="UTF-8"'),
                                (b"content-type", b"text/plain; charset=utf-8")]})
        await send({"type": "http.response.body", "body": b"Authentication required"})


app = FastAPI(title="QRS Atlas API", version="1.0.0", description=DESCRIPTION,
              openapi_tags=[{"name": "overview"}, {"name": "dataset"}, {"name": "records"},
                            {"name": "models"}])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:4173", "http://127.0.0.1:4173"],
    allow_methods=["*"], allow_headers=["*"],
)
_password = os.environ.get("QRS_AUTH_PASSWORD", "")
if _password:
    app.add_middleware(BasicAuthMiddleware, username=os.environ.get("QRS_AUTH_USER", "qrs"),
                       password=_password)

app.include_router(overview.router)
app.include_router(dataset.router)
app.include_router(records.router)
app.include_router(models_api.router)


@app.on_event("startup")
def _startup() -> None:
    db.engine()
    from .training.registry import import_all
    import_all()


# ---- production: serve the built frontend from the same origin -------------------------
STATIC_DIR = Path(os.environ.get("QRS_STATIC_DIR", config.ROOT / "frontend" / "dist"))
if (STATIC_DIR / "index.html").exists():
    if (STATIC_DIR / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "Not found")
        candidate = (STATIC_DIR / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(STATIC_DIR.resolve()):
            return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")   # client-side routes
