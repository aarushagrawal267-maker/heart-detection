# QRS Atlas - single-image deployment (frontend served by the FastAPI backend).
# Research and education use only. Not a medical device.

# ---------- 1. build the React frontend ----------
FROM node:22-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---------- 2. Python runtime ----------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QRS_ROOT=/app \
    QRS_STATIC_DIR=/app/frontend/dist
RUN useradd --create-home --uid 1000 qrs && mkdir -p /app && chown qrs:qrs /app
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r backend/requirements.txt

# Files are owned by the runtime user at copy time (a later chown would duplicate layers).
COPY --chown=qrs:qrs configs/ configs/
COPY --chown=qrs:qrs backend/ backend/
COPY --chown=qrs:qrs models/ models/
COPY --chown=qrs:qrs --from=frontend /build/dist frontend/dist

USER qrs
WORKDIR /app/backend
# Download MIT-BIH from PhysioNet at build time, verify every file against the published
# SHA-256 checksums and build the beat set, so the container starts ready.
RUN python -m app.cli download

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/api/health', timeout=4)"
# PORT is provided by Render / Railway; Fly.io and Hugging Face use 8000.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
