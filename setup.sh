#!/usr/bin/env bash
# QRS Atlas setup for Linux / macOS (the Windows equivalent is setup.ps1).
#   ./setup.sh              CPU PyTorch, download + verify MIT-BIH, build beat set, install frontend
#   ./setup.sh --skip-data  skip the dataset download (do it later from the Dataset Explorer)
# Research and education only. Not a medical device.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
SKIP_DATA=0
for a in "$@"; do
  case "$a" in
    --skip-data) SKIP_DATA=1 ;;
    -h|--help) sed -n 2,5p "$0"; exit 0 ;;
    *) echo "Unknown option: $a" >&2; exit 2 ;;
  esac
done

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || { echo "ERROR: $PY not found. Install Python 3.11-3.13." >&2; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)' \
  || echo "WARNING: Python 3.11-3.13 is recommended; found $("$PY" --version)."
command -v node >/dev/null || { echo "ERROR: Node.js 18+ not found on PATH." >&2; exit 1; }
command -v npm >/dev/null || { echo "ERROR: npm not found on PATH." >&2; exit 1; }

TORCH_VERSION="$(grep -E '^TORCH_VERSION=' backend/torch-version.txt | cut -d= -f2)"

echo "==> Creating Python environment (.venv)"
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
echo "==> Installing PyTorch ${TORCH_VERSION} (CPU build)"
pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
echo "==> Installing backend requirements"
pip install -r backend/requirements.txt

cd "$ROOT/backend"
if [ "$SKIP_DATA" -eq 0 ]; then
  echo "==> Downloading and verifying the MIT-BIH Arrhythmia Database (about 100 MB)"
  python -m app.cli download
else
  echo "==> Skipping dataset download (--skip-data)"
fi
echo "==> Creating the SQLite database and importing the bundled models"
python -m app.cli init-db

cd "$ROOT/frontend"
echo "==> Installing frontend packages"
npm ci --no-audit --no-fund

echo
echo "Setup complete. Start the app with:  ./start.sh"
