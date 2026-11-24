#!/usr/bin/env bash
# Start QRS Atlas on Linux / macOS (the Windows equivalent is start.ps1).
#   Backend  http://localhost:8000  (API docs at /docs)
#   Frontend http://localhost:5173
# Press Ctrl+C to stop both. If those ports are taken, choose others:
#   QRS_BACKEND_PORT=8001 QRS_FRONTEND_PORT=5174 ./start.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
[ -x .venv/bin/python ] || { echo "ERROR: Python environment missing. Run ./setup.sh first." >&2; exit 1; }
[ -f frontend/node_modules/vite/bin/vite.js ] || { echo "ERROR: Frontend packages missing. Run ./setup.sh first." >&2; exit 1; }
mkdir -p .run
BPORT="${QRS_BACKEND_PORT:-8000}"
FPORT="${QRS_FRONTEND_PORT:-5173}"

port_busy() {
  if command -v lsof >/dev/null; then lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
  else (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; fi
}
for p in "$BPORT" "$FPORT"; do
  if port_busy "$p"; then
    echo "ERROR: port $p is already in use. Stop the other program, or pick free ports, e.g." >&2
    echo "       QRS_BACKEND_PORT=8001 QRS_FRONTEND_PORT=5174 ./start.sh" >&2
    exit 1
  fi
done

PIDS=()
cleanup() {
  echo; echo "Stopping QRS Atlas ..."
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Starting backend on http://localhost:$BPORT"
(cd "$ROOT/backend" && exec "$ROOT/.venv/bin/python" -m uvicorn app.main:app --host 127.0.0.1 --port "$BPORT") \
  > .run/backend.log 2>&1 &
PIDS+=($!)
echo "==> Starting frontend on http://localhost:$FPORT"
(cd "$ROOT/frontend" && QRS_BACKEND_URL="http://127.0.0.1:$BPORT" exec node node_modules/vite/bin/vite.js --port "$FPORT" --strictPort --host 127.0.0.1) \
  > .run/frontend.log 2>&1 &
PIDS+=($!)

wait_url() {
  for _ in $(seq 1 90); do
    if "$ROOT/.venv/bin/python" -c "import urllib.request,sys; urllib.request.urlopen(sys.argv[1], timeout=2)" "$1" 2>/dev/null; then
      echo "    $2 is up"; return 0
    fi
    sleep 1
  done
  echo "ERROR: $2 did not start. See .run/$3" >&2; tail -n 30 ".run/$3" >&2; return 1
}
wait_url "http://127.0.0.1:$BPORT/api/health" Backend backend.log
wait_url "http://127.0.0.1:$FPORT/" Frontend frontend.log

echo
echo "QRS Atlas is running:  http://localhost:$FPORT   (API docs: http://localhost:$BPORT/docs)"
echo "Press Ctrl+C to stop."
wait
