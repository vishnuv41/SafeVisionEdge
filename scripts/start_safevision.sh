#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${SAFEVISION_HOST:-0.0.0.0}"
PORT="${SAFEVISION_PORT:-8000}"
PID_FILE="$ROOT_DIR/.safevision.pid"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/safevision.log"

if [[ -x "${HOME:-}/hailo-venv/bin/python" ]]; then
  PY_BIN="${HOME}/hailo-venv/bin/python"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PY_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/.venv-cpu-test/bin/python" ]]; then
  PY_BIN="$ROOT_DIR/.venv-cpu-test/bin/python"
elif [[ -x "${HOME:-}/.venv-cpu-test/bin/python" ]]; then
  PY_BIN="${HOME}/.venv-cpu-test/bin/python"
else
  PY_BIN="$(command -v python3)"
fi

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" >/dev/null 2>&1; then
    echo "SafeVision Edge is already running (PID $OLD_PID)."
    echo "Open dashboard: http://127.0.0.1:${PORT}/"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

if [[ "${1:-}" == "--foreground" ]]; then
  echo "Starting SafeVision Edge in foreground on http://${HOST}:${PORT}/ ..."
  exec "$PY_BIN" -m uvicorn api:app --app-dir "$ROOT_DIR/src" --host "$HOST" --port "$PORT"
fi

mkdir -p "$LOG_DIR" "$ROOT_DIR/outputs"
nohup "$PY_BIN" -m uvicorn api:app --app-dir "$ROOT_DIR/src" --host "$HOST" --port "$PORT" >"$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" >"$PID_FILE"

echo "Waiting for SafeVision Edge backend + ExecuTorch model startup (PID $NEW_PID)..."
READY=0
for _ in $(seq 1 30); do
  if ! kill -0 "$NEW_PID" >/dev/null 2>&1; then
    echo "ERROR: SafeVision Edge exited unexpectedly. Recent logs:"
    tail -n 30 "$LOG_FILE" || true
    rm -f "$PID_FILE"
    exit 1
  fi
  if command -v curl >/dev/null 2>&1 && curl -fsS "http://127.0.0.1:${PORT}/api/v1/system/health" >/dev/null 2>&1; then
    READY=1
    break
  elif "$PY_BIN" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/api/v1/system/health', timeout=1)" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [[ "$READY" -ne 1 ]]; then
  echo "WARNING: Server started (PID $NEW_PID) but health check did not answer within 30s."
  echo "Check logs: tail -n 50 $LOG_FILE"
  exit 1
fi

LAN_IP=""
if command -v hostname >/dev/null 2>&1; then
  LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
fi

echo "SafeVision Edge is ONLINE (offline local inference active)."
echo "  Pi browser URL : http://127.0.0.1:${PORT}/"
if [[ -n "$LAN_IP" ]]; then
  echo "  LAN browser URL: http://${LAN_IP}:${PORT}/"
fi
echo "  Logs           : $LOG_FILE"
echo "  Stop command   : ./scripts/stop_safevision.sh"