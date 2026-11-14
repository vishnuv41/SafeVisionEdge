#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT_DIR/.safevision.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found at $PID_FILE."
  pkill -f "uvicorn api:app --app-dir $ROOT_DIR/src" >/dev/null 2>&1 && echo "Stopped matching SafeVision Edge process." || echo "SafeVision Edge is not running."
  exit 0
fi

PID="$(cat "$PID_FILE")"
if [[ -z "$PID" ]] || ! kill -0 "$PID" >/dev/null 2>&1; then
  echo "SafeVision Edge PID $PID is not active; cleaning stale PID file."
  rm -f "$PID_FILE"
  exit 0
fi

echo "Stopping SafeVision Edge (PID $PID)..."
kill "$PID"
for _ in $(seq 1 10); do
  if ! kill -0 "$PID" >/dev/null 2>&1; then
    rm -f "$PID_FILE"
    echo "SafeVision Edge stopped cleanly."
    exit 0
  fi
  sleep 0.5
done

echo "Force stopping PID $PID..."
kill -9 "$PID" >/dev/null 2>&1 || true
rm -f "$PID_FILE"
echo "SafeVision Edge stopped."