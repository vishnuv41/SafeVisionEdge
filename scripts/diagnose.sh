#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:---full}"
if [[ "$MODE" != "--full" && "$MODE" != "--benchmark" ]]; then
  echo "Usage: $0 [--full | --benchmark]"
  exit 1
fi

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

mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/outputs/reports"
STAMP="$(date -u +%Y%m%d_%H%M%SZ)"
LOG_FILE="$ROOT_DIR/logs/diagnose_${MODE#--}_${STAMP}.log"

{
  echo "=== SafeVision Edge Diagnostics ($MODE) @ $STAMP ==="
  echo "Host Kernel : $(uname -a)"
  if [[ -r /proc/device-tree/model ]]; then
    echo "Board Model : $(tr -d '\0' </proc/device-tree/model)"
  fi
  if [[ -r /etc/os-release ]]; then
    grep -E '^(PRETTY_NAME|VERSION_CODENAME)=' /etc/os-release || true
  fi
  echo "Python Bin  : $PY_BIN ($("$PY_BIN" --version 2>&1))"
  if command -v vcgencmd >/dev/null 2>&1; then
    echo "Pi Temp     : $(vcgencmd measure_temp || true)"
    echo "Pi Throttle : $(vcgencmd get_throttled || true)"
  fi
  echo ""

  if [[ "$MODE" == "--full" ]]; then
    echo "[1/3] Environment, XNNPACK, SHA-256 & 1-frame inference check..."
    "$PY_BIN" "$ROOT_DIR/scripts/check_pi_env.py"

    echo ""
    echo "[2/3] Standalone unit & FastAPI + real-.pte integration suite..."
    "$PY_BIN" -m unittest discover -s "$ROOT_DIR/tests" -v

    echo ""
    echo "[3/3] 3-frame hardware validation & timestamped report..."
    "$PY_BIN" "$ROOT_DIR/scripts/validate_pi_hardware.py" --runs 3 --warmup 1
  else
    echo "[1/2] Environment & SHA-256 preflight check..."
    "$PY_BIN" "$ROOT_DIR/scripts/check_pi_env.py" --skip-infer

    echo ""
    echo "[2/2] Sustained 20-frame ExecuTorch XNNPACK benchmark & thermal report..."
    "$PY_BIN" "$ROOT_DIR/scripts/validate_pi_hardware.py" --runs 20 --warmup 5 --update-dashboard-benchmark
  fi

  echo ""
  echo "Diagnostic log saved to: $LOG_FILE"
} 2>&1 | tee "$LOG_FILE"