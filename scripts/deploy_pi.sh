#!/usr/bin/env bash
# ==============================================================================
# SafeVision Edge — Raspberry Pi 5 + Hailo-8 AI HAT+ Automated Deployment Script
# ==============================================================================
# Preserves existing working HailoRT 4.24.0, firmware, and ~/hailo-venv.
# Installs pure ARM64 CPU PyTorch/ExecuTorch dependencies without NVIDIA CUDA bloat.
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=================================================================="
echo " SafeVision Edge — Deployment on Raspberry Pi 5 + Hailo-8 AI HAT+"
echo "=================================================================="
echo "Deployment Root : $ROOT_DIR"
echo "Host Machine    : $(uname -m) ($(uname -s))"
echo "Kernel Release  : $(uname -r)"

# 1. Environment and Architecture Check
ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
  echo "WARNING: Expected ARM64 (aarch64) architecture on Raspberry Pi 5, got '$ARCH'."
  echo "Proceeding, but precompiled binaries and PCIe bindings are target-specific."
fi

# 2. Identify / Link Python Virtual Environment
# Priority: use ~/hailo-venv if it exists to preserve precompiled HailoRT bindings
HAILO_VENV="${HOME}/hailo-venv"
LOCAL_VENV="$ROOT_DIR/.venv"

if [[ -d "$HAILO_VENV" && -x "$HAILO_VENV/bin/python" ]]; then
  echo "[OK] Found pre-existing HailoRT virtual environment: $HAILO_VENV"
  PY_BIN="$HAILO_VENV/bin/python"
  PIP_BIN="$HAILO_VENV/bin/pip"
elif [[ -d "$LOCAL_VENV" && -x "$LOCAL_VENV/bin/python" ]]; then
  echo "[OK] Found local virtual environment: $LOCAL_VENV"
  PY_BIN="$LOCAL_VENV/bin/python"
  PIP_BIN="$LOCAL_VENV/bin/pip"
else
  echo "[INFO] Creating clean virtual environment at $LOCAL_VENV ..."
  python3 -m venv "$LOCAL_VENV"
  PY_BIN="$LOCAL_VENV/bin/python"
  PIP_BIN="$LOCAL_VENV/bin/pip"
fi

echo "Active Python   : $($PY_BIN --version) ($PY_BIN)"

# 3. Verify CPU Model Hash Integrity (Ground Truth IDD YOLOv11-S)
EXPECTED_PTE_HASH="e9c932aa70edbe3d8065612b576aaa25b4cd86840a3017864461a436d5959b56"
PTE_FILE="$ROOT_DIR/models/custom_model.pte"
if [[ ! -f "$PTE_FILE" ]]; then
  echo "ERROR: Ground truth CPU model missing: $PTE_FILE"
  exit 1
fi
ACTUAL_PTE_HASH="$(sha256sum "$PTE_FILE" | awk '{print tolower($1)}')"
if [[ "$ACTUAL_PTE_HASH" != "$EXPECTED_PTE_HASH" ]]; then
  echo "ERROR: CPU model hash mismatch!"
  echo "  Expected: $EXPECTED_PTE_HASH"
  echo "  Actual  : $ACTUAL_PTE_HASH"
  exit 1
fi
echo "[OK] Verified custom_model.pte SHA-256: $ACTUAL_PTE_HASH"

# 4. Check Hailo-8 Hardware & Driver
echo "--- Checking Hailo-8 Hardware ---"
if command -v hailortcli >/dev/null 2>&1; then
  echo "[OK] hailortcli found at $(command -v hailortcli)"
  hailortcli --version || true
  hailortcli fw-control identify || true
else
  echo "[INFO] hailortcli not found in PATH."
fi

if [[ -e /dev/hailo0 ]]; then
  echo "[OK] PCIe device node /dev/hailo0 exists."
else
  echo "[INFO] /dev/hailo0 not currently visible. (Is the AI HAT+ firmly attached?)"
fi

# 5. Check Hailo Python Extension (without reinstalling)
if "$PY_BIN" -c "import hailo_platform; print('hailo_platform version:', getattr(hailo_platform, '__version__', 'unknown'))" 2>/dev/null; then
  echo "[OK] hailo_platform module is loaded and operational."
else
  echo "[INFO] hailo_platform is not present in $PY_BIN."
  if [[ -f "/usr/lib/python3/dist-packages/hailo_platform" || -d "/usr/local/lib/python3/dist-packages/hailo_platform" ]]; then
    echo "[INFO] Global hailo_platform detected; linking to venv..."
  fi
fi

# 6. Install SafeVision Dependencies (Strictly CPU index for torch, non-destructive check)
echo "--- Ensuring SafeVision Backend Dependencies (Safe Non-Destructive Mode) ---"
"$PY_BIN" "$ROOT_DIR/scripts/check_venv_deps.py"

# 7. Run Hardware and System Diagnostics
echo "--- Running Full SafeVision Diagnostics ---"
"$PY_BIN" "$ROOT_DIR/scripts/check_pi_env.py"

echo "=================================================================="
echo " SafeVision Edge Deployment Complete!"
echo " Start the server with:"
echo "   $ROOT_DIR/scripts/start_safevision.sh"
echo " Or run in foreground:"
echo "   $ROOT_DIR/scripts/start_safevision.sh --foreground"
echo "=================================================================="
