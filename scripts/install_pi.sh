#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXPECTED_MODEL_SHA256="e9c932aa70edbe3d8065612b576aaa25b4cd86840a3017864461a436d5959b56"
MODEL_FILE="$ROOT_DIR/models/custom_model.pte"

echo "=== SafeVision Edge — Raspberry Pi 5 ARM64 Setup ==="

# 1. OS & ARM64 verification (strictly reject 32-bit ARM and non-Linux hosts)
OS_NAME="$(uname -s)"
ARCH="$(uname -m)"
echo "Host OS / Arch : $OS_NAME / $ARCH"
if [[ -r /etc/os-release ]]; then
  OS_PRETTY="$(grep -E '^PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '"' || true)"
  echo "OS Release     : ${OS_PRETTY:-Unknown Linux}"
  if [[ "${OS_PRETTY,,}" != *"debian"* && "${OS_PRETTY,,}" != *"raspbian"* && "${OS_PRETTY,,}" != *"raspberry"* && "${OS_PRETTY,,}" != *"ubuntu"* ]]; then
    echo "WARNING: Non-Debian/Raspberry Pi OS detected ($OS_PRETTY); apt prerequisites assume Debian/Raspberry Pi OS."
  fi
fi

if [[ "$OS_NAME" != "Linux" ]]; then
  echo "ERROR: install_pi.sh must be executed on Linux (Raspberry Pi OS 64-bit). Detected: $OS_NAME"
  exit 1
fi

case "$ARCH" in
  aarch64|arm64) ;;
  arm*|armv7*|armv6*|i686|i386)
    echo "ERROR: Unsupported 32-bit architecture '$ARCH'. ExecuTorch ARM64 wheels require 64-bit Raspberry Pi OS (aarch64)."
    exit 1
    ;;
  x86_64)
    echo "ERROR: Detected x86_64. install_pi.sh targets Raspberry Pi 5 ARM64 (aarch64)."
    exit 1
    ;;
  *)
    echo "ERROR: Unsupported CPU architecture '$ARCH'. Expected aarch64."
    exit 1
    ;;
esac

if command -v getconf >/dev/null 2>&1; then
  USERLAND_BITS="$(getconf LONG_BIT || echo 64)"
  if [[ "$USERLAND_BITS" != "64" ]]; then
    echo "ERROR: 32-bit userland ($USERLAND_BITS-bit) detected. Install 64-bit Raspberry Pi OS."
    exit 1
  fi
fi

# 2. Detect Raspberry Pi hardware model where exposed
PI_MODEL="Unknown ARM64 Linux device"
if [[ -r /proc/device-tree/model ]]; then
  PI_MODEL="$(tr -d '\0' </proc/device-tree/model)"
fi
echo "Hardware model : $PI_MODEL"
if [[ "$PI_MODEL" != *"Raspberry Pi 5"* ]]; then
  echo "WARNING: Expected 'Raspberry Pi 5', detected '$PI_MODEL'. Proceeding on ARM64 Linux."
fi

# 3. Check Python version compatibility (executorch==1.6.0 requires Python 3.11-3.14)
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is not installed. Run: sudo apt update && sudo apt install -y python3 python3-venv"
  exit 1
fi

PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
echo "Python version : $PY_VER"
python3 - <<'PY'
import sys
if sys.version_info < (3, 11) or sys.version_info >= (3, 15):
    raise SystemExit(
        f"ERROR: Python {sys.version.split()[0]} is not supported by executorch==1.6.0 wheels (requires 3.11 - 3.14; numpy==2.3.5 has cp311-cp314 aarch64 wheels)."
    )
PY

# 4. Check glibc >= 2.28 (required by manylinux_2_28_aarch64 wheels)
if command -v ldd >/dev/null 2>&1; then
  GLIBC_VER="$(ldd --version 2>/dev/null | head -n1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
  if [[ -n "$GLIBC_VER" ]]; then
    echo "glibc version  : $GLIBC_VER"
    python3 - "$GLIBC_VER" <<'PY'
import sys
parts = tuple(int(x) for x in sys.argv[1].split("."))
if parts < (2, 28):
    raise SystemExit(f"ERROR: glibc {sys.argv[1]} < 2.28 required by ExecuTorch manylinux_2_28_aarch64 wheels.")
PY
  fi
fi

# 5. Check disk space (>= 2.5 GiB free) and RAM (>= 1.5 GiB minimum, warn < 3.5 GiB)
FREE_MB="$(df -Pm "$ROOT_DIR" | awk 'NR==2 {print $4}')"
echo "Free disk space: ${FREE_MB} MiB"
if (( FREE_MB < 2560 )); then
  echo "ERROR: Insufficient free disk space (${FREE_MB} MiB < 2560 MiB needed for Torch CPU + ExecuTorch + OpenCV)."
  exit 1
fi

if [[ -r /proc/meminfo ]]; then
  MEM_TOTAL_MB="$(awk '/MemTotal:/ {printf "%.0f", $2/1024}' /proc/meminfo)"
  echo "System RAM     : ${MEM_TOTAL_MB} MiB"
  if (( MEM_TOTAL_MB < 1536 )); then
    echo "ERROR: System RAM too low (${MEM_TOTAL_MB} MiB < 1536 MiB)."
    exit 1
  elif (( MEM_TOTAL_MB < 3500 )); then
    echo "WARNING: Detected ${MEM_TOTAL_MB} MiB RAM. A 4 GB or 8 GB Raspberry Pi 5 is strongly recommended."
  fi
fi

# 6. Verify trained .pte model exists and has untouched SHA-256 before setup
if [[ ! -f "$MODEL_FILE" ]]; then
  echo "ERROR: Trained model file missing: $MODEL_FILE"
  exit 1
fi
ACTUAL_SHA="$(sha256sum "$MODEL_FILE" | awk '{print tolower($1)}')"
if [[ "$ACTUAL_SHA" != "$EXPECTED_MODEL_SHA256" ]]; then
  echo "ERROR: Model SHA-256 mismatch!"
  echo "  Expected: $EXPECTED_MODEL_SHA256"
  echo "  Actual  : $ACTUAL_SHA"
  exit 1
fi
echo "Model verified : custom_model.pte (SHA-256 matches fine-tuned INT8 export)"

# 7. Check & install required Debian/Raspberry Pi OS apt prerequisites
if command -v apt-get >/dev/null 2>&1; then
  MISSING_PKGS=()
  for pkg in python3-venv libglib2.0-0 ffmpeg curl ca-certificates; do
    dpkg -s "$pkg" >/dev/null 2>&1 || MISSING_PKGS+=("$pkg")
  done
  # OpenGL helper package name varies between Bookworm (libgl1) and minimal variants
  if ! dpkg -s libgl1 >/dev/null 2>&1; then
    MISSING_PKGS+=("libgl1")
  fi
  if (( ${#MISSING_PKGS[@]} > 0 )); then
    echo "Installing missing system packages: ${MISSING_PKGS[*]}"
    if [[ "$(id -u)" -eq 0 ]]; then
      apt-get update && apt-get install -y --no-install-recommends "${MISSING_PKGS[@]}"
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get update && sudo apt-get install -y --no-install-recommends "${MISSING_PKGS[@]}"
    else
      echo "ERROR: Missing system packages (${MISSING_PKGS[*]}) and sudo is unavailable."
      exit 1
    fi
  fi
else
  echo "WARNING: apt-get not available; ensure libglib2.0, libgl1, ffmpeg, curl, and python3 venv are installed."
fi

# 8. Identify / Link Python Virtual Environment & Install Dependencies Safely
# If ~/hailo-venv exists, prioritize it to preserve HailoRT 4.24.0 and numpy C-ABI.
HAILO_VENV="${HOME}/hailo-venv"
if [[ -d "$HAILO_VENV" && -x "$HAILO_VENV/bin/python" ]]; then
  echo "Found pre-existing HailoRT virtual environment at $HAILO_VENV"
  # shellcheck disable=SC1090
  source "$HAILO_VENV/bin/activate"
elif [[ ! -d "$ROOT_DIR/.venv" ]]; then
  echo "Creating Python virtual environment at $ROOT_DIR/.venv (with system-site-packages for HailoRT)..."
  python3 -m venv --system-site-packages "$ROOT_DIR/.venv"
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
else
  if [[ -f "$ROOT_DIR/.venv/pyvenv.cfg" ]]; then
    sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' "$ROOT_DIR/.venv/pyvenv.cfg" 2>/dev/null || true
  fi
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
fi

# Use safe pre-check script to install only missing dependencies non-destructively
# Skips pip/setuptools upgrade and aggressive uninstalls to strictly protect HailoRT bindings
echo "--- Checking & Installing Missing Dependencies Safely ---"
python "$ROOT_DIR/scripts/check_venv_deps.py"

chmod +x "$ROOT_DIR"/scripts/*.sh

# 9. Verify actual runtime imports, XNNPACK backend, model load, and real test image inference
echo ""
echo "Running strict runtime, XNNPACK, SHA-256, and 1-frame inference verification..."
python "$ROOT_DIR/scripts/check_pi_env.py"

echo ""
echo "=== Installation & Verification Complete ==="
echo "Next steps:"
echo "  1. Run full diagnostics & Pi benchmark: ./scripts/diagnose.sh --benchmark"
echo "  2. Start offline SafeVision Edge server: ./scripts/start_safevision.sh"