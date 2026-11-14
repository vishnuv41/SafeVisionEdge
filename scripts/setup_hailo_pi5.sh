#!/usr/bin/env bash
# ==============================================================================
# SafeVision Edge — Turnkey Hailo-8 AI HAT+ Setup on Raspberry Pi 5
# Configures PCIe Gen 3, installs hailo-all, and links HailoRT Python to .venv
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== SafeVision Edge — Hailo-8 AI HAT+ Setup (Raspberry Pi 5) ==="

# 1. Verify Linux aarch64
ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
  echo "ERROR: Unsupported architecture '$ARCH'. Hailo-8 AI HAT+ requires Raspberry Pi 5 64-bit OS (aarch64)."
  exit 1
fi

PI_MODEL="Unknown"
if [[ -r /proc/device-tree/model ]]; then
  PI_MODEL="$(tr -d '\0' </proc/device-tree/model)"
fi
echo "Hardware Board : $PI_MODEL"

# 2. Check and configure PCIe Gen 3 in /boot/firmware/config.txt
CONFIG_TXT="/boot/firmware/config.txt"
[[ ! -f "$CONFIG_TXT" && -f "/boot/config.txt" ]] && CONFIG_TXT="/boot/config.txt"

if [[ -f "$CONFIG_TXT" ]]; then
  echo "Boot config    : $CONFIG_TXT"
  NEED_REBOOT=0

  if ! grep -qE "^dtparam=pciex1" "$CONFIG_TXT"; then
    echo "Adding 'dtparam=pciex1' to $CONFIG_TXT ..."
    echo -e "\n# Raspberry Pi AI HAT+ (Hailo-8 PCIe)\ndtparam=pciex1" | sudo tee -a "$CONFIG_TXT" >/dev/null
    NEED_REBOOT=1
  fi

  if ! grep -qE "^dtparam=pciex1_gen=3" "$CONFIG_TXT"; then
    echo "Adding 'dtparam=pciex1_gen=3' to $CONFIG_TXT ..."
    echo "dtparam=pciex1_gen=3" | sudo tee -a "$CONFIG_TXT" >/dev/null
    NEED_REBOOT=1
  fi

  if [[ "$NEED_REBOOT" -eq 1 ]]; then
    echo "WARNING: PCIe configuration was updated in $CONFIG_TXT."
    echo "A reboot will be required after package installation to initialize the PCIe HAT+."
  fi
fi

# 3. Install official Raspberry Pi Hailo software stack
echo ""
echo "Checking Hailo apt packages (hailo-all)..."
if command -v apt-get >/dev/null 2>&1; then
  if ! dpkg -s hailo-all >/dev/null 2>&1 && ! dpkg -s python3-hailort >/dev/null 2>&1; then
    echo "Installing hailo-all (driver, firmware, hailortcli, python3-hailort)..."
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends hailo-all dkms
  else
    echo "Hailo system packages are already installed."
  fi
else
  echo "WARNING: apt-get not available; ensure hailo-pci and python3-hailort are installed."
fi

# 4. Check PCIe device detection
echo ""
echo "Checking PCIe hardware..."
if command -v lspci >/dev/null 2>&1; then
  if lspci -d 1e60: >/dev/null 2>&1 || lspci | grep -iE "hailo" >/dev/null 2>&1; then
    echo "  [OK] Hailo-8 PCIe device detected on PCI bus:"
    lspci | grep -iE "hailo" || true
  else
    echo "  [WARNING] Hailo-8 PCIe device not detected in lspci."
    echo "  Ensure the AI HAT+ is firmly seated on the PCIe ribbon cable and reboot if you just edited config.txt."
  fi
fi

# 5. Check /dev/hailo0
if compgen -G "/dev/hailo*" >/dev/null; then
  echo "  [OK] Device nodes found: $(ls -d /dev/hailo* 2>/dev/null)"
else
  echo "  [INFO] No /dev/hailo* device nodes yet. (Requires reboot if hailo-all was just installed)."
fi

# 6. Check hailortcli
if command -v hailortcli >/dev/null 2>&1; then
  echo "  [OK] hailortcli is available: $(command -v hailortcli)"
  echo "Querying firmware status..."
  hailortcli fw-control identify 2>&1 | head -n 12 || true
fi

# 7. Ensure .venv has access to hailo_platform
echo ""
echo "Linking HailoRT to SafeVision virtual environment..."
if [[ -f "$ROOT_DIR/.venv/pyvenv.cfg" ]]; then
  sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' "$ROOT_DIR/.venv/pyvenv.cfg" 2>/dev/null || true
  echo "  [OK] .venv updated with include-system-site-packages = true."
fi

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  if "$ROOT_DIR/.venv/bin/python" -c "import hailo_platform; print('  [OK] hailo_platform imported successfully inside .venv')" 2>/dev/null; then
    :
  else
    echo "  [INFO] hailo_platform not yet accessible inside .venv."
    echo "         After rebooting with hailo-all installed, run: ./scripts/install_pi.sh"
  fi
fi

# 8. Check HEF model
echo ""
HEF_FILE="$ROOT_DIR/models/custom_model.hef"
if [[ -f "$HEF_FILE" ]]; then
  HEF_SIZE="$(du -h "$HEF_FILE" | cut -f1)"
  echo "  [OK] custom_model.hef present ($HEF_SIZE)"
else
  echo "  [INFO] models/custom_model.hef is not present yet."
  echo "         SafeVision will run on CPU (ExecuTorch INT8 PT2E) and report NPU_MODEL_NOT_READY."
  echo "         When compiled with Hailo DFC, copy your .hef file to: models/custom_model.hef"
fi

echo ""
echo "=== Hailo-8 Setup Finished ==="
echo "To verify overall system health, run:"
echo "  ./scripts/check_pi_env.py"
