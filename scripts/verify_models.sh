#!/usr/bin/env bash
# ==============================================================================
# SafeVision Edge — Model Integrity & Provenance Verification Script
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

echo "=================================================================="
echo " SafeVision Edge — Model Integrity Verification"
echo "=================================================================="

# 1. CPU Model Verification
EXPECTED_PTE_SHA256="e9c932aa70edbe3d8065612b576aaa25b4cd86840a3017864461a436d5959b56"
PTE_PATH="$ROOT_DIR/models/custom_model.pte"

echo -n "Checking CPU Model ($PTE_PATH) ... "
if [[ ! -f "$PTE_PATH" ]]; then
  echo "FAIL (File missing)"
  exit 1
fi

ACTUAL_PTE_SHA256="$(sha256sum "$PTE_PATH" | awk '{print tolower($1)}')"
if [[ "$ACTUAL_PTE_SHA256" == "$EXPECTED_PTE_SHA256" ]]; then
  echo "PASS"
  echo "  SHA-256: $ACTUAL_PTE_SHA256 (Ground truth YOLOv11-S INT8 ExecuTorch)"
else
  echo "FAIL (SHA-256 mismatch!)"
  echo "  Expected: $EXPECTED_PTE_SHA256"
  echo "  Actual  : $ACTUAL_PTE_SHA256"
  exit 1
fi

# 2. Hailo NPU Model Verification (Single Configured Path: models/npu/custom_model.hef)
HEF_FILE="$ROOT_DIR/models/npu/custom_model.hef"

echo -n "Checking Hailo NPU Model ($HEF_FILE) ... "
if [[ ! -f "$HEF_FILE" ]]; then
  echo "NOT READY (custom_model.hef missing)"
  echo "  Note: Compile fine-tuned ONNX -> HEF using ./conversion/compile_hef.sh on x86_64 Linux."
else
  HEF_SHA256="$(sha256sum "$HEF_FILE" | awk '{print tolower($1)}')"
  echo "FOUND ($HEF_FILE)"
  echo "  SHA-256: $HEF_SHA256"
  if command -v hailortcli >/dev/null 2>&1; then
    echo "  Running hailortcli parse-hef ..."
    hailortcli parse-hef "$HEF_FILE" || true
  fi
fi

echo "=================================================================="
echo " Model verification check complete."
echo "=================================================================="
