#!/usr/bin/env bash
# ==============================================================================
# SafeVision Edge — Hailo-8 HEF Compilation Script (Linux x86_64)
# ==============================================================================
# Run this script on an x86_64 Linux machine (Ubuntu 22.04 / 24.04)
# with Hailo Dataflow Compiler (DFC) installed, or via the official Hailo Docker.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

ONNX_MODEL="$ROOT_DIR/models/npu/custom_model.onnx"
CALIB_NPY="$ROOT_DIR/conversion/calib_nhwc_uint8.npy"
ALLS_SCRIPT="$ROOT_DIR/conversion/yolov11s_hailo.alls"
OUT_HEF="$ROOT_DIR/models/npu/custom_model.hef"
NET_NAME="safevision_yolov11s"

echo "=================================================================="
echo " SafeVision Edge — Compiling 7-Class YOLOv11-S to Hailo-8 HEF"
echo "=================================================================="
echo "ONNX Model   : $ONNX_MODEL"
echo "Calibration  : $CALIB_NPY"
echo "Model Script : $ALLS_SCRIPT"
echo "Output HEF   : $OUT_HEF"

if [[ ! -f "$ONNX_MODEL" ]]; then
  echo "ERROR: Validated ONNX model missing: $ONNX_MODEL"
  exit 1
fi

if [[ ! -f "$CALIB_NPY" ]]; then
  echo "ERROR: Calibration dataset missing: $CALIB_NPY"
  exit 1
fi

# Method 1: Using compile_hef.py (Python API via hailo_sdk_client)
if python3 -c "import hailo_sdk_client" >/dev/null 2>&1; then
  echo "[OK] Found hailo_sdk_client in active Python environment."
  python3 "$ROOT_DIR/conversion/compile_hef.py" \
    --onnx "$ONNX_MODEL" \
    --calib "$CALIB_NPY" \
    --alls "$ALLS_SCRIPT" \
    --out "$OUT_HEF" \
    --hw-arch hailo8
  echo "[SUCCESS] Generated $OUT_HEF"
  exit 0
fi

# Method 2: Using native Hailo CLI tools (hailo parser / optimize / compiler)
if command -v hailo >/dev/null 2>&1; then
  echo "[OK] Found 'hailo' CLI command in PATH."
  mkdir -p "$ROOT_DIR/models/npu"
  HAR_PARSED="$ROOT_DIR/models/npu/${NET_NAME}_parsed.har"
  HAR_QUANT="$ROOT_DIR/models/npu/${NET_NAME}_quantized.har"

  echo "Step 1/3: Parsing ONNX -> HAR ..."
  hailo parser onnx "$ONNX_MODEL" \
    --hw-arch hailo8 \
    --start-node-names images \
    --net-name "$NET_NAME" \
    --output-har-path "$HAR_PARSED"

  echo "Step 2/3: Quantizing & Optimizing with 256 IDD Calibration Images ..."
  hailo optimize "$HAR_PARSED" \
    --calib-set-path "$CALIB_NPY" \
    --model-script "$ALLS_SCRIPT" \
    --output-har-path "$HAR_QUANT"

  echo "Step 3/3: Compiling to Hailo-8 HEF ..."
  hailo compiler "$HAR_QUANT" \
    --hw-arch hailo8 \
    --output-dir "$ROOT_DIR/models/npu"

  # Rename output to custom_model.hef if necessary
  if [[ -f "$ROOT_DIR/models/npu/${NET_NAME}.hef" ]]; then
    mv "$ROOT_DIR/models/npu/${NET_NAME}.hef" "$OUT_HEF"
  fi

  echo "[SUCCESS] Generated $OUT_HEF"
  exit 0
fi

echo "=================================================================="
echo "ERROR: Neither 'hailo_sdk_client' Python package nor 'hailo' CLI found."
echo ""
echo "To compile on an x86_64 Linux machine:"
echo "1. Download Hailo Dataflow Compiler from https://hailo.ai/developer-zone/software-downloads/"
echo "   File: hailo_dataflow_compiler-3.29.0-py3-none-linux_x86_64.whl"
echo "2. Install: pip install hailo_dataflow_compiler-*.whl"
echo "3. Run: ./conversion/compile_hef.sh"
echo "=================================================================="
exit 1
