# SafeVision Edge — Hailo-8 DFC Model Compilation Package

This package contains all assets required to compile the fine-tuned 7-class SafeVision **IDD YOLOv11-S** model to a Hailo-8 `.hef` binary for the Raspberry Pi AI HAT+ (26 TOPS).

---

## 1. System Requirements

- **Operating System**: Linux x86_64 (Ubuntu 22.04 LTS or Ubuntu 24.04 LTS)
- **Architecture**: x86_64 (Intel / AMD 64-bit) — *DFC cannot run on ARM64 Raspberry Pi*
- **Python**: 3.10 or 3.11
- **Toolchain**: Hailo Dataflow Compiler (DFC) v3.28 / v3.29 / v3.30

---

## 2. Package Contents

| File | Purpose | Size | SHA-256 |
| :--- | :--- | :--- | :--- |
| `models/checkpoints/IDD_YOLOv11S_best.pt` | Original fine-tuned 7-class PyTorch checkpoint | 18.28 MB | `7cde73595172e8b6a71d0071c473d869b7cc036bf835ef03d25303ff8a8791ff` |
| `models/npu/custom_model.onnx` | Validated static ONNX export (Opset 12, shape `[1, 11, 8400]`) | 36.18 MB | `5831eb58b2d5a30232653db5e6d50b9ce23a32aec416e9acee5551044ef327bd` |
| `conversion/calib_nhwc_uint8.npy` | 256-sample IDD road scene calibration dataset (`uint8` NHWC) | 300.0 MB | N/A |
| `conversion/yolov11s_hailo.alls` | Hailo model script (normalization & quantization flavor) | 1 KB | N/A |
| `conversion/compile_hef.py` | Automated compilation script (Python API via `hailo_sdk_client`) | 5 KB | N/A |
| `conversion/compile_hef.sh` | Bash launcher (supports CLI `hailo` and Python `ClientRunner`) | 3 KB | N/A |

---

## 3. Step-by-Step Compilation Commands

### Option A: Direct Python Execution (on Ubuntu x86_64)
```bash
# 1. Install Hailo DFC wheel from Hailo Developer Zone
pip install hailo_dataflow_compiler-3.29.0-py3-none-linux_x86_64.whl

# 2. Run compilation
python3 conversion/compile_hef.py \
  --onnx models/npu/custom_model.onnx \
  --calib conversion/calib_nhwc_uint8.npy \
  --alls conversion/yolov11s_hailo.alls \
  --out models/npu/custom_model.hef \
  --hw-arch hailo8
```

### Option B: Using Hailo Official Docker
```bash
docker run --rm -v $(pwd):/workspace -w /workspace hailo_dataflow_compiler:v3.29.0 \
  python3 conversion/compile_hef.py \
    --onnx models/npu/custom_model.onnx \
    --calib conversion/calib_nhwc_uint8.npy \
    --alls conversion/yolov11s_hailo.alls \
    --out models/npu/custom_model.hef \
    --hw-arch hailo8
```

---

## 4. Deploying Output to Raspberry Pi

Once compilation completes, copy the generated binary:
```bash
scp models/npu/custom_model.hef pi@<PI_IP>:~/SafeVision_PI_Deployment/models/npu/custom_model.hef
```
On the Raspberry Pi 5, verify with:
```bash
hailortcli parse-hef ~/SafeVision_PI_Deployment/models/npu/custom_model.hef
```
It will report:
- `Architecture: HAILO8`
- `Network: safevision_yolov11s`
- `7 classes` (`person`, `bicycle`, `motorcycle`, `auto_rickshaw`, `car`, `bus`, `truck`)
