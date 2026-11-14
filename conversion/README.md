# SafeVision Edge — Hailo-8 Model Conversion Pipeline

This folder contains the standalone, offline conversion workflow to compile the fine-tuned **IDD YOLOv11-S** checkpoint (`outputs/training/IDD_YOLOv11S/weights/best.pt`) into a **Hailo-8** `.hef` binary (`models/custom_model.hef`) for the Raspberry Pi AI HAT+ (26 TOPS).

> **Important:** The existing CPU model (`models/custom_model.pte`, ExecuTorch INT8 PT2E + XNNPACK) **cannot** run directly on Hailo-8 and is preserved untouched.

---

## Architecture & Host Split

| Stage | Target Environment | Tooling |
| --- | --- | --- |
| 1. `best.pt` → `custom_model.onnx` | x86_64 Workstation (offline dev env) | PyTorch + Ultralytics + ONNX (`opset=12`, static `[1,3,640,640]`) |
| 2. Calibration dataset prep | x86_64 Workstation | `prepare_calib.py` → `(N, 640, 640, 3)` `uint8` NHWC `.npy` |
| 3. ONNX → `.hef` compilation | x86_64 Ubuntu Linux (22.04/24.04) | **Hailo Dataflow Compiler** (`hailo_sdk_client`, `hw_arch="hailo8"`) |
| 4. `.hef` deployment & validation | Raspberry Pi 5 (Ubuntu 26.04 / Pi OS `aarch64`) | **HailoRT** + PCIe driver (`hailo_pci`) + `hailo_platform` Python wheel |

---

## Step-by-Step Commands

### 1. Export `best.pt` to Validated Static ONNX
```bash
python conversion/export_onnx.py \
  --weights outputs/training/IDD_YOLOv11S/weights/best.pt \
  --out models/custom_model.onnx \
  --imgsz 640 --opset 12
```
Verify output shape is `[1, 11, 8400]` (4 box coords + 7 IDD classes: `person`, `bicycle`, `motorcycle`, `auto_rickshaw`, `car`, `bus`, `truck`).

### 2. Prepare NHWC uint8 Calibration Dataset
From an image directory:
```bash
python conversion/prepare_calib.py \
  --source path/to/idd_calibration_images \
  --out conversion/calib_nhwc_uint8.npy \
  --max-samples 256
```
Or from an existing NCHW float32 `.npy` calibration tensor (e.g. `calib_set_1024x640x640.npy`):
```bash
python conversion/prepare_calib.py \
  --source calib_set_1024x640x640.npy \
  --out conversion/calib_nhwc_uint8.npy \
  --max-samples 256
```

### 3. Compile ONNX → Hailo-8 `.hef` (x86_64 Linux with Hailo DFC)
Check DFC availability:
```bash
python conversion/compile_hef.py --check-only
```
Compile for Hailo-8 (`26 TOPS`, AI HAT+):
```bash
python conversion/compile_hef.py \
  --onnx models/custom_model.onnx \
  --calib conversion/calib_nhwc_uint8.npy \
  --out models/custom_model.hef \
  --hw-arch hailo8
```
This writes `models/custom_model.hef` and `models/HEF_PROVENANCE.json` (with SHA-256 checksum).

> **YOLOv11-S Operator Compatibility Note:** If Hailo DFC parser flags DFL/Softmax/Concat nodes at the YOLOv11 Detect head, inspect the ONNX end nodes before `Detect` post-processing (or pass `--end-nodes` and add Hailo's `nms_postprocess` command in an `.alls` model script). `src/npu/hailo_decoder.py` supports both raw `[1, 11, 8400]` outputs and Hailo on-chip `HailoNMSByClass` outputs automatically.

### 4. Validate on Raspberry Pi 5 + Hailo-8
Copy `models/custom_model.hef` to `models/custom_model.hef` on the Pi and run:
```bash
python conversion/validate_hef.py --pte models/custom_model.pte --hef models/custom_model.hef
```
