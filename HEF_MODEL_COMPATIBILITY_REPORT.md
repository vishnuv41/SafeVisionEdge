# SafeVision Edge — Hailo HEF Model Compatibility Report
**Date:** 2026-11-13  
**Status:** AUDITED — GENERIC MODEL REJECTED; CONVERSION PROCEDURE DEFINED  
**Target Hardware:** Raspberry Pi AI HAT+ (Hailo-8, 26 TOPS)  
**Required Model Format:** Compiled `.hef` (Hailo Executable Format)  

---

## 1. Technical Audit of Bundled HEF vs Ground Truth PTE

### Provenance Comparison

| Characteristic | Fine-Tuned Ground Truth (`custom_model.pte`) | Bundled Generic Binary (`generic_coco_yolov11s.hef.unapproved`) |
| :--- | :--- | :--- |
| **Model File** | `models/custom_model.pte` | `models/generic_coco_yolov11s.hef.unapproved` |
| **File Size** | 12,265,372 bytes (~11.70 MiB) | 19,577,758 bytes (~18.67 MiB) |
| **SHA-256 Hash** | `e9c932aa70edbe3d8065612b576aaa25b4cd86840a3017864461a436d5959b56` | `8a500ccc9e2797295dc0cb0c41cd69ae1d728f2190dc51fdfdf8345a4f7fa036` |
| **Provenance Source** | Fine-tuned on IDD (India Driving Dataset) | Pre-compiled Hailo Model Zoo release (`v2.17.0`) |
| **Base Architecture** | YOLOv11-S | YOLOv11-S |
| **Input Shape** | `[1, 3, 640, 640]` float32 (NCHW) | `[1, 640, 640, 3]` uint8 (NHWC) |
| **Output Classes** | **7 classes** | **80 classes** (COCO dataset) |
| **Class 0** | `person` | `person` |
| **Class 1** | `bicycle` | `bicycle` |
| **Class 2** | `motorcycle` | `car` *(Conflicted)* |
| **Class 3** | `auto_rickshaw` | `motorcycle` *(Conflicted)* |
| **Class 4** | `car` | `airplane` *(Conflicted)* |
| **Class 5** | `bus` | `bus` *(Matches)* |
| **Class 6** | `truck` | `train` *(Conflicted)* |
| **SafeVision Status** | **APPROVED & VERIFIED** | **NOT APPROVED — STRICTLY REJECTED** |

---

## 2. Rejection Decision & Justification

### Why the Generic Binary Cannot Be Used
1. **Critical Class Omission:** Standard COCO has no `auto_rickshaw` category. In the fine-tuned SafeVision model, `auto_rickshaw` is a distinct class crucial for Indian traffic analysis and Safety Pulse calculations.
2. **Index Transposition:** In COCO, class 2 is `car` and class 3 is `motorcycle`. In SafeVision IDD, class 2 is `motorcycle` and class 3 is `auto_rickshaw`. Using the generic model would cause cars to be registered as motorcycles, and motorcycles as auto-rickshaws, completely corrupting Safety Pulse metrics.
3. **Benchmarking Honesty:** Benchmarking a 7-class model on CPU against an 80-class model on NPU violates experimental validity.

### Enforcement in Code
- The file has been renamed to:
  `models/generic_coco_yolov11s.hef.unapproved`
- `models/HEF_PROVENANCE.json` records `"approved_for_safevision": false`.
- `src/npu/hailo_status.py` inspects `HEF_PROVENANCE.json`. If `approved_for_safevision` is `false` or `custom_model.hef` is missing, the system reports `NPU_MODEL_NOT_READY` and refuses to initialize the Hailo NPU engine.

---

## 3. Step-by-Step Conversion Pipeline for the Fine-Tuned 7-Class Model

To compile the genuine fine-tuned SafeVision YOLOv11-S model into an approved `models/custom_model.hef`, follow this pipeline on an x86_64 Linux machine equipped with the **Hailo Dataflow Compiler (DFC)**.

### Step 1: Export Fine-Tuned Checkpoint to Static ONNX
Using `conversion/export_onnx.py`:
```bash
python conversion/export_onnx.py \
  --weights outputs/training/IDD_YOLOv11S/weights/best.pt \
  --out models/custom_model.onnx \
  --imgsz 640 \
  --opset 12
```
*Validation:* Verify output shape is static `[1, 11, 8400]` (4 box coordinates + 7 class scores).

### Step 2: Prepare Calibration Dataset (NHWC uint8)
Using `conversion/prepare_calib.py`:
```bash
python conversion/prepare_calib.py \
  --source path/to/idd_calibration_images \
  --out conversion/calib_nhwc_uint8.npy \
  --max-samples 256
```

### Step 3: Compile ONNX to HEF Using Hailo DFC
Using `conversion/compile_hef.py`:
```bash
python conversion/compile_hef.py \
  --onnx models/custom_model.onnx \
  --calib conversion/calib_nhwc_uint8.npy \
  --out models/custom_model.hef \
  --hw-arch hailo8
```
This produces `models/custom_model.hef` compiled specifically for Hailo-8 (26 TOPS) with the exact 7 classes.

### Step 4: Deploy and Validate on Raspberry Pi 5
1. Copy `models/custom_model.hef` to the Raspberry Pi:
   ```bash
   scp models/custom_model.hef pi@<pi-ip>:~/SafeVision_PI_Deployment/models/
   ```
2. Update `models/HEF_PROVENANCE.json` with `"approved_for_safevision": true`.
3. Validate parity between CPU and NPU:
   ```bash
   python conversion/validate_hef.py \
     --pte models/custom_model.pte \
     --hef models/custom_model.hef
   ```

When both models are loaded with identical 7-class weights, the UI enables both **NPU** and **Compare Both** modes with mathematically honest detection agreement.
