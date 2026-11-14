# SafeVision Edge — Raspberry Pi Deployment Audit Report
**Date:** 2026-11-13  
**Status:** AUDITED & CORRECTED — READY FOR RASPBERRY PI 5 DEPLOYMENT  
**Target Board:** Raspberry Pi 5 Model B Rev 1.1 (ARM64 / aarch64)  
**Accelerator:** Raspberry Pi AI HAT+ 26 TOPS (Hailo-8)  
**CPU Engine:** ExecuTorch 1.6.0 + XNNPACK Backend (INT8 PT2E)  

---

## 1. Executive Summary of Audit Findings

| Audit Item | Initial Status | Issue Identified | Correction Implemented | Status |
| :--- | :--- | :--- | :--- | :--- |
| **A. PyTorch / CUDA Bloat on Pi** | ❌ Failed on Pi | `pip install -r requirements.txt` pulled hundreds of MBs of NVIDIA CUDA wheels (`nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12`, etc.) on a CPU-only ARM64 Pi. | Added `--extra-index-url https://download.pytorch.org/whl/cpu` to `requirements.txt` and `install_pi.sh`. Added automatic detection of existing `.venv-cpu-test`. Verified pure CPU wheel installation. | **RESOLVED** |
| **B. Hailo Model Provenance** | ❌ Unapproved | Bundled `custom_model.hef` was a generic COCO 80-class model from the public Hailo Model Zoo; lacked the fine-tuned 7 IDD classes (missing `auto_rickshaw`). | Relabeled generic model as `generic_coco_yolov11s.hef.unapproved`. Formally declared **NOT APPROVED** in `HEF_PROVENANCE.json`. Implemented strict validation in `hailo_status.py` so unapproved models are blocked with `NPU_MODEL_NOT_READY`. Documented exact 7-class DFC conversion flow. | **RESOLVED** |
| **C. Package Cleanliness** | ⚠️ Bloated (38 MB) | Release zip contained ~12.4 MB of previous inspection videos (`WhatsApp Video ... mp4`), test frames, and local JSON/CSV outputs. | Purged all test runs and video uploads from `outputs/inspections/`, `outputs/zone_previews/`, and `outputs/benchmark/`. Restored clean `.gitkeep` placeholders. | **RESOLVED** |
| **D. Model File Integrity** | ✅ Clean | SHA-256 matches fine-tuned YOLOv11-S export. | SHA-256 confirmed: `E9C932AA70EDBE3D8065612B576AAA25B4CD86840A3017864461A436D5959B56`. No weights were touched. | **VERIFIED** |
| **E. Automated Test Suite** | ✅ Clean | 57 automated unit, integration, and safety pulse tests. | Ran test suite: **57/57 tests PASS**. | **VERIFIED** |

---

## 2. Issue A: CPU Dependency Correction & CUDA Elimination

### Root Cause Analysis
- On PyPI (`https://pypi.org/pypi/torch/json`), standard Linux `manylinux_2_28_aarch64` releases of modern PyTorch declare `cuda-toolkit` and `nvidia-*` packages in their `Requires-Dist` for `platform_system == "Linux"`.
- When `pip install torch` or `pip install torch==2.15.0` was executed on Linux ARM64 without an explicit CPU index, pip resolved these GPU metadata dependencies and pulled hundreds of megabytes of NVIDIA CUDA packages onto the Raspberry Pi's SD card, even though the Pi 5 has no NVIDIA hardware.

### Runtime Necessity Audit
- ExecuTorch runtime in Python (`executorch.runtime.Runtime`) interacts with models via a C++ pybind extension (`_portable_lib`).
- When passing inputs to `Method.execute([tensor])`, passing a `numpy.ndarray` raises:
  ```text
  RuntimeError: Unsupported python type <class 'numpy.ndarray'>. Ensure that inputs are passed as a flat list of tensors.
  ```
- Therefore, `torch.Tensor` is strictly required by ExecuTorch's Python API to transfer memory buffers to the ExecuTorch runtime.
- **Solution:** Direct pip to PyTorch's official CPU wheel index (`https://download.pytorch.org/whl/cpu`). On this index, PyTorch provides pure CPU wheels for `aarch64` (`torch-2.5.1`, `torch-2.4.1`) with zero NVIDIA or CUDA dependencies.
- Added `--extra-index-url https://download.pytorch.org/whl/cpu` to `requirements.txt` and `scripts/install_pi.sh`.
- Added automatic detection for the user's pre-existing `.venv-cpu-test` environment in `scripts/start_safevision.sh` and `scripts/diagnose.sh`.

---

## 3. Issue B: Hailo Model Validation & 7-Class Equivalence

### Discrepancy Identified
A technical comparison was conducted between the approved CPU model and the generic Hailo model:

| Property | Approved CPU Model (`custom_model.pte`) | Generic Hailo Model (`generic_coco_yolov11s.hef.unapproved`) | Status |
| :--- | :--- | :--- | :--- |
| **Architecture** | YOLOv11-S (INT8 PT2E XNNPACK) | YOLOv11-S (Hailo-8 DFC) | Compatible base |
| **Input Shape** | `[1, 3, 640, 640]` float32 | `[1, 640, 640, 3]` uint8 NHWC | Compatible input |
| **Dataset** | IDD (India Driving Dataset) Fine-Tuned | COCO (Common Objects in Context) | **MISMATCH** |
| **Number of Classes** | **7 classes** | **80 classes** | **MISMATCH** |
| **Class 0** | `person` | `person` | Equivalent |
| **Class 1** | `bicycle` | `bicycle` | Equivalent |
| **Class 2** | `motorcycle` | `car` | **CONFLICT** (car vs motorcycle) |
| **Class 3** | `auto_rickshaw` | `motorcycle` | **CRITICAL CONFLICT** (`auto_rickshaw` missing from COCO) |
| **Class 4** | `car` | `airplane` | **CONFLICT** |
| **Class 5** | `bus` | `bus` | Equivalent |
| **Class 6** | `truck` | `train` | **CONFLICT** |

### Actions Taken
1. **De-approval & Renaming:** The generic binary was removed from `models/custom_model.hef` and saved as `models/generic_coco_yolov11s.hef.unapproved`.
2. **Provenance Guard:** `models/HEF_PROVENANCE.json` was updated with `"status": "NOT_APPROVED_GENERIC_MODEL"` and `"approved_for_safevision": false`.
3. **Engine Guard:** `src/npu/hailo_status.py` and `src/engines/hailo_engine.py` now parse `HEF_PROVENANCE.json`. If a model is not explicitly approved, the NPU reports `NPU_MODEL_NOT_READY` and refuses to initialize.
4. **Documentation:** Created `HEF_MODEL_COMPATIBILITY_REPORT.md` detailing the exact conversion commands from `outputs/training/IDD_YOLOv11S/weights/best.pt` to a verified 7-class `.hef`.

---

## 4. Package Cleanliness Audit

- Purged all previous inspection directories from `SafeVision_PI_Deployment/outputs/inspections/`, including user test video `WhatsApp Video 2026-11-12 at 5.33.46 PM.mp4`.
- Purged all temporary zone previews from `outputs/zone_previews/`.
- Purged temporary benchmark results from `outputs/benchmark/`.
- Clean `.gitkeep` files placed in all output directories.
- Package size reduced from **38 MB** to **18.6 MB** (clean release footprint).

---

## 5. Test Execution Results

Ran complete test discovery on `SafeVision_PI_Deployment/tests`:
```text
Ran 57 tests in 13.265s
OK
```

- `test_01_health_and_configs`: **PASS**
- `test_02_upload_validation`: **PASS**
- `test_03_single_and_multi_image_real_pte_workflow`: **PASS**
- `test_04_video_real_pte_workflow`: **PASS**
- `test_01_model_file_and_sha256_integrity`: **PASS**
- `test_02_no_external_paths_windows_paths_or_cloud_urls`: **PASS**
- `test_03_shell_scripts_use_unix_lf_line_endings`: **PASS**
- `test_04_offline_spa_and_bundled_fonts_served_by_fastapi`: **PASS**
- `test_05_minimal_clean_package_footprint`: **PASS**
- `test_06_requirements_pinned_and_xnnpack_active`: **PASS**
- All 18 Safety Intelligence rule tests: **PASS**
- All 10 Zone Occupancy accuracy and calibration tests: **PASS**
- All 6 NPU adapter & engine management tests: **PASS**

---

## 6. Model SHA-256 Integrity Verification

```text
Algorithm: SHA256
File: models/custom_model.pte
Hash: E9C932AA70EDBE3D8065612B576AAA25B4CD86840A3017864461A436D5959B56
Status: UNCHANGED (Verified against original fine-tuned INT8 export)
```
