# SafeVision Edge — Offline Raspberry Pi 5 Deployment Package

Self-contained offline deployment bundle for **SafeVision Edge — Offline Junction Safety Intelligence** running custom fine-tuned **YOLOv11-S INT8 (`custom_model.pte`)** via **ExecuTorch + XNNPACK** with optional **Hailo-8 NPU accelerator** integration, behind FastAPI + bundled React dashboard.

> **Audit Status:** This package has been audited against CPU dependency bloat (NVIDIA CUDA packages strictly eliminated) and Hailo model provenance (unapproved generic models blocked). Complete reports:
> - [`PI_DEPLOYMENT_AUDIT.md`](PI_DEPLOYMENT_AUDIT.md) — Comprehensive deployment audit & test results.
> - [`ARM64_DEPENDENCY_REPORT.md`](ARM64_DEPENDENCY_REPORT.md) — ExecuTorch runtime dependency analysis & CPU-only installation.
> - [`HEF_MODEL_COMPATIBILITY_REPORT.md`](HEF_MODEL_COMPATIBILITY_REPORT.md) — Hailo-8 model validation & 7-class conversion pipeline.

---

## 1. Included Files & Folders

```text
SafeVision_PI_Deployment/
├── PI_DEPLOYMENT_AUDIT.md          # Comprehensive deployment audit findings & resolutions
├── ARM64_DEPENDENCY_REPORT.md      # CPU-only PyTorch/ExecuTorch ARM64 dependency report
├── HEF_MODEL_COMPATIBILITY_REPORT.md # Hailo-8 model compatibility & 7-class conversion procedure
├── DEPENDENCY_COMPATIBILITY_REPORT.md # Pinned ARM64 wheel matrix
├── README.md                       # Transfer, install, start, stop & troubleshooting instructions
├── requirements.txt                # Pinned runtime deps with CPU-only index (--extra-index-url)
├── config/                         # Runtime configuration JSONs
│   ├── class_names.json            # 7 trained classes (0..6, incl. 3: auto_rickshaw)
│   ├── preprocessing_config.json   # Static 1x3x640x640 RGB float32 + (1,11,8400) decoder contract
│   ├── safety_config.json          # Density thresholds, VRU share rule, pulse bands, blur default
│   └── zones.json                  # Configurable normalized polygons (road, crossing, footpath)
├── models/
│   ├── custom_model.pte            # YOLOv11-S INT8 PT2E + XNNPACK ExecuTorch model (11.70 MiB)
│   ├── MODEL_PROVENANCE.json       # SHA-256 checksum, export toolchain, IDD val100 baseline
│   ├── generic_coco_yolov11s.hef.unapproved # Generic COCO model (NOT APPROVED for SafeVision)
│   └── HEF_PROVENANCE.json         # Hailo model provenance & approval guard
├── conversion/                     # Conversion pipeline from best.pt -> ONNX -> Hailo HEF
├── scripts/
│   ├── setup_hailo_pi5.sh          # Turnkey PCIe Gen 3 & hailo-all setup script for Pi 5
│   ├── check_pi_env.py             # Hardware, OS, glibc/ARM64, XNNPACK, SHA-256 & 1-frame check
│   ├── validate_pi_hardware.py     # Multi-frame XNNPACK validation, 7-class box audit & thermal/FPS report
│   ├── install_pi.sh               # Strict ARM64 installer (CPU-only, rejects CUDA, verifies)
│   ├── diagnose.sh                 # Full verification (--full) or 20-frame benchmark (--benchmark)
│   ├── start_safevision.sh         # Starts FastAPI + static SPA server (0.0.0.0:8000)
│   └── stop_safevision.sh          # Gracefully stops background server via PID file
├── src/                            # FastAPI backend, ExecuTorch CPU engine, Hailo NPU adapter
├── static/                         # Production Vite React SPA + local bundled Inter fonts
└── tests/                          # 57 automated unit, integration, and safety pulse tests
```

---

## 2. Deployment Commands

### Windows PowerShell transfer

```powershell
scp -r E:\SafeVision_PI_Deployment pi@raspberrypi.local:~/
```

*(Alternatively, copy `E:\SafeVision_PI_Deployment_FINAL.zip` onto a USB drive or SCP it, unzip with `unzip SafeVision_PI_Deployment_FINAL.zip`, and enter `SafeVision_PI_Deployment`.)*

### Raspberry Pi installation

*(Requires **Raspberry Pi OS 64-bit** (`aarch64`) with Python **3.11–3.14**. Internet is needed only for initial `pip install`; normal operation afterward is 100% offline.)*

```bash
cd ~/SafeVision_PI_Deployment
chmod +x scripts/*.sh
./scripts/install_pi.sh
```

### Full diagnostics

```bash
./scripts/diagnose.sh --full
```

### Raspberry Pi benchmark

```bash
./scripts/diagnose.sh --benchmark
```

### Start application

```bash
./scripts/start_safevision.sh
```

### Dashboard

- On the Raspberry Pi itself: `http://127.0.0.1:8000/`
- **From another device on the same network (LAN/Wi-Fi):** The backend binds to `0.0.0.0:8000` by default, and `./scripts/start_safevision.sh` prints your Pi's LAN URL automatically (or find it with `hostname -I`). Open `http://<raspberry-pi-ip>:8000/` (for example, `http://192.168.1.50:8000/`) in any laptop or phone browser on the same local network without internet.

### Stop application

```bash
./scripts/stop_safevision.sh
```

---

## 3. Stored Outputs & Troubleshooting

- **Inspection Results:** `outputs/inspections/<INSPECTION_ID>/` (`frames/*.jpg`, `pedestrian_heatmap.png`, `*_counts.csv`, `safety_brief.md/.json`, `benchmark.json`)
- **Diagnostics & Hardware Reports:** `logs/diagnose_*.log` and `outputs/reports/pi_validation_*.{json,md}`
- **Server Logs:** `logs/safevision.log`
- **Common Errors:**
  - `32-bit` error during install: Reflash/boot **Raspberry Pi OS 64-bit (`aarch64`)**; ExecuTorch wheels do not support 32-bit `armv7l`.
  - `CRLF` error (`bash\r`): Run `sed -i 's/\r$//' scripts/*.sh`.
  - Port `8000` busy: Run `./scripts/stop_safevision.sh` or set `SAFEVISION_PORT=8080 ./scripts/start_safevision.sh`.
  - Thermal throttling: Attach active cooling and check `vcgencmd measure_temp && vcgencmd get_throttled` or **Reports and settings -> Performance Benchmark**.

---

## 4. Hailo-8 NPU Integration, Comparison Benchmark & Interactive Zone Editor

### Engine Modes (Strictly Three — No Auto Mode)

- **CPU (`cpu`)**: Preserved ExecuTorch + XNNPACK INT8 PT2E engine (`models/custom_model.pte`).
- **NPU (`npu`)**: Independent Hailo-8 HailoRT engine (`models/custom_model.hef`). Reports explicit status (`NPU_HARDWARE_UNAVAILABLE`, `HAILORT_UNAVAILABLE`, `NPU_MODEL_NOT_READY`) without silently falling back to CPU.
- **Compare Both (`compare`)**: Runs CPU and Hailo-8 NPU side-by-side on identical frames during inspection or benchmark runs.

### Hailo-8 Model Conversion

Because `.pte` cannot execute directly on Hailo-8, compile the fine-tuned YOLOv11-S checkpoint offline into `models/custom_model.hef` using the pipeline in [conversion/README.md](conversion/README.md) (`export_onnx.py` -> `prepare_calib.py` -> `compile_hef.py` -> `validate_hef.py`).

### CPU vs NPU Comparison Benchmark & Telemetry

- Open the **NPU & Benchmark** page in the web UI (or call `POST /api/v1/benchmark/compare`, `GET /api/v1/hardware/status`, `GET /api/v1/benchmark/results`, `GET /api/v1/benchmark/results/csv`).
- Excludes warm-up runs from measured timing, interleaves CPU and NPU inference, samples real system RAM/CPU/temperature/throttling telemetry (`Unavailable` when sensors/tools are absent), and exports JSON/CSV reports under `outputs/benchmark/{cpu,npu,comparison}/`.

### Interactive Polygon Zone Editor

- Open **Zone editor** in the navigation bar (or click **Open Interactive Zone Editor** in Inspect / Reports and settings).
- Upload an image or video (extracts a representative middle frame) or load an existing inspection frame, then draw polygons, drag vertices, add edge midpoint vertices, move/resize polygons, assign zone types (`road`, `footpath`, `crossing`), undo/redo, preview optional heuristic suggestions, and save scene-specific zone profiles (`config/zone_scenes/<scene_id>.json`).
