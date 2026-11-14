# SafeVision Edge — Raspberry Pi 5 + Hailo-8 AI HAT+ Deployment Guide

Welcome to the **SafeVision Edge** offline deployment package for **Raspberry Pi 5** equipped with the **Raspberry Pi AI HAT+ (Hailo-8, 26 TOPS)**.

---

## 1. Target Hardware & Runtime Environment

- **Board**: Raspberry Pi 5 Model B (Rev 1.1)
- **Architecture**: ARM64 / aarch64
- **Operating System**: Ubuntu 26.04 LTS / Raspberry Pi OS (Bookworm 64-bit)
- **AI Accelerator**: Raspberry Pi AI HAT+ (Hailo-8, 26 TOPS, PCI ID `1e60:2864`, `/dev/hailo0`)
- **Hailo Driver & Firmware**: Version 4.24.0
- **HailoRT CLI & Python**: Version 4.24.0 in `~/hailo-venv`
- **CPU Inference Engine**: ExecuTorch 1.6.0 with XNNPACK Backend (INT8 PT2E)

---

## 2. Quick Start: 3 Steps to Launch

### Step 1: Extract Deployment Archive
```bash
unzip SafeVision_PI_Deployment_FINAL.zip -d ~/SafeVision_PI_Deployment
cd ~/SafeVision_PI_Deployment
chmod +x scripts/*.sh
```

### Step 2: Run Deployment Setup & Diagnostics
```bash
# Automatically detects ~/hailo-venv, checks hardware, and installs CPU-only deps
./scripts/deploy_pi.sh

# Run full hardware, PCIe driver, and model diagnostics
./scripts/diagnose.sh

# Verify CPU (.pte) and Hailo NPU (.hef) model integrity
./scripts/verify_models.sh
```

### Step 3: Start SafeVision Edge Server
```bash
./scripts/start_safevision.sh
```
*To run in the foreground with live console logs:*
```bash
./scripts/start_safevision.sh --foreground
```

---

## 3. Accessing the Dashboard

Once started, the backend binds to `0.0.0.0:8000`. You can open the dashboard:
- **On the Raspberry Pi**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **From another PC / Phone on the same Wi-Fi/LAN**: `http://<PI_IP_ADDRESS>:8000/`
  *(Find your IP by running `hostname -I`)*

---

## 4. Inference Engine Modes

SafeVision Edge strictly provides three independent operational modes:

| Mode | Engine Backend | Model Used | Behavior |
| :--- | :--- | :--- | :--- |
| **CPU** | ExecuTorch + XNNPACK INT8 | `models/custom_model.pte` | High-accuracy offline CPU inference on Pi 5 Cortex-A76 cores (~140–160 ms). |
| **NPU** | HailoRT 4.24.0 PCIe | `models/npu/custom_model.hef` | 26 TOPS acceleration on Hailo-8 chip. Never silently falls back to CPU if unavailable. |
| **COMPARE** | CPU + Hailo-8 Concurrent | Both models | Executes both engines on identical frames, reporting inference latency, FPS, and bounding-box IoU agreement. |

---

## 5. Model Verification & Ground Truth Hashes

| Model | Format | Expected SHA-256 Hash |
| :--- | :--- | :--- |
| `models/custom_model.pte` | ExecuTorch INT8 | `e9c932aa70edbe3d8065612b576aaa25b4cd86840a3017864461a436d5959b56` |
| `models/npu/custom_model.hef` | Hailo-8 HEF (7 classes) | Verified via `hailortcli parse-hef` |

### Required 7 Traffic Classes:
1. `person`
2. `bicycle`
3. `motorcycle`
4. `auto_rickshaw`
5. `car`
6. `bus`
7. `truck`

> **Notice:** The generic COCO 80-class HEF has been completely removed from this deployment package to eliminate dead weight. Only authentic 7-class models containing `auto_rickshaw` and custom IDD classes are supported.

---

## 6. Standalone Hardware Test Command

To run a direct physical inference test on the Hailo-8 hardware without launching the web server:
```bash
~/hailo-venv/bin/python scripts/test_hailo_inference.py --hef models/npu/custom_model.hef
```

---

## 7. Stopping the Service

```bash
./scripts/stop_safevision.sh
```
