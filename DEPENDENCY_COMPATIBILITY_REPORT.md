# SafeVision Edge — Dependency & ARM64 Compatibility Report

Verified for Raspberry Pi 5 Model B + Hailo-8 AI HAT+ running Ubuntu 26.04 LTS ARM64 (Python 3.14.4, HailoRT 4.24.0, kernel 7.0.0-1023-raspi).

---

## 1. Dependency Strategy & `~/hailo-venv` Protection

The Raspberry Pi 5 runs a working Hailo-8 setup inside `~/hailo-venv` where `hailo_platform` (HailoRT 4.24.0) is installed and operational. 

Because `hailo_platform` is a compiled C/C++ Python extension linked against numpy's C-ABI:
- **Never force-reinstall or downgrade `numpy`** inside `~/hailo-venv`. Doing so can break the C-ABI and render `hailo_platform` unimportable.
- **Avoid rigid exact pins (`==`)** across requirements files. Rigid pins like `numpy==2.3.5` or `torch==2.5.1` cause pip to attempt uninstallation and source compilation, especially under newer Python versions such as Python 3.14.
- All dependencies use **compatible lower bounds** (`>=`), and installation is mediated by `scripts/check_venv_deps.py`.

---

## 2. Reconciled Dependency Matrix

| Package | Constraint | Role | Linux `aarch64` Status | Runtime Note |
| --- | --- | --- | --- | --- |
| `numpy` | `>=1.24.0` | Numerical backend & box decoding | Pre-installed in `~/hailo-venv` | **PROTECTED**: Preserved as compiled with HailoRT |
| `opencv-python-headless` | `>=4.8.0` | Image/video I/O, letterbox, blur | Precompiled `aarch64` | Pure headless CPU build, zero GUI dependencies |
| `torch` | `>=2.4.0` | CPU tensor wrapper for ExecuTorch | CPU wheel via `--extra-index-url` | Strictly CPU-only; no CUDA packages |
| `executorch` | `>=0.4.0` | INT8 CPU inference with XNNPACK | Precompiled `aarch64` | Executes `custom_model.pte` |
| `psutil` | `>=5.9.0` | CPU, RAM & system telemetry | Precompiled `aarch64` | Reads `/sys/class/thermal` and memory stats |
| `fastapi` | `>=0.100.0` | Offline REST API + SPA host | `py3-none-any` | Pure Python ASGI application |
| `pydantic` | `>=2.0.0` | Schema validation | `py3-none-any` / `pydantic-core` | High performance data validation |
| `uvicorn[standard]` | `>=0.22.0` | ASGI HTTP server | Precompiled `aarch64` | Localhost/LAN web server |
| `python-multipart` | `>=0.0.6` | Image & video file uploads | `py3-none-any` | Handles image/video inference endpoints |
| `httpx` | `>=0.25.0` | Test client & internal HTTP | `py3-none-any` | Required by test suite |

---

## 3. Safe Dependency Installer (`scripts/check_venv_deps.py`)

To ensure `~/hailo-venv` is never disrupted:
1. `check_venv_deps.py` runs before any package is installed.
2. If `hailo_platform` is detected, it flags `numpy` as **PROTECTED**.
3. It checks each required module by importing it.
4. Only truly missing packages are installed using `--no-cache-dir` and the CPU PyTorch index (`--extra-index-url https://download.pytorch.org/whl/cpu`).
5. `install_pi.sh` and `deploy_pi.sh` both invoke `check_venv_deps.py` instead of running blind destructive `pip install -r requirements.txt` or setuptools upgrades.

---

## 4. Hardware and Runtime Verification

- **CPU Model**: `models/custom_model.pte` (SHA-256: `e9c932aa70edbe3d8065612b576aaa25b4cd86840a3017864461a436d5959b56`). Executed by ExecuTorch XNNPACK backend.
- **NPU Model**: `models/npu/custom_model.hef`. Single configured target path. Requires compilation with Hailo Dataflow Compiler (DFC) on an x86_64 Linux system before NPU mode is enabled.
- **Engine Selection**: Strict 3 modes (`cpu`, `npu`, `compare`). `auto` is rejected. No silent fallbacks.
