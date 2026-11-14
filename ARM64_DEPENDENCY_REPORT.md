# SafeVision Edge — ARM64 Dependency & Runtime Report
**Target System:** Raspberry Pi 5 (ARM64 / aarch64 Linux)  
**OS:** Raspberry Pi OS 64-bit (Debian Bookworm) / Ubuntu 24.04/26.04 LTS  
**Python Version:** Python 3.11.x (Bookworm default) or 3.12.x  

---

## 1. ExecuTorch Runtime Requirements & PyTorch Necessity Audit

### Runtime Architecture
SafeVision Edge uses **ExecuTorch 1.6.0** with the **XNNPACK** backend to run the INT8 PT2E model `models/custom_model.pte`.

```text
FastAPI / App Backend
       │
       ▼
   infer.py (EdgeDetector)
       │
       ▼
executorch.runtime (Runtime.load_program -> load_method("forward"))
       │
       ▼
_portable_lib.so (C++ pybind11 extension)
       │
       ▼
torch::Tensor memory buffer (ATen pybind type caster)
       │
       ▼
ExecuTorch Core Runtime + XnnpackBackend (CPU ARM NEON)
```

### Empirical Test on PyTorch Necessity
An empirical experiment was executed to determine whether ExecuTorch in Python can execute using raw NumPy arrays without PyTorch:

```python
from executorch.runtime import Runtime, Verification
import numpy as np

runtime = Runtime.get()
program = runtime.load_program("models/custom_model.pte", verification=Verification.Minimal)
forward = program.load_method("forward")

dummy_np = np.zeros((1, 3, 640, 640), dtype=np.float32)
forward.execute([dummy_np])
```

**Result:**
```text
RuntimeError: Unsupported python type <class 'numpy.ndarray'>. Ensure that inputs are passed as a flat list of tensors.
```

**Conclusion:**
ExecuTorch's Python pybind extension (`_portable_lib`) binds directly to PyTorch's C++ tensor structure (`torch::Tensor`). In Python, input buffers must be created using `torch.from_numpy(...)`, and output buffers are returned as `torch.Tensor`. Thus, **PyTorch is genuinely required at runtime by ExecuTorch's Python API.**

---

## 2. Why PyPI Triggered NVIDIA CUDA Downloads on Raspberry Pi

### The PyPI Packaging Issue
When installing from standard PyPI (`https://pypi.org/simple`), modern Linux wheels for `torch` publish unified dependency metadata that includes GPU packages:

```text
# PyPI torch 2.15.0 metadata (https://pypi.org/pypi/torch/2.15.0/json):
Requires-Dist:
  cuda-toolkit[cublas,cudart,cufft,cufile,cupti,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.2.2; platform_system == "Linux"
  cuda-bindings<14,>=13.0.3; platform_system == "Linux" and python_version < "3.15"
  nvidia-cudnn-cu13==9.26.0.51; platform_system == "Linux"
  nvidia-nccl-cu13==2.30.7; platform_system == "Linux"
  triton~=3.9.0; platform_system == "Linux" and python_version < "3.16"
```

Because `platform_system == "Linux"` evaluates to `True` on the Raspberry Pi 5, standard `pip install torch` attempts to download the NVIDIA CUDA runtime libraries and CUDA toolkit, resulting in hundreds of megabytes of unnecessary GPU packages on a CPU-only ARM system.

---

## 3. The Corrected ARM64 CPU-Only Dependency Solution

### PyTorch Dedicated CPU Wheel Index
PyTorch hosts pure CPU-only builds on its dedicated wheel index:
**`https://download.pytorch.org/whl/cpu`**

These wheels:
1. Are compiled specifically for CPU execution (`+cpu` tag).
2. Do **not** declare any `nvidia-*` or `cuda-*` dependencies in their metadata.
3. Include standard `aarch64` (`manylinux_2_28_aarch64`) builds compatible with Raspberry Pi 5.

### Pinned Dependency Specification (`requirements.txt`)
```text
--extra-index-url https://download.pytorch.org/whl/cpu

numpy==2.3.5
opencv-python-headless==5.0.0.93
torch==2.5.1
executorch==1.6.0
psutil==7.2.2
fastapi==0.162.5
pydantic==2.14.3
uvicorn[standard]==0.54.0
python-multipart==0.0.33
httpx==0.28.1
```

### Installation Command for Clean Raspberry Pi Environments
```bash
# In an activated virtual environment with system-site-packages:
pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

---

## 4. Reusing Pre-Existing `.venv-cpu-test`

If your Raspberry Pi already has the validated `.venv-cpu-test` environment with working ExecuTorch, you do not need to reinstall. The startup scripts (`scripts/start_safevision.sh` and `scripts/diagnose.sh`) automatically detect:
1. `~/SafeVision_PI_Deployment/.venv`
2. `~/SafeVision_PI_Deployment/.venv-cpu-test`
3. `~/.venv-cpu-test`

If `.venv-cpu-test` is detected, it will be used automatically without triggering any pip installs.
