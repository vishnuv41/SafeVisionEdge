"""Environment, ARM64 compatibility, XNNPACK, and model integrity check for SafeVision_PI_Deployment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

EXPECTED_MODEL_SHA256 = "E9C932AA70EDBE3D8065612B576AAA25B4CD86840A3017864461A436D5959B56"
EXPECTED_CLASSES = [
    "person",
    "bicycle",
    "motorcycle",
    "auto_rickshaw",
    "car",
    "bus",
    "truck",
]

PKG_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PKG_ROOT / "src"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _detect_pi_model() -> str | None:
    dt_model = Path("/proc/device-tree/model")
    if dt_model.is_file():
        try:
            return dt_model.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="ignore").strip()
        except Exception:
            return None
    return None


def _read_pi_thermals() -> tuple[float | None, bool | None]:
    temp_c: float | None = None
    throttled: bool | None = None
    thermal_file = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal_file.is_file():
        try:
            temp_c = round(int(thermal_file.read_text(encoding="utf-8").strip()) / 1000.0, 1)
        except Exception:
            temp_c = None
    if shutil.which("vcgencmd"):
        try:
            out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True, timeout=1.5).strip()
            if "=" in out:
                hex_val = int(out.split("=", 1)[1], 16)
                throttled = (hex_val & 0xF) != 0
        except Exception:
            throttled = None
    return temp_c, throttled


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SafeVision Edge Raspberry Pi 5 runtime.")
    parser.add_argument("--skip-infer", action="store_true", help="Check files and imports without running test inference.")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    arch = platform.machine()
    sys_name = platform.system()
    py_ver = platform.python_version()
    pi_board = _detect_pi_model()
    is_arm64 = arch.lower() in {"aarch64", "arm64"}
    disk = shutil.disk_usage(PKG_ROOT)
    free_gb = disk.free / (1024 ** 3)

    print("=== SafeVision Edge Deployment Diagnostics ===")
    print(f"Package root   : {PKG_ROOT}")
    print(f"OS / Kernel    : {platform.platform()}")
    print(f"Architecture   : {arch} ({sys_name})")
    print(f"Hardware board : {pi_board or 'Non-Raspberry-Pi / host workstation'}")
    print(f"Python version : {py_ver}")
    print(f"Free disk space: {free_gb:.2f} GiB")

    if sys.version_info < (3, 11) or sys.version_info >= (3, 15):
        errors.append(f"Python 3.11-3.14 required by pinned ARM64 wheels (found {py_ver}).")
    if free_gb < 1.0:
        errors.append(f"Insufficient free disk space ({free_gb:.2f} GiB < 1.0 GiB).")
    elif free_gb < 2.5:
        warnings.append(f"Low disk space ({free_gb:.2f} GiB); keep >= 2.5 GiB free for logs and video inspections.")

    if sys_name == "Linux" and not is_arm64:
        warnings.append(f"Target expects 64-bit ARM64/aarch64 Raspberry Pi OS; running on '{arch}'.")
    if sys_name != "Linux":
        warnings.append("Running on non-Linux host; Raspberry Pi hardware compatibility is NOT verified until tested on Pi 5.")

    temp_c, throttled = _read_pi_thermals()
    if temp_c is not None:
        print(f"CPU temperature: {temp_c} C")
    if throttled is not None:
        print(f"Throttled flag : {throttled}")
        if throttled:
            warnings.append("Raspberry Pi reported active or past thermal/under-voltage throttling.")

    required_pkgs = [
        ("numpy", "numpy"),
        ("cv2", "opencv-python-headless"),
        ("torch", "torch"),
        ("executorch", "executorch"),
        ("psutil", "psutil"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("multipart", "python-multipart"),
        ("httpx", "httpx"),
    ]
    print("\n=== Python Dependencies ===")
    cuda_detected: list[str] = []
    for dist_name in ("nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12", "nvidia-cublas-cu12", "cuda-toolkit"):
        if importlib.util.find_spec(dist_name.replace("-", "_")) is not None:
            cuda_detected.append(dist_name)
    if cuda_detected and is_arm64:
        warnings.append(
            f"Unnecessary GPU/CUDA packages detected on ARM64 CPU-only host: {', '.join(cuda_detected)}. "
            "Consider purging them to save SD card space."
        )

    for mod_name, dist_label in required_pkgs:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {dist_label:<24} {ver}")
            if mod_name == "psutil":
                vm = mod.virtual_memory()
                total_gb = vm.total / (1024 ** 3)
                avail_gb = vm.available / (1024 ** 3)
                print(f"       System RAM: {total_gb:.2f} GiB total, {avail_gb:.2f} GiB available")
                if total_gb < 1.5:
                    errors.append(f"System RAM too low ({total_gb:.2f} GiB < 1.5 GiB).")
                elif total_gb < 3.5:
                    warnings.append(f"Detected {total_gb:.2f} GiB RAM; 4 GB or 8 GB Pi 5 recommended.")
        except Exception as exc:
            errors.append(f"Missing dependency '{dist_label}' ({mod_name}): {exc}")
            print(f"  [FAIL] {dist_label:<24} {exc}")

    print("\n=== Configuration & Static SPA ===")
    for cfg_name in ("class_names.json", "preprocessing_config.json", "safety_config.json", "zones.json"):
        p = PKG_ROOT / "config" / cfg_name
        if not p.is_file():
            errors.append(f"Missing config file: {p}")
            print(f"  [FAIL] config/{cfg_name}")
        else:
            print(f"  [OK] config/{cfg_name}")

    cls_path = PKG_ROOT / "config" / "class_names.json"
    if cls_path.is_file():
        cls_map = json.loads(cls_path.read_text(encoding="utf-8"))
        ordered = [cls_map[str(i)] for i in range(len(cls_map))]
        if ordered != EXPECTED_CLASSES:
            errors.append(f"Class mapping mismatch: {ordered} != {EXPECTED_CLASSES}")
        else:
            print(f"  [OK] 7 detection classes verified: {ordered}")

    spa_index = PKG_ROOT / "static" / "index.html"
    if not spa_index.is_file():
        errors.append(f"Frontend build missing: {spa_index}")
        print("  [FAIL] static/index.html")
    else:
        print("  [OK] static/index.html (bundled offline SPA present)")

    print("\n=== Model Integrity & XNNPACK Backend ===")
    model_path = PKG_ROOT / "models" / "custom_model.pte"
    if not model_path.is_file():
        errors.append(f"Missing model file: {model_path}")
        print(f"  [FAIL] {model_path}")
    else:
        actual_sha = _sha256_file(model_path)
        size_mb = model_path.stat().st_size / (1024 ** 2)
        print(f"  File   : {model_path} ({size_mb:.2f} MiB)")
        print(f"  SHA256 : {actual_sha}")
        if actual_sha != EXPECTED_MODEL_SHA256:
            errors.append(f"SHA-256 mismatch! Expected {EXPECTED_MODEL_SHA256}, got {actual_sha}")
            print("  [FAIL] SHA-256 mismatch")
        else:
            print("  [OK] SHA-256 matches fine-tuned YOLOv11-S INT8 export")

    if not errors:
        try:
            from executorch.runtime import Runtime

            rt = Runtime.get()
            backends = list(rt.backend_registry.registered_backend_names)
            print(f"  Registered backends: {backends}")
            if "XnnpackBackend" not in backends:
                errors.append(f"XnnpackBackend missing from ExecuTorch runtime backends: {backends}")
            else:
                print("  [OK] XnnpackBackend available in ExecuTorch runtime")

            prog = rt.load_program(str(model_path))
            fwd = prog.load_method("forward")
            meta = fwd.metadata
            print(f"  Method meta: {meta}")
        except Exception as exc:
            errors.append(f"Failed verifying ExecuTorch XNNPACK backend or model metadata: {exc}")

    if not args.skip_infer and not errors:
        print("\n=== ExecuTorch XNNPACK Smoke Inference (CPU-Only) ===")
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        try:
            import cv2
            from infer import EdgeDetector, load_configs

            class_names, prep_cfg, _ = load_configs(PKG_ROOT / "config")
            t_load0 = time.perf_counter()
            detector = EdgeDetector(model_path, class_names, prep_cfg)
            load_ms = (time.perf_counter() - t_load0) * 1000.0
            print(f"  [OK] Loaded custom_model.pte in {load_ms:.1f} ms")

            fixture = PKG_ROOT / "tests" / "fixtures" / "sample_road_1.jpg"
            frame = cv2.imread(str(fixture))
            if frame is None:
                raise RuntimeError(f"Could not read test fixture: {fixture}")
            t_inf0 = time.perf_counter()
            dets, inf_ms = detector.predict_frame(frame)
            total_ms = (time.perf_counter() - t_inf0) * 1000.0
            print(
                f"  [OK] Inference on {fixture.name}: {len(dets)} detections, "
                f"inference={inf_ms:.1f} ms, total={total_ms:.1f} ms"
            )
            os.environ.setdefault("SAFEVISION_DIAG_OK", "1")
        except Exception as exc:
            errors.append(f"ExecuTorch inference failed: {exc}")
            print(f"  [FAIL] {exc}")

    print("\n=== Hailo-8 NPU Readiness (Optional Accelerator) ===")
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    try:
        from npu.hailo_status import check_hailo_status

        hef_path = PKG_ROOT / "models" / "custom_model.hef"
        npu_st = check_hailo_status(hef_path=hef_path)
        print(f"  Status code     : {npu_st['status_code']}")
        print(f"  Available       : {npu_st['available']}")
        print(f"  Device nodes    : {npu_st['device_nodes'] or 'None detected'}")
        print(f"  PCI driver      : {npu_st['driver_loaded']}")
        print(f"  HailoRT Python  : {npu_st['hailort_available']}")
        print(f"  hailortcli      : {npu_st['hailort_cli_available']}")
        print(f"  HEF model ready : {npu_st['model_exists']} ({hef_path})")
        if not npu_st["available"]:
            warnings.append(f"Hailo-8 NPU not ready ({npu_st['status_code']}): {npu_st['message']}")
    except Exception as exc:
        warnings.append(f"Hailo-8 NPU diagnostic probe failed: {exc}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nDiagnostics status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
