#!/usr/bin/env python3
"""SafeVision Edge — Virtual Environment & Dependency Integrity Pre-Check.

Safely checks and prepares dependencies for Raspberry Pi 5 + Hailo-8 AI HAT+.
Protects ~/hailo-venv and hailo_platform against destructive pip reinstalls,
preserving the precompiled numpy C-ABI and existing driver bindings.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

# Required runtime modules and their pip package names
REQUIRED_COMPONENTS = [
    ("numpy", "numpy>=1.24.0", "Core numerical backend"),
    ("cv2", "opencv-python-headless>=4.8.0", "Computer vision & letterbox"),
    ("torch", "torch>=2.4.0", "CPU tensor wrapper (CPU-only index)"),
    ("executorch", "executorch>=0.4.0", "ExecuTorch XNNPACK CPU runtime"),
    ("fastapi", "fastapi>=0.100.0", "REST API framework"),
    ("uvicorn", "uvicorn[standard]>=0.22.0", "ASGI web server"),
    ("multipart", "python-multipart>=0.0.6", "Multipart form upload parser"),
    ("pydantic", "pydantic>=2.0.0", "Data schemas & validation"),
    ("psutil", "psutil>=5.9.0", "System hardware telemetry"),
    ("httpx", "httpx>=0.25.0", "HTTP client for testing"),
]


def check_hailo_platform() -> tuple[bool, str]:
    """Check if hailo_platform is present in the current Python environment."""
    try:
        mod = importlib.import_module("hailo_platform")
        ver = getattr(mod, "__version__", "installed")
        return True, str(ver)
    except Exception:
        return False, "not installed"


def get_installed_version(module_name: str) -> str | None:
    """Return installed version of module, or None if import fails."""
    try:
        mod = importlib.import_module(module_name)
        return str(getattr(mod, "__version__", "present"))
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safe dependency check and non-destructive installer for SafeVision Edge."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check and report status without installing missing packages.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Exit with 0 if all satisfied, 1 if any missing, no changes.",
    )
    args = parser.parse_args()

    print("==================================================================")
    print(" SafeVision Edge — Environment & Dependency Check")
    print("==================================================================")
    print(f"Python Binary   : {sys.executable}")
    print(f"Python Version  : {sys.version.split()[0]}")
    print(f"Prefix / Venv   : {sys.prefix}")

    # Check HailoRT integration
    has_hailo, hailo_ver = check_hailo_platform()
    if has_hailo:
        print(f"[PROTECTED] hailo_platform detected (HailoRT {hailo_ver}).")
        print("            Existing numpy will be PRESERVED to protect C-ABI bindings.")
    else:
        print("[INFO] hailo_platform not detected in active Python environment.")
        print("       (HailoRT is accessed when running inside ~/hailo-venv on Raspberry Pi).")

    missing_pkgs: list[tuple[str, str, str]] = []
    satisfied_pkgs: list[tuple[str, str, str]] = []

    for mod_name, pkg_spec, desc in REQUIRED_COMPONENTS:
        ver = get_installed_version(mod_name)
        if ver is not None:
            satisfied_pkgs.append((mod_name, pkg_spec, ver))
            print(f"  [OK]       {pkg_spec:<32} {ver}")
        else:
            missing_pkgs.append((mod_name, pkg_spec, desc))
            print(f"  [MISSING]  {pkg_spec:<32} ({desc})")

    # Check for unwanted CUDA packages
    cuda_found = []
    for dist in ("nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12", "nvidia-cublas-cu12", "cuda-toolkit"):
        if importlib.util.find_spec(dist.replace("-", "_")) is not None:
            cuda_found.append(dist)
    if cuda_found:
        print(f"[WARNING] Unnecessary CUDA packages found: {', '.join(cuda_found)}")

    if not missing_pkgs:
        print("\n[SUCCESS] All SafeVision backend dependencies are satisfied!")
        return 0

    if args.check_only:
        print(f"\n[FAIL] {len(missing_pkgs)} required dependencies are missing.")
        return 1

    if args.dry_run:
        print(f"\n[DRY RUN] Would install {len(missing_pkgs)} missing packages:")
        for _, pkg_spec, desc in missing_pkgs:
            print(f"  - {pkg_spec} ({desc})")
        print("No changes made.")
        return 0

    print(f"\n--- Installing {len(missing_pkgs)} missing packages safely ---")
    pip_cmd_base = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
    ]

    for mod_name, pkg_spec, desc in missing_pkgs:
        # If numpy is already present and hailo is present, never touch numpy
        if mod_name == "numpy" and has_hailo:
            print(f"[SKIP] Preserving existing numpy for hailo_platform C-ABI.")
            continue

        cmd = pip_cmd_base + [pkg_spec]
        print(f"Installing: {pkg_spec} ...")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"ERROR: Failed to install {pkg_spec} (exit code {res.returncode})")
            return res.returncode

    # Verify after installation
    all_ok = True
    for mod_name, pkg_spec, _ in missing_pkgs:
        ver = get_installed_version(mod_name)
        if ver is None:
            print(f"ERROR: Package {pkg_spec} failed verification after install.")
            all_ok = False
        else:
            print(f"  [VERIFIED] {pkg_spec:<32} {ver}")

    if all_ok:
        print("\n[SUCCESS] All missing dependencies installed successfully.")
        return 0
    else:
        print("\n[ERROR] Some dependencies could not be resolved.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
