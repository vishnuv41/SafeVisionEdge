"""Compile validated YOLOv11-S ONNX model to Hailo-8 `.hef` using Hailo Dataflow Compiler."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL_SCRIPT = """# Hailo-8 YOLOv11-S normalization & quantization script
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
model_optimization_flavor(optimization_level=2, compression_level=0)
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def check_dfc_environment() -> dict[str, Any]:
    """Check whether Hailo Dataflow Compiler (`hailo_sdk_client`) is available on this host."""
    try:
        sdk = importlib.import_module("hailo_sdk_client")
        ver = getattr(sdk, "__version__", "installed")
        return {
            "available": True,
            "version": ver,
            "platform": platform.platform(),
            "machine": platform.machine(),
        }
    except ImportError as exc:
        return {
            "available": False,
            "version": None,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "error": str(exc),
            "note": (
                "Hailo Dataflow Compiler (hailo_sdk_client) runs on x86_64 Linux (Ubuntu 22.04/24.04) "
                "and is NOT installed on this host."
            ),
        }


def compile_onnx_to_hef(
    onnx_path: Path,
    calib_npy: Path,
    out_hef: Path,
    *,
    hw_arch: str = "hailo8",
    net_name: str = "safevision_yolov11s",
    end_nodes: list[str] | None = None,
    alls_script: Path | None = None,
) -> dict[str, Any]:
    """Translate, quantize (INT8), and compile ONNX -> Hailo-8 `.hef`."""
    env = check_dfc_environment()
    if not env["available"]:
        raise RuntimeError(
            f"Hailo Dataflow Compiler unavailable ({env['error']}). {env['note']}"
        )
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not calib_npy.is_file():
        raise FileNotFoundError(f"Calibration dataset not found: {calib_npy}")

    from hailo_sdk_client import ClientRunner  # type: ignore[import-not-found]

    calib = np.load(str(calib_npy))
    if calib.ndim != 4 or calib.shape[-1] != 3:
        raise ValueError(f"Expected NHWC calibration array (N, 640, 640, 3), got {calib.shape}")

    runner = ClientRunner(hw_arch=hw_arch)
    runner.translate_onnx_model(
        str(onnx_path),
        net_name,
        start_node_names=["images"],
        end_node_names=end_nodes,
        net_input_shapes={"images": [1, 3, 640, 640]},
    )

    script_text = alls_script.read_text(encoding="utf-8") if alls_script and alls_script.is_file() else DEFAULT_MODEL_SCRIPT
    runner.load_model_script(script_text)
    runner.optimize(calib)
    hef_bytes = runner.compile()

    out_hef.parent.mkdir(parents=True, exist_ok=True)
    out_hef.write_bytes(hef_bytes)

    har_path = out_hef.with_suffix(".har")
    runner.save_har(str(har_path))

    provenance = {
        "compiled_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hef_file": str(out_hef),
        "hef_sha256": _sha256(out_hef),
        "hef_size_bytes": out_hef.stat().st_size,
        "har_file": str(har_path),
        "hw_arch": hw_arch,
        "net_name": net_name,
        "source_onnx": str(onnx_path),
        "source_onnx_sha256": _sha256(onnx_path),
        "calib_dataset": str(calib_npy),
        "calib_samples": int(calib.shape[0]),
        "dfc_version": env["version"],
        "classes": [
            "person",
            "bicycle",
            "motorcycle",
            "auto_rickshaw",
            "car",
            "bus",
            "truck",
        ],
    }
    prov_path = out_hef.parent / "HEF_PROVENANCE.json"
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile YOLOv11-S ONNX to Hailo-8 .hef")
    parser.add_argument("--onnx", type=Path, default=Path("models/custom_model.onnx"))
    parser.add_argument("--calib", type=Path, default=Path("conversion/calib_nhwc_uint8.npy"))
    parser.add_argument("--out", type=Path, default=Path("models/custom_model.hef"))
    parser.add_argument("--hw-arch", default="hailo8", choices=["hailo8", "hailo8l"])
    parser.add_argument("--end-nodes", nargs="*", default=None, help="Optional ONNX end node names for split/NMS.")
    parser.add_argument("--alls", type=Path, default=None, help="Optional Hailo .alls model script.")
    parser.add_argument("--check-only", action="store_true", help="Only check DFC compiler availability.")
    args = parser.parse_args()

    if args.check_only:
        print(json.dumps(check_dfc_environment(), indent=2))
        return 0

    prov = compile_onnx_to_hef(
        args.onnx,
        args.calib,
        args.out,
        hw_arch=args.hw_arch,
        end_nodes=args.end_nodes,
        alls_script=args.alls,
    )
    print(json.dumps(prov, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
