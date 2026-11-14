"""Real Hailo-8 hardware, driver, firmware, HailoRT, and HEF readiness checks."""

from __future__ import annotations

import importlib
import importlib.util
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable


class HailoStatusCode:
    DEVICE_NOT_FOUND = "DEVICE_NOT_FOUND"
    RUNTIME_NOT_INSTALLED = "RUNTIME_NOT_INSTALLED"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_INCOMPATIBLE = "MODEL_INCOMPATIBLE"
    MODEL_LOADING = "MODEL_LOADING"
    READY = "READY"
    INFERENCE_ERROR = "INFERENCE_ERROR"

    # Backward compatibility aliases
    NPU_HARDWARE_UNAVAILABLE = "DEVICE_NOT_FOUND"
    HAILORT_UNAVAILABLE = "RUNTIME_NOT_INSTALLED"
    NPU_MODEL_NOT_READY = "MODEL_NOT_FOUND"
    NPU_ERROR = "INFERENCE_ERROR"


def _default_check_device_nodes() -> list[str]:
    dev_dir = Path("/dev")
    if not dev_dir.is_dir():
        return []
    return sorted(str(p) for p in dev_dir.glob("hailo*"))


def _default_check_driver_loaded() -> bool:
    if Path("/sys/module/hailo_pci").is_dir() or Path("/sys/class/hailo_chardev").is_dir():
        return True
    return len(_default_check_device_nodes()) > 0


def _default_run_hailortcli(args: list[str], timeout: float = 2.5) -> tuple[int, str]:
    cli = shutil.which("hailortcli")
    if not cli:
        return -1, ""
    try:
        proc = subprocess.run(
            [cli, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode, out.strip()
    except Exception as exc:
        return -2, str(exc)


def _default_import_hailort() -> Any | None:
    spec = importlib.util.find_spec("hailo_platform")
    if spec is None:
        return None
    return importlib.import_module("hailo_platform")


def read_hailo_temperature_c(
    run_cli_fn: Callable[[list[str]], tuple[int, str]] | None = None,
) -> float | None:
    """Read real Hailo-8 temperature in Celsius from sysfs or hailortcli; never fabricate."""
    # Check Linux hwmon sysfs entries for hailo thermal zones
    hwmon_root = Path("/sys/class/hwmon")
    if hwmon_root.is_dir():
        for hdir in sorted(hwmon_root.iterdir()):
            name_file = hdir / "name"
            if name_file.is_file():
                try:
                    sensor_name = name_file.read_text(encoding="utf-8", errors="ignore").strip().lower()
                except Exception:
                    sensor_name = ""
                if "hailo" in sensor_name:
                    temp_file = hdir / "temp1_input"
                    if temp_file.is_file():
                        try:
                            raw = int(temp_file.read_text(encoding="utf-8").strip())
                            return round(raw / 1000.0 if raw > 200 else float(raw), 1)
                        except Exception:
                            pass

    # Fallback: check hailortcli fw-control temperature if device exists
    if _default_check_device_nodes() or run_cli_fn is not None:
        cli_fn = run_cli_fn or _default_run_hailortcli
        rc, out = cli_fn(["fw-control", "temperature"])
        if rc == 0 and out:
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:C|°C|celsius)", out, re.IGNORECASE)
            if m:
                return round(float(m.group(1)), 1)
    return None


def _parse_fw_identify(output: str) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if not output:
        return info
    for line in output.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        kl = k.strip().lower()
        val = v.strip()
        if "device architecture" in kl or "architecture" in kl:
            info["device_architecture"] = val
        elif "firmware version" in kl:
            info["firmware_version"] = val
        elif "serial number" in kl:
            info["serial_number"] = val
        elif "part number" in kl:
            info["part_number"] = val
        elif "board name" in kl:
            info["board_name"] = val
    return info


def check_hailo_status(
    hef_path: Path,
    *,
    device_nodes_fn: Callable[[], list[str]] | None = None,
    driver_loaded_fn: Callable[[], bool] | None = None,
    import_hailort_fn: Callable[[], Any | None] | None = None,
    run_cli_fn: Callable[[list[str]], tuple[int, str]] | None = None,
) -> dict[str, Any]:
    """Inspect real Hailo-8 device, driver, firmware, HailoRT, and `.hef` model readiness."""
    hef_file = Path(hef_path)
    sys_name = platform.system()
    arch = platform.machine()

    nodes = (device_nodes_fn or _default_check_device_nodes)()
    driver_loaded = (driver_loaded_fn or _default_check_driver_loaded)()
    cli_path = shutil.which("hailortcli")
    cli_fn = run_cli_fn or _default_run_hailortcli

    hailort_mod = None
    hailort_error: str | None = None
    try:
        hailort_mod = (import_hailort_fn or _default_import_hailort)()
    except Exception as exc:
        hailort_error = str(exc)

    hailort_available = hailort_mod is not None
    hailort_version = getattr(hailort_mod, "__version__", None) if hailort_mod is not None else None

    fw_info: dict[str, Any] = {}
    scan_devices: list[str] = []
    if nodes or cli_path or run_cli_fn is not None:
        if cli_path or run_cli_fn is not None:
            rc_scan, out_scan = cli_fn(["scan"])
            if rc_scan == 0 and out_scan:
                scan_devices = [
                    ln.strip()
                    for ln in out_scan.splitlines()
                    if ln.strip() and ("hailo" in ln.lower() or "000" in ln)
                ]
            if nodes or scan_devices:
                rc_id, out_id = cli_fn(["fw-control", "identify"])
                if rc_id == 0 and out_id:
                    fw_info = _parse_fw_identify(out_id)

    device_connected = bool(nodes or scan_devices or fw_info.get("device_architecture"))
    model_exists = hef_file.is_file()
    model_size_mb = round(hef_file.stat().st_size / (1024 ** 2), 2) if model_exists else 0.0
    temp_c = read_hailo_temperature_c(run_cli_fn=run_cli_fn) if device_connected else None

    # Check model provenance approval status
    prov_file = hef_file.parent / "HEF_PROVENANCE.json"
    hef_approved = True
    unapproved_reason = ""
    if prov_file.is_file():
        try:
            prov_data = json.loads(prov_file.read_text(encoding="utf-8"))
            if prov_data.get("approved_for_safevision") is False:
                hef_approved = False
                unapproved_reason = prov_data.get("rejection_reason", "Model not approved for 7-class SafeVision inference.")
        except Exception:
            pass

    # Determine status code cleanly with explicit priority (never silently fake readiness)
    if not device_connected:
        status_code = HailoStatusCode.DEVICE_NOT_FOUND
        if sys_name != "Linux":
            message = f"Hailo-8 PCIe hardware unavailable on {sys_name} ({arch}); requires Linux ARM64/x86_64 with Hailo-8 AI HAT+."
        elif not driver_loaded:
            message = "Hailo-8 PCIe device not detected (/dev/hailo* absent, hailo_pci driver not loaded)."
        else:
            message = "Hailo driver loaded, but no active Hailo-8 device detected."
    elif not hailort_available:
        status_code = HailoStatusCode.RUNTIME_NOT_INSTALLED
        message = (
            f"Hailo-8 device detected, but HailoRT Python package (hailo_platform) is not available"
            f"{': ' + hailort_error if hailort_error else '.'}"
        )
    elif not model_exists:
        status_code = HailoStatusCode.MODEL_NOT_FOUND
        message = (
            f"Hailo-8 hardware and HailoRT ready, but 7-class .hef model not found at {hef_file.name}. "
            "Compile fine-tuned weights using conversion/compile_hef.py."
        )
    elif not hef_approved:
        status_code = HailoStatusCode.MODEL_INCOMPATIBLE
        message = f"Hailo-8 HEF model NOT APPROVED / INCOMPATIBLE for SafeVision 7-class inference: {unapproved_reason}"
    else:
        # Check architecture and class count from HEF metadata if hailort is available
        classes_verified = True
        mismatch_msg = ""
        if hailort_mod is not None:
            try:
                hef_obj = hailort_mod.HEF(str(hef_file))
                out_infos = hef_obj.get_output_vstream_infos()
                if out_infos:
                    for o in out_infos:
                        shape = tuple(getattr(o, "shape", ()))
                        if hasattr(o, "nms_shape") or (len(shape) >= 1 and shape[-1] == 80):
                            classes_verified = False
                            mismatch_msg = "HEF output contains 80 classes (COCO generic), expected 7 SafeVision classes."
                            break
            except Exception:
                pass

        if not classes_verified:
            status_code = HailoStatusCode.MODEL_INCOMPATIBLE
            message = f"Hailo-8 HEF model incompatible: {mismatch_msg}"
        else:
            status_code = HailoStatusCode.READY
            message = f"Hailo-8 NPU ready ({fw_info.get('device_architecture', 'HAILO8')}, model={hef_file.name})."

    # Also surface clear secondary readiness notes when hardware is absent AND model is absent
    model_status = "ready" if (model_exists and hef_approved and status_code == HailoStatusCode.READY) else ("NOT_APPROVED" if not hef_approved else (HailoStatusCode.MODEL_INCOMPATIBLE if status_code == HailoStatusCode.MODEL_INCOMPATIBLE else HailoStatusCode.MODEL_NOT_FOUND))

    return {
        "available": status_code == HailoStatusCode.READY,
        "status_code": status_code,
        "message": message,
        "device_connected": device_connected,
        "device_nodes": nodes,
        "scan_devices": scan_devices,
        "driver_loaded": driver_loaded,
        "hailort_available": hailort_available,
        "hailort_version": hailort_version,
        "hailort_cli_available": bool(cli_path),
        "hailort_error": hailort_error,
        "firmware_available": bool(fw_info.get("firmware_version")),
        "firmware_version": fw_info.get("firmware_version"),
        "device_architecture": fw_info.get("device_architecture"),
        "board_name": fw_info.get("board_name"),
        "temperature_c": temp_c,
        "model_status": model_status,
        "model_exists": model_exists,
        "model_name": hef_file.name,
        "model_path": str(hef_file),
        "model_size_mb": model_size_mb,
    }
