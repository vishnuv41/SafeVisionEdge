"""Sustained hardware telemetry monitor for Raspberry Pi 5 + Hailo-8 NPU."""

from __future__ import annotations

import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from npu.hailo_status import check_hailo_status, read_hailo_temperature_c


def _read_cpu_temperature_c() -> float | None:
    """Read real CPU temperature in Celsius if exposed by Linux thermal sysfs or psutil."""
    thermal_paths = [
        Path("/sys/class/thermal/thermal_zone0/temp"),
        Path("/sys/class/hwmon/hwmon0/temp1_input"),
    ]
    for p in thermal_paths:
        if p.is_file():
            try:
                raw = int(p.read_text(encoding="utf-8").strip())
                return round(raw / 1000.0 if raw > 200 else float(raw), 1)
            except Exception:
                continue

    if hasattr(psutil, "sensors_temperatures"):
        try:
            temps = psutil.sensors_temperatures()
            for key in ("cpu_thermal", "coretemp", "k10temp", "acpitz"):
                if key in temps and temps[key]:
                    cur = temps[key][0].current
                    if cur is not None:
                        return round(float(cur), 1)
            for entries in temps.values():
                if entries and entries[0].current is not None:
                    return round(float(entries[0].current), 1)
        except Exception:
            pass
    return None


def _read_cpu_freq_mhz() -> float | None:
    """Read real current CPU frequency in MHz if available."""
    try:
        freq = psutil.cpu_freq()
        if freq is not None and freq.current and freq.current > 0:
            return round(float(freq.current), 1)
    except Exception:
        pass
    sys_freq = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    if sys_freq.is_file():
        try:
            khz = int(sys_freq.read_text(encoding="utf-8").strip())
            if khz > 0:
                return round(khz / 1000.0, 1)
        except Exception:
            pass
    return None


def _read_throttling_status() -> dict[str, Any]:
    """Read real Raspberry Pi throttling status; never report 'No throttling' when unavailable."""
    if shutil.which("vcgencmd"):
        try:
            out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True, timeout=1.5).strip()
            if "=" in out:
                bits = int(out.split("=", 1)[1], 16)
                active_now = (bits & 0xF) != 0
                occurred_past = (bits & 0xF0000) != 0
                return {
                    "supported": True,
                    "throttled": active_now,
                    "past_throttled": occurred_past,
                    "raw_hex": hex(bits),
                    "status": "Throttling active" if active_now else ("Past throttling detected" if occurred_past else "No throttling"),
                }
        except Exception:
            pass

    fw_throttled = Path("/sys/devices/platform/soc/soc:firmware/get_throttled")
    if fw_throttled.is_file():
        try:
            bits = int(fw_throttled.read_text(encoding="utf-8").strip(), 16)
            active_now = (bits & 0xF) != 0
            occurred_past = (bits & 0xF0000) != 0
            return {
                "supported": True,
                "throttled": active_now,
                "past_throttled": occurred_past,
                "raw_hex": hex(bits),
                "status": "Throttling active" if active_now else ("Past throttling detected" if occurred_past else "No throttling"),
            }
        except Exception:
            pass

    return {
        "supported": False,
        "throttled": None,
        "past_throttled": None,
        "raw_hex": None,
        "status": "Unavailable",
    }


def _read_power_telemetry() -> dict[str, Any]:
    """Read real hardware power sensor if available (never estimate from CPU% or TOPS)."""
    hwmon_root = Path("/sys/class/hwmon")
    if hwmon_root.is_dir():
        for hdir in sorted(hwmon_root.iterdir()):
            p_input = hdir / "power1_input"
            if p_input.is_file():
                try:
                    microwatts = int(p_input.read_text(encoding="utf-8").strip())
                    if microwatts > 0:
                        return {
                            "available": True,
                            "source": str(p_input),
                            "watts": round(microwatts / 1_000_000.0, 3),
                        }
                except Exception:
                    continue
    return {
        "available": False,
        "source": None,
        "watts": None,
        "status": "Unavailable (no validated power meter connected)",
    }


class HardwareMonitor:
    """Samples process & system telemetry periodically during sustained benchmark runs."""

    def __init__(self, hef_path: Path | None = None, sample_interval_sec: float = 0.25) -> None:
        self.hef_path = hef_path
        self.sample_interval_sec = max(0.05, float(sample_interval_sec))
        self._proc = psutil.Process()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []
        self._start_ts: float | None = None
        self._end_ts: float | None = None
        self._lock = threading.Lock()

    def snapshot(self, include_npu: bool = True) -> dict[str, Any]:
        """Take one immediate hardware telemetry snapshot without fabricating unavailable sensors."""
        vm = psutil.virtual_memory()
        proc_rss_mb = round(self._proc.memory_info().rss / (1024 ** 2), 1)
        proc_cpu = round(self._proc.cpu_percent(interval=None), 1)
        sys_cpu = round(psutil.cpu_percent(interval=None), 1)
        cpu_temp = _read_cpu_temperature_c()
        cpu_freq = _read_cpu_freq_mhz()
        throttle = _read_throttling_status()
        power = _read_power_telemetry()

        npu_info: dict[str, Any] = {
            "available": False,
            "status_code": "NOT_CHECKED",
            "temperature_c": None,
            "device_connected": False,
        }
        if include_npu:
            if self.hef_path is not None:
                st = check_hailo_status(self.hef_path)
                npu_info = {
                    "available": bool(st.get("available")),
                    "status_code": st.get("status_code"),
                    "message": st.get("message"),
                    "device_connected": bool(st.get("device_connected")),
                    "device_architecture": st.get("device_architecture"),
                    "temperature_c": st.get("temperature_c"),
                    "model_exists": bool(st.get("model_exists")),
                }
            else:
                npu_info["temperature_c"] = read_hailo_temperature_c()

        mach = platform.machine()
        return {
            "timestamp_epoch": round(time.time(), 3),
            "platform": platform.platform(),
            "architecture": mach,
            "is_raspberry_pi": "aarch64" in mach.lower() or "arm" in mach.lower(),
            "process_rss_mb": proc_rss_mb,
            "system_ram_used_mb": round(vm.used / (1024 ** 2), 1),
            "system_ram_total_mb": round(vm.total / (1024 ** 2), 1),
            "system_ram_percent": round(vm.percent, 1),
            "process_cpu_percent": proc_cpu,
            "system_cpu_percent": sys_cpu,
            "cpu_freq_mhz": cpu_freq,
            "cpu_temperature_c": cpu_temp,
            "throttling": throttle,
            "npu": npu_info,
            "power": power,
        }

    def start(self) -> None:
        """Start periodic background sampling."""
        self._proc.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        with self._lock:
            self._samples = []
            self._start_ts = time.perf_counter()
            self._end_ts = None
        self._stop_event.clear()
        self._samples.append(self.snapshot(include_npu=False))

        def _loop() -> None:
            while not self._stop_event.wait(self.sample_interval_sec):
                snap = self.snapshot(include_npu=False)
                with self._lock:
                    self._samples.append(snap)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        """Stop sampling and return aggregated sustained hardware telemetry summary."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        end_snap = self.snapshot(include_npu=True)
        with self._lock:
            self._end_ts = time.perf_counter()
            self._samples.append(end_snap)
            samples = list(self._samples)
            duration_s = round(max(0.0, self._end_ts - (self._start_ts or self._end_ts)), 3)

        def _agg(vals: list[float | None]) -> dict[str, float | None]:
            clean = [float(v) for v in vals if v is not None]
            if not clean:
                return {"min": None, "mean": None, "max": None}
            return {
                "min": round(min(clean), 2),
                "mean": round(float(np.mean(clean)), 2),
                "max": round(max(clean), 2),
            }

        power_vals = [s["power"]["watts"] for s in samples if s["power"].get("available")]
        throttles = [s["throttling"] for s in samples]
        throttle_supported = any(t.get("supported") for t in throttles)
        if throttle_supported:
            any_throttled = any(bool(t.get("throttled")) for t in throttles if t.get("supported"))
            any_past = any(bool(t.get("past_throttled")) for t in throttles if t.get("supported"))
            throttle_summary = {
                "supported": True,
                "throttled": any_throttled,
                "past_throttled": any_past,
                "status": "Throttling active" if any_throttled else ("Past throttling detected" if any_past else "No throttling"),
            }
        else:
            throttle_summary = {
                "supported": False,
                "throttled": None,
                "past_throttled": None,
                "status": "Unavailable",
            }

        return {
            "duration_sec": duration_s,
            "sample_count": len(samples),
            "process_rss_mb": _agg([s["process_rss_mb"] for s in samples]),
            "system_ram_used_mb": _agg([s["system_ram_used_mb"] for s in samples]),
            "system_ram_total_mb": end_snap["system_ram_total_mb"],
            "system_ram_percent": _agg([s["system_ram_percent"] for s in samples]),
            "process_cpu_percent": _agg([s["process_cpu_percent"] for s in samples]),
            "system_cpu_percent": _agg([s["system_cpu_percent"] for s in samples]),
            "cpu_freq_mhz": _agg([s["cpu_freq_mhz"] for s in samples]),
            "cpu_temperature_c": _agg([s["cpu_temperature_c"] for s in samples]),
            "npu_temperature_c": end_snap["npu"].get("temperature_c"),
            "npu_available": end_snap["npu"].get("available"),
            "npu_status_code": end_snap["npu"].get("status_code"),
            "throttling": throttle_summary,
            "power": {
                "available": bool(power_vals),
                "avg_watts": round(float(np.mean(power_vals)), 3) if power_vals else None,
                "status": "Measured" if power_vals else "Unavailable (no validated power meter connected)",
            },
        }
