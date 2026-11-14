"""Real ExecuTorch XNNPACK hardware validation and benchmarking script for Raspberry Pi 5."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil

EXPECTED_SHA256 = "E9C932AA70EDBE3D8065612B576AAA25B4CD86840A3017864461A436D5959B56"

PKG_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PKG_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from counter import build_csv_row, render_annotations  # noqa: E402
from infer import EdgeDetector, load_configs  # noqa: E402
from safety_pulse import ALL_CLASSES, analyze_frame, load_safety_config  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _detect_pi_board() -> str | None:
    p = Path("/proc/device-tree/model")
    if p.is_file():
        try:
            return p.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="ignore").strip()
        except Exception:
            return None
    return None


def _read_os_pretty() -> str | None:
    p = Path("/etc/os-release")
    if p.is_file():
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    return None


def _read_pi_thermals() -> tuple[float | None, bool | None, int | None]:
    temp_c: float | None = None
    throttled: bool | None = None
    throttled_bits: int | None = None
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
                throttled_bits = int(out.split("=", 1)[1], 16)
                throttled = (throttled_bits & 0xF) != 0
        except Exception:
            throttled = None
    return temp_c, throttled, throttled_bits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate real ExecuTorch XNNPACK inference, decoding, 7-class counts, and metrics."
    )
    parser.add_argument("--model", type=Path, default=PKG_ROOT / "models" / "custom_model.pte")
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        default=[
            PKG_ROOT / "tests" / "fixtures" / "sample_road_1.jpg",
            PKG_ROOT / "tests" / "fixtures" / "sample_road_2.jpg",
        ],
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10, help="Measured frames across input images.")
    parser.add_argument(
        "--update-dashboard-benchmark",
        action="store_true",
        help="Also write outputs/benchmark.json for /api/v1/system/benchmark.",
    )
    args = parser.parse_args()

    model_path = args.model.resolve()
    sha = _sha256(model_path)
    if sha != EXPECTED_SHA256:
        raise RuntimeError(f"Model SHA-256 mismatch: expected {EXPECTED_SHA256}, got {sha}")

    frames_bgr: list[tuple[str, np.ndarray]] = []
    for img_path in args.images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        frames_bgr.append((img_path.name, bgr))

    class_names, prep_cfg, zone_analyzer = load_configs(PKG_ROOT / "config")
    safety_cfg = load_safety_config(PKG_ROOT / "config" / "safety_config.json")
    blur_persons_enabled = bool(safety_cfg.get("privacy_blur_persons", True))

    from executorch.runtime import Runtime

    rt = Runtime.get()
    backends = list(rt.backend_registry.registered_backend_names)
    if "XnnpackBackend" not in backends:
        raise RuntimeError(f"XnnpackBackend not present in registered backends: {backends}")

    t_load0 = time.perf_counter()
    detector = EdgeDetector(model_path, class_names, prep_cfg)
    model_load_ms = (time.perf_counter() - t_load0) * 1000.0

    proc = psutil.Process()
    proc.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None)

    temp_before_c, throttled_before, _ = _read_pi_thermals()

    # Warmup passes
    for w_idx in range(args.warmup):
        _, frame = frames_bgr[w_idx % len(frames_bgr)]
        detector.predict_frame(frame)

    inf_ms_list: list[float] = []
    e2e_ms_list: list[float] = []
    aggregate_counts = {cls: 0 for cls in ALL_CLASSES}
    sample_boxes_first_frame: list[dict[str, Any]] = []
    annotated_sample: np.ndarray | None = None
    first_frame_safety: dict[str, Any] = {}

    for run_idx in range(args.runs):
        name, frame = frames_bgr[run_idx % len(frames_bgr)]
        t0 = time.perf_counter()
        dets, inf_ms = detector.predict_frame(frame)
        safety = analyze_frame(dets, frame.shape[:2], zone_analyzer, cfg=safety_cfg)
        vis = render_annotations(
            frame,
            dets,
            safety,
            inference_ms=inf_ms,
            total_ms=(time.perf_counter() - t0) * 1000.0,
            frame_label=name,
            blur_privacy=blur_persons_enabled,
        )
        if zone_analyzer is not None:
            vis = zone_analyzer.draw_zones(vis, safety.get("zones"))
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        _ = build_csv_row(
            frame_id=run_idx + 1,
            safety=safety,
            inference_ms=inf_ms,
            total_ms=e2e_ms,
        )

        inf_ms_list.append(inf_ms)
        e2e_ms_list.append(e2e_ms)
        for cls in ALL_CLASSES:
            aggregate_counts[cls] += int(safety["counts"].get(cls, 0))

        if run_idx == 0:
            annotated_sample = vis
            first_frame_safety = safety
            sample_boxes_first_frame = [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": round(float(d.confidence), 4),
                    "bbox_xyxy": [round(float(d.x1), 1), round(float(d.y1), 1), round(float(d.x2), 1), round(float(d.y2), 1)],
                }
                for d in dets[:15]
            ]

    temp_after_c, throttled_after, throttle_bits = _read_pi_thermals()
    rss_mb = round(proc.memory_info().rss / (1024 ** 2), 1)
    proc_cpu = round(proc.cpu_percent(interval=None), 1)
    sys_cpu = round(psutil.cpu_percent(interval=None), 1)

    inf_mean = float(np.mean(inf_ms_list))
    e2e_mean = float(np.mean(e2e_ms_list))
    mach = platform.machine()
    is_pi = "aarch64" in mach.lower() or "arm" in mach.lower()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    reports_dir = PKG_ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = reports_dir / f"validated_annotated_{stamp}.jpg"
    if annotated_sample is not None:
        cv2.imwrite(str(annotated_path), annotated_sample)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "board": _detect_pi_board() or "Non-Raspberry-Pi workstation",
            "os_pretty": _read_os_pretty() or platform.platform(),
            "platform": platform.platform(),
            "machine": mach,
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "ram_total_mb": round(psutil.virtual_memory().total / (1024 ** 2), 1),
            "is_raspberry_pi": is_pi,
        },
        "runtime": {
            "registered_backends": backends,
            "xnnpack_active": "XnnpackBackend" in backends,
            "model_path": str(model_path),
            "model_sha256": sha,
            "model_size_mb": round(model_path.stat().st_size / (1024 ** 2), 3),
            "model_load_ms": round(model_load_ms, 2),
        },
        "validation_runs": {
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
            "input_images": [name for name, _ in frames_bgr],
            "inference_ms": {
                "mean": round(inf_mean, 2),
                "median": round(float(np.median(inf_ms_list)), 2),
                "p95": round(float(np.percentile(inf_ms_list, 95)), 2),
                "fps": round(1000.0 / inf_mean, 2),
            },
            "end_to_end_ms": {
                "mean": round(e2e_mean, 2),
                "median": round(float(np.median(e2e_ms_list)), 2),
                "p95": round(float(np.percentile(e2e_ms_list, 95)), 2),
                "fps": round(1000.0 / e2e_mean, 2),
            },
        },
        "resources_and_thermals": {
            "process_rss_mb": rss_mb,
            "process_cpu_percent": proc_cpu,
            "system_cpu_percent": sys_cpu,
            "cpu_temperature_before_c": temp_before_c,
            "cpu_temperature_after_c": temp_after_c,
            "throttling_detected_after": throttled_after if throttled_after is not None else throttled_before,
            "throttled_raw_bits": throttle_bits,
        },
        "detections_verification": {
            "classes_expected": list(ALL_CLASSES),
            "aggregate_counts_across_runs": aggregate_counts,
            "first_frame_counts": first_frame_safety.get("counts", {}),
            "first_frame_safety_pulse": first_frame_safety.get("safety_pulse"),
            "first_frame_safety_label": first_frame_safety.get("safety_pulse_label"),
            "first_frame_co_presence": first_frame_safety.get("co_presence"),
            "first_frame_sample_boxes": sample_boxes_first_frame,
            "annotated_output_file": str(annotated_path),
        },
    }

    json_report_path = reports_dir / f"pi_validation_{stamp}.json"
    md_report_path = reports_dir / f"pi_validation_{stamp}.md"
    json_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        f"# SafeVision Edge Hardware Validation ({stamp})",
        "",
        f"- **Board:** {report['hardware']['board']}",
        f"- **OS:** {report['hardware']['os_pretty']} (`{mach}`)",
        f"- **Python:** `{platform.python_version()}`",
        f"- **ExecuTorch Backends:** `{backends}`",
        f"- **Model SHA-256:** `{sha}`",
        f"- **Model Load Time:** `{report['runtime']['model_load_ms']} ms`",
        f"- **Mean Inference Latency:** `{report['validation_runs']['inference_ms']['mean']} ms` (`{report['validation_runs']['inference_ms']['fps']} FPS`)",
        f"- **Mean End-to-End Latency:** `{report['validation_runs']['end_to_end_ms']['mean']} ms` (`{report['validation_runs']['end_to_end_ms']['fps']} FPS`)",
        f"- **Process RSS RAM:** `{rss_mb} MiB` (`process_cpu={proc_cpu}%`, `system_cpu={sys_cpu}%`)",
        f"- **CPU Temperature (before/after):** `{temp_before_c}` / `{temp_after_c}` °C",
        f"- **Throttling Detected:** `{report['resources_and_thermals']['throttling_detected_after']}`",
        f"- **Aggregate 7-Class Counts ({args.runs} runs):** `{aggregate_counts}`",
        f"- **Annotated Validation Image:** `{annotated_path}`",
        "",
    ]
    md_report_path.write_text("\n".join(md_lines), encoding="utf-8")

    if args.update_dashboard_benchmark:
        dash_bm = {
            "platform": report["hardware"]["platform"],
            "machine": mach,
            "processor": report["hardware"]["processor"],
            "python": report["hardware"]["python"],
            "is_raspberry_pi": is_pi,
            "model_path": str(model_path),
            "model_size_mb": report["runtime"]["model_size_mb"],
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
            "forward_only_ms": report["validation_runs"]["inference_ms"],
            "end_to_end_ms": report["validation_runs"]["end_to_end_ms"],
            "process_rss_mb": rss_mb,
            "process_cpu_percent": proc_cpu,
            "system_cpu_percent": sys_cpu,
        }
        bm_path = PKG_ROOT / "outputs" / "benchmark.json"
        bm_path.parent.mkdir(parents=True, exist_ok=True)
        bm_path.write_text(json.dumps(dash_bm, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2), flush=True)
    print(f"\nSaved validation report: {json_report_path}")
    print(f"Saved markdown summary : {md_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
