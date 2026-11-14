"""Persistence and CSV/JSON formatting for CPU vs Hailo-8 NPU comparison reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def save_comparison_reports(report: dict[str, Any], outputs_dir: Path) -> dict[str, str]:
    """Write cpu/latest.json, npu/latest.json, comparison/latest.json, and comparison/latest.csv."""
    bm_root = Path(outputs_dir) / "benchmark"
    cpu_dir = bm_root / "cpu"
    npu_dir = bm_root / "npu"
    cmp_dir = bm_root / "comparison"
    for d in (cpu_dir, npu_dir, cmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    cpu_path = cpu_dir / "latest.json"
    npu_path = npu_dir / "latest.json"
    cmp_json_path = cmp_dir / "latest.json"
    cmp_csv_path = cmp_dir / "latest.csv"

    cpu_payload = {
        "timestamp_utc": report.get("timestamp_utc"),
        "platform": report.get("platform"),
        "warmup_runs_excluded": report.get("warmup_runs_excluded"),
        "measured_runs": report.get("measured_runs"),
        "engine": report.get("cpu"),
        "hardware_monitoring": report.get("hardware_monitoring"),
    }
    npu_payload = {
        "timestamp_utc": report.get("timestamp_utc"),
        "platform": report.get("platform"),
        "warmup_runs_excluded": report.get("warmup_runs_excluded"),
        "measured_runs": report.get("measured_runs"),
        "engine": report.get("npu"),
        "hardware_monitoring": report.get("hardware_monitoring"),
    }

    cpu_path.write_text(json.dumps(cpu_payload, indent=2), encoding="utf-8")
    npu_path.write_text(json.dumps(npu_payload, indent=2), encoding="utf-8")
    cmp_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Write flat CSV side-by-side comparison rows
    fieldnames = [
        "timestamp_utc",
        "engine_id",
        "available",
        "status_code",
        "model_name",
        "runtime",
        "backend",
        "processed_frames",
        "failed_frames",
        "mean_inference_ms",
        "median_inference_ms",
        "p95_inference_ms",
        "inference_fps",
        "mean_total_ms",
        "median_total_ms",
        "p95_total_ms",
        "end_to_end_fps",
        "total_detections",
        "cpu_temp_mean_c",
        "npu_temp_c",
        "ram_rss_mean_mb",
        "throttling_status",
        "duration_sec",
    ]
    hw = report.get("hardware_monitoring", {})
    acc = report.get("accuracy_comparison", {})
    rows = []
    for eid in ("cpu", "npu"):
        eng = report.get(eid) or {}
        det_side = acc.get(eid) or {}
        rows.append(
            {
                "timestamp_utc": report.get("timestamp_utc"),
                "engine_id": eid,
                "available": eng.get("available"),
                "status_code": eng.get("status_code"),
                "model_name": eng.get("model_name"),
                "runtime": eng.get("runtime"),
                "backend": eng.get("backend"),
                "processed_frames": eng.get("processed_frames"),
                "failed_frames": eng.get("failed_frames"),
                "mean_inference_ms": eng.get("mean_inference_ms"),
                "median_inference_ms": eng.get("median_inference_ms"),
                "p95_inference_ms": eng.get("p95_inference_ms"),
                "inference_fps": eng.get("inference_fps"),
                "mean_total_ms": eng.get("mean_total_ms"),
                "median_total_ms": eng.get("median_total_ms"),
                "p95_total_ms": eng.get("p95_total_ms"),
                "end_to_end_fps": eng.get("end_to_end_fps"),
                "total_detections": det_side.get("total_detections"),
                "cpu_temp_mean_c": (hw.get("cpu_temperature_c") or {}).get("mean"),
                "npu_temp_c": hw.get("npu_temperature_c"),
                "ram_rss_mean_mb": (hw.get("process_rss_mb") or {}).get("mean"),
                "throttling_status": (hw.get("throttling") or {}).get("status"),
                "duration_sec": report.get("duration_sec"),
            }
        )

    with open(cmp_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "cpu_json": str(cpu_path),
        "npu_json": str(npu_path),
        "comparison_json": str(cmp_json_path),
        "comparison_csv": str(cmp_csv_path),
    }


def load_latest_comparison(outputs_dir: Path) -> dict[str, Any]:
    """Load latest saved comparison benchmark report if present."""
    cmp_path = Path(outputs_dir) / "benchmark" / "comparison" / "latest.json"
    if not cmp_path.is_file():
        return {"available": False}
    data = json.loads(cmp_path.read_text(encoding="utf-8"))
    data["available"] = True
    return data
