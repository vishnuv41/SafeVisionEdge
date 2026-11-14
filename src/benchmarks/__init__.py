"""Controlled CPU vs Hailo-8 NPU comparison benchmark and hardware telemetry."""

from benchmarks.comparison_runner import ComparisonRunner, compute_detection_agreement
from benchmarks.hardware_monitor import HardwareMonitor
from benchmarks.performance_report import load_latest_comparison, save_comparison_reports

__all__ = [
    "ComparisonRunner",
    "HardwareMonitor",
    "compute_detection_agreement",
    "load_latest_comparison",
    "save_comparison_reports",
]
