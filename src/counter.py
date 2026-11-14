"""Per-frame road-user counting, privacy blurring, CSV logging, and annotation rendering."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from decode import Detection

CSV_COLUMNS = [
    "frame_id",
    "person",
    "bicycle",
    "motorcycle",
    "auto_rickshaw",
    "car",
    "bus",
    "truck",
    "persons_footpath",
    "persons_road",
    "persons_crossing",
    "vehicles_road",
    "vehicles_crossing",
    "density_class",
    "co_presence",
    "vru_share",
    "safety_pulse",
    "inference_ms",
    "total_ms",
]

CLASS_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (255, 190, 50),   # person
    1: (50, 220, 120),   # bicycle
    2: (0, 165, 255),    # motorcycle
    3: (0, 235, 235),    # auto_rickshaw
    4: (255, 100, 100),  # car
    5: (220, 70, 220),   # bus
    6: (60, 60, 255),    # truck
}


def count_detections(
    detections: Iterable[Detection], class_names: dict[int, str]
) -> dict[str, int]:
    """Return zero-filled per-class counts keyed by `<class_name>`."""
    counts = {name: 0 for name in class_names.values()}
    for det in detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1
    return counts


def blur_persons(
    img_bgr: np.ndarray, detections: Iterable[Detection], kernel_size: int = 35
) -> np.ndarray:
    """Apply Gaussian blur to `person` bounding boxes for privacy before saving outputs."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    k = max(3, kernel_size | 1)  # Ensure odd kernel size >= 3
    for det in detections:
        if det.class_name != "person" and det.class_id != 0:
            continue
        x1 = max(0, min(w - 1, int(np.floor(det.x1))))
        y1 = max(0, min(h - 1, int(np.floor(det.y1))))
        x2 = max(x1 + 1, min(w, int(np.ceil(det.x2))))
        y2 = max(y1 + 1, min(h, int(np.ceil(det.y2))))
        roi = out[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        kx = min(k, (roi.shape[1] - 1) | 1 if roi.shape[1] > 1 else 1)
        ky = min(k, (roi.shape[0] - 1) | 1 if roi.shape[0] > 1 else 1)
        if kx >= 3 or ky >= 3:
            out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (max(1, kx), max(1, ky)), 0)
    return out


def build_csv_row(
    frame_id: str | int,
    safety: dict[str, Any],
    inference_ms: float,
    total_ms: float,
) -> dict[str, object]:
    """Format one complete per-frame safety CSV record with zero-filled absent classes."""
    counts = safety["counts"]
    return {
        "frame_id": frame_id,
        "person": int(counts.get("person", 0)),
        "bicycle": int(counts.get("bicycle", 0)),
        "motorcycle": int(counts.get("motorcycle", 0)),
        "auto_rickshaw": int(counts.get("auto_rickshaw", 0)),
        "car": int(counts.get("car", 0)),
        "bus": int(counts.get("bus", 0)),
        "truck": int(counts.get("truck", 0)),
        "persons_footpath": int(safety.get("persons_footpath", 0)),
        "persons_road": int(safety.get("persons_road", 0)),
        "persons_crossing": int(safety.get("persons_crossing", 0)),
        "vehicles_road": int(safety.get("vehicles_road", 0)),
        "vehicles_crossing": int(safety.get("vehicles_crossing", 0)),
        "density_class": str(safety.get("density_class", "Low")),
        "co_presence": str(safety.get("co_presence", "NO")),
        "vru_share": round(float(safety.get("vru_share", 0.0)), 4),
        "safety_pulse": int(safety.get("safety_pulse", 0)),
        "inference_ms": round(float(inference_ms), 2),
        "total_ms": round(float(total_ms), 2),
    }


def write_counts_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def render_annotations(
    img_bgr: np.ndarray,
    detections: list[Detection],
    safety: dict[str, Any],
    inference_ms: float,
    total_ms: float,
    frame_label: str = "",
    blur_privacy: bool = True,
) -> np.ndarray:
    """Apply optional person blurring, then draw boxes, labels, and Safety Pulse HUD."""
    vis = blur_persons(img_bgr, detections) if blur_privacy else img_bgr.copy()
    h, w = vis.shape[:2]
    counts = safety["counts"]

    for det in detections:
        color = CLASS_PALETTE.get(det.class_id, (200, 200, 200))
        x1, y1, x2, y2 = int(round(det.x1)), int(round(det.y1)), int(round(det.x2)), int(round(det.y2))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(th + 4, y1)
        cv2.rectangle(vis, (x1, ty - th - 4), (min(w, x1 + tw + 4), ty), color, -1)
        cv2.putText(vis, label, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    banner = (
        f"{frame_label + ' | ' if frame_label else ''}"
        f"P:{counts.get('person', 0)} B:{counts.get('bicycle', 0)} "
        f"M:{counts.get('motorcycle', 0)} A:{counts.get('auto_rickshaw', 0)} "
        f"C:{counts.get('car', 0)} Bus:{counts.get('bus', 0)} T:{counts.get('truck', 0)} | "
        f"Pulse:{safety['safety_pulse']}({safety['safety_pulse_label']}) "
        f"Den:{safety['density_class']} CoP:{safety['co_presence']} VRU:{safety['vru_share']:.2f} | "
        f"inf:{inference_ms:.1f}ms tot:{total_ms:.1f}ms"
    )
    cv2.rectangle(vis, (0, 0), (w, 28), (15, 15, 15), -1)
    cv2.putText(vis, banner, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return vis
