"""Post-processing and normalization helpers for Hailo-8 detections."""

from __future__ import annotations

from typing import Any

from decode import Detection


def filter_and_scale_detections(
    detections: list[Detection],
    orig_shape: tuple[int, int],
    conf_thres: float = 0.25,
    max_det: int = 300,
) -> list[Detection]:
    """Clip detections to frame bounds, apply confidence floor, and sort descending."""
    h, w = orig_shape[:2]
    filtered: list[Detection] = []
    for det in detections:
        if det.confidence < conf_thres:
            continue
        x1 = max(0.0, min(float(w), float(det.x1)))
        y1 = max(0.0, min(float(h), float(det.y1)))
        x2 = max(0.0, min(float(w), float(det.x2)))
        y2 = max(0.0, min(float(h), float(det.y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        filtered.append(
            Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(det.confidence),
                class_id=int(det.class_id),
                class_name=str(det.class_name),
            )
        )
    filtered.sort(key=lambda d: d.confidence, reverse=True)
    return filtered[:max_det]


def detections_to_normalized(detections: list[Detection]) -> list[dict[str, Any]]:
    """Convert internal Detection objects to the unified normalized API schema."""
    return [
        {
            "class_id": int(det.class_id),
            "class_name": str(det.class_name),
            "confidence": round(float(det.confidence), 4),
            "bbox": [
                round(float(det.x1), 2),
                round(float(det.y1), 2),
                round(float(det.x2), 2),
                round(float(det.y2), 2),
            ],
        }
        for det in detections
    ]
