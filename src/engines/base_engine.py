"""Base engine interface and normalized detection contract for SafeVision Edge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from decode import Detection


class EngineMode:
    """Strictly supported inference modes (no auto mode)."""

    CPU = "cpu"
    NPU = "npu"
    COMPARE = "compare"


ALLOWED_ENGINE_MODES = (EngineMode.CPU, EngineMode.NPU, EngineMode.COMPARE)


def detection_to_dict(det: Detection) -> dict[str, Any]:
    """Convert internal Detection object to normalized JSON-serializable dict."""
    return {
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


def dict_to_detection(item: dict[str, Any]) -> Detection:
    """Convert normalized detection dict back into internal Detection dataclass."""
    bbox = item["bbox"]
    return Detection(
        x1=float(bbox[0]),
        y1=float(bbox[1]),
        x2=float(bbox[2]),
        y2=float(bbox[3]),
        confidence=float(item["confidence"]),
        class_id=int(item["class_id"]),
        class_name=str(item["class_name"]),
    )


@dataclass
class EngineResult:
    """Structured inference result returned by any SafeVision inference engine."""

    engine_id: str
    engine_name: str
    model_name: str
    detections: list[Detection]
    inference_ms: float
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0
    error: str | None = None
    normalized_detections: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.normalized_detections and self.detections:
            self.normalized_detections = [detection_to_dict(d) for d in self.detections]
        if self.total_ms <= 0.0:
            self.total_ms = max(self.inference_ms, self.preprocess_ms + self.inference_ms + self.postprocess_ms)


class BaseInferenceEngine(ABC):
    """Abstract base class for independent CPU and NPU inference engines."""

    engine_id: str = "base"
    display_name: str = "Base Engine"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when runtime, hardware (if needed), and model are ready."""

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """Return factual readiness, model, runtime, and error diagnostics."""

    @abstractmethod
    def load(self) -> None:
        """Initialize runtime and load model resources."""

    @abstractmethod
    def predict_frame(
        self,
        img_bgr: np.ndarray,
        conf_thres: float | None = None,
        iou_thres: float | None = None,
    ) -> tuple[list[Detection], float]:
        """Run inference on one BGR image and return `(detections, inference_ms)`."""

    def predict_frame_detailed(
        self,
        img_bgr: np.ndarray,
        conf_thres: float | None = None,
        iou_thres: float | None = None,
    ) -> EngineResult:
        """Run inference and return timing + normalized detection payload."""
        import time

        t0 = time.perf_counter()
        dets, inf_ms = self.predict_frame(img_bgr, conf_thres=conf_thres, iou_thres=iou_thres)
        tot_ms = max(inf_ms, (time.perf_counter() - t0) * 1000.0)
        return EngineResult(
            engine_id=self.engine_id,
            engine_name=self.display_name,
            model_name=self.get_status().get("model_name", "unknown"),
            detections=dets,
            inference_ms=float(inf_ms),
            total_ms=float(tot_ms),
        )

    def close(self) -> None:
        """Release any allocated runtime or device resources."""
        return None
