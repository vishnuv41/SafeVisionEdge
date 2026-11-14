"""CPU inference engine adapter wrapping the existing ExecuTorch + XNNPACK EdgeDetector."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from decode import Detection
from engines.base_engine import BaseInferenceEngine, EngineResult
from infer import EdgeDetector


class CPUInferenceEngine(BaseInferenceEngine):
    """Adapter around the preserved ExecuTorch XNNPACK `EdgeDetector`."""

    engine_id = "cpu"
    display_name = "CPU (ExecuTorch + XNNPACK)"

    def __init__(
        self,
        model_path: Path,
        class_names: dict[int, str],
        prep_cfg: dict[str, Any],
        detector: EdgeDetector | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.class_names = dict(class_names)
        self.prep_cfg = dict(prep_cfg)
        self._detector = detector
        self._lock = threading.Lock()
        self._error: str | None = None

    def load(self) -> None:
        with self._lock:
            if self._detector is not None:
                return
            try:
                self._detector = EdgeDetector(self.model_path, self.class_names, self.prep_cfg)
                self._error = None
            except Exception as exc:
                self._error = str(exc)
                raise

    def is_available(self) -> bool:
        if self._detector is not None:
            return True
        if not self.model_path.is_file():
            return False
        try:
            import executorch.runtime  # noqa: F401
        except Exception:
            return False
        return True

    def get_status(self) -> dict[str, Any]:
        model_exists = self.model_path.is_file()
        size_mb = round(self.model_path.stat().st_size / (1024 ** 2), 2) if model_exists else 0.0
        loaded = self._detector is not None
        status_code = "READY" if loaded or self.is_available() else "CPU_UNAVAILABLE"
        if not model_exists:
            status_code = "CPU_MODEL_MISSING"
        elif self._error and not loaded:
            status_code = "CPU_ERROR"
        return {
            "engine_id": self.engine_id,
            "display_name": self.display_name,
            "available": self.is_available(),
            "loaded": loaded,
            "status_code": status_code,
            "runtime": "ExecuTorch",
            "backend": "XNNPACK",
            "model_name": self.model_path.name,
            "model_path": str(self.model_path),
            "model_exists": model_exists,
            "model_size_mb": size_mb,
            "quantization": "INT8 PT2E",
            "classes": [self.class_names[i] for i in sorted(self.class_names)],
            "error": self._error,
        }

    def predict_frame(
        self,
        img_bgr: np.ndarray,
        conf_thres: float | None = None,
        iou_thres: float | None = None,
    ) -> tuple[list[Detection], float]:
        if self._detector is None:
            self.load()
        assert self._detector is not None
        return self._detector.predict_frame(img_bgr, conf_thres=conf_thres, iou_thres=iou_thres)

    def predict_frame_detailed(
        self,
        img_bgr: np.ndarray,
        conf_thres: float | None = None,
        iou_thres: float | None = None,
    ) -> EngineResult:
        t0 = time.perf_counter()
        dets, inf_ms = self.predict_frame(img_bgr, conf_thres=conf_thres, iou_thres=iou_thres)
        tot_ms = max(inf_ms, (time.perf_counter() - t0) * 1000.0)
        return EngineResult(
            engine_id=self.engine_id,
            engine_name=self.display_name,
            model_name=self.model_path.name,
            detections=dets,
            inference_ms=float(inf_ms),
            total_ms=float(tot_ms),
        )
