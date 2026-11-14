"""Hailo-8 NPU inference engine adapter implementing `BaseInferenceEngine`."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from decode import Detection
from engines.base_engine import BaseInferenceEngine, EngineResult
from npu.hailo_device import HailoDeviceSession
from npu.hailo_inference import HailoInferenceRunner
from npu.hailo_status import HailoStatusCode, check_hailo_status


class HailoInferenceEngine(BaseInferenceEngine):
    """Independent Hailo-8 NPU inference engine using HailoRT."""

    engine_id = "npu"
    display_name = "Hailo-8 NPU (HailoRT)"

    def __init__(
        self,
        hef_path: Path,
        class_names: dict[int, str],
        prep_cfg: dict[str, Any],
        runner: HailoInferenceRunner | None = None,
        status_probe: Callable[[Path], dict[str, Any]] | None = None,
    ) -> None:
        self.hef_path = Path(hef_path)
        self.class_names = dict(class_names)
        self.prep_cfg = dict(prep_cfg)
        self._runner = runner
        self._status_probe = status_probe
        self._lock = threading.Lock()
        self._last_error: str | None = None

    def _probe_status(self) -> dict[str, Any]:
        if self._status_probe is not None:
            return self._status_probe(self.hef_path)
        return check_hailo_status(self.hef_path)

    def _ensure_runner(self) -> HailoInferenceRunner:
        if self._runner is None:
            session = HailoDeviceSession(self.hef_path)
            self._runner = HailoInferenceRunner(
                hef_path=self.hef_path,
                class_names=self.class_names,
                prep_cfg=self.prep_cfg,
                session=session,
            )
        return self._runner

    def is_available(self) -> bool:
        if self._runner is not None and self._runner.session.is_loaded:
            return True
        return bool(self._probe_status().get("available", False))

    def get_status(self) -> dict[str, Any]:
        if self._runner is not None and self._runner.session.is_loaded and self._status_probe is None:
            # Session is already active (e.g. injected mock runner in tests)
            meta = self._runner.session.metadata()
            return {
                "engine_id": self.engine_id,
                "display_name": self.display_name,
                "available": True,
                "loaded": True,
                "status_code": HailoStatusCode.READY,
                "message": f"Hailo-8 NPU session active ({self.hef_path.name}).",
                "runtime": "HailoRT",
                "backend": "Hailo-8",
                "device_connected": True,
                "driver_loaded": True,
                "hailort_available": True,
                "firmware_available": True,
                "model_exists": True,
                "model_status": "ready",
                "model_name": self.hef_path.name,
                "model_path": str(self.hef_path),
                "classes": [self.class_names[i] for i in sorted(self.class_names)],
                "session_metadata": meta,
                "error": self._last_error,
            }

        st = self._probe_status()
        loaded = self._runner is not None and self._runner.session.is_loaded
        return {
            "engine_id": self.engine_id,
            "display_name": self.display_name,
            "available": bool(st.get("available", False)),
            "loaded": loaded,
            "status_code": st.get("status_code", HailoStatusCode.NPU_HARDWARE_UNAVAILABLE),
            "message": st.get("message", ""),
            "runtime": "HailoRT",
            "backend": "Hailo-8",
            "device_connected": bool(st.get("device_connected", False)),
            "device_nodes": st.get("device_nodes", []),
            "driver_loaded": bool(st.get("driver_loaded", False)),
            "hailort_available": bool(st.get("hailort_available", False)),
            "hailort_version": st.get("hailort_version"),
            "firmware_available": bool(st.get("firmware_available", False)),
            "firmware_version": st.get("firmware_version"),
            "device_architecture": st.get("device_architecture"),
            "temperature_c": st.get("temperature_c"),
            "model_status": st.get("model_status", HailoStatusCode.NPU_MODEL_NOT_READY),
            "model_exists": bool(st.get("model_exists", False)),
            "model_name": st.get("model_name", self.hef_path.name),
            "model_path": st.get("model_path", str(self.hef_path)),
            "model_size_mb": st.get("model_size_mb", 0.0),
            "classes": [self.class_names[i] for i in sorted(self.class_names)],
            "error": self._last_error or st.get("hailort_error"),
        }

    def load(self) -> None:
        with self._lock:
            runner = self._ensure_runner()
            try:
                runner.load()
                self._last_error = None
            except Exception as exc:
                self._last_error = str(exc)
                raise

    def predict_frame(
        self,
        img_bgr: np.ndarray,
        conf_thres: float | None = None,
        iou_thres: float | None = None,
    ) -> tuple[list[Detection], float]:
        runner = self._ensure_runner()
        try:
            dets, inf_ms, _ = runner.predict_frame(img_bgr, conf_thres=conf_thres, iou_thres=iou_thres)
            self._last_error = None
            return dets, inf_ms
        except Exception as exc:
            self._last_error = str(exc)
            raise

    def predict_frame_detailed(
        self,
        img_bgr: np.ndarray,
        conf_thres: float | None = None,
        iou_thres: float | None = None,
    ) -> EngineResult:
        runner = self._ensure_runner()
        t0 = time.perf_counter()
        try:
            dets, inf_ms, timings = runner.predict_frame(img_bgr, conf_thres=conf_thres, iou_thres=iou_thres)
            self._last_error = None
            tot_ms = max(timings.get("total_ms", inf_ms), (time.perf_counter() - t0) * 1000.0)
            return EngineResult(
                engine_id=self.engine_id,
                engine_name=self.display_name,
                model_name=self.hef_path.name,
                detections=dets,
                inference_ms=float(inf_ms),
                preprocess_ms=float(timings.get("preprocess_ms", 0.0)),
                postprocess_ms=float(timings.get("postprocess_ms", 0.0)),
                total_ms=float(tot_ms),
            )
        except Exception as exc:
            self._last_error = str(exc)
            raise

    def close(self) -> None:
        with self._lock:
            if self._runner is not None:
                self._runner.close()
