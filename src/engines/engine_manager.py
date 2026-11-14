"""Engine Manager enforcing strictly three modes: cpu, npu, compare (no auto mode)."""

from __future__ import annotations

import threading
from typing import Any

from engines.base_engine import ALLOWED_ENGINE_MODES, BaseInferenceEngine, EngineMode


class InvalidEngineModeError(ValueError):
    """Raised when an unsupported inference mode (including 'auto') is requested."""


class EngineManager:
    """Coordinates CPU and Hailo-8 NPU inference engines without silent fallback."""

    def __init__(
        self,
        cpu_engine: BaseInferenceEngine,
        npu_engine: BaseInferenceEngine,
        initial_mode: str = EngineMode.CPU,
    ) -> None:
        self._lock = threading.Lock()
        self.cpu_engine = cpu_engine
        self.npu_engine = npu_engine
        self._selected_mode = self.validate_mode(initial_mode)

    @staticmethod
    def validate_mode(mode: str) -> str:
        norm = str(mode or "").strip().lower()
        if norm not in ALLOWED_ENGINE_MODES:
            raise InvalidEngineModeError(
                f"Invalid engine mode '{mode}'. Allowed modes are strictly {list(ALLOWED_ENGINE_MODES)} (no auto mode)."
            )
        return norm

    @property
    def selected_mode(self) -> str:
        with self._lock:
            return self._selected_mode

    def select_mode(self, mode: str) -> dict[str, Any]:
        validated = self.validate_mode(mode)
        with self._lock:
            self._selected_mode = validated
        return self.get_status()

    def get_engines(self) -> dict[str, Any]:
        cpu_st = self.cpu_engine.get_status()
        npu_st = self.npu_engine.get_status()
        return {
            "selected_mode": self.selected_mode,
            "allowed_modes": [
                {
                    "id": EngineMode.CPU,
                    "label": "CPU",
                    "description": "ExecuTorch + XNNPACK INT8 PT2E (.pte)",
                    "available": bool(cpu_st.get("available")),
                    "model_name": cpu_st.get("model_name"),
                },
                {
                    "id": EngineMode.NPU,
                    "label": "NPU",
                    "description": "Hailo-8 AI HAT+ via HailoRT (.hef)",
                    "available": bool(npu_st.get("available")),
                    "model_name": npu_st.get("model_name"),
                    "status_code": npu_st.get("status_code"),
                },
                {
                    "id": EngineMode.COMPARE,
                    "label": "Compare Both",
                    "description": "Run CPU and Hailo-8 NPU on identical frames and compare latency & detections",
                    "available": bool(cpu_st.get("available") and npu_st.get("available")),
                    "requires_both": True,
                },
            ],
            "engines": {
                "cpu": cpu_st,
                "npu": npu_st,
            },
        }

    def get_status(self) -> dict[str, Any]:
        cpu_st = self.cpu_engine.get_status()
        npu_st = self.npu_engine.get_status()
        mode = self.selected_mode
        if mode == EngineMode.CPU:
            selected_ready = bool(cpu_st.get("available"))
            selected_message = "CPU ExecuTorch + XNNPACK engine ready." if selected_ready else cpu_st.get("error", "CPU engine unavailable")
        elif mode == EngineMode.NPU:
            selected_ready = bool(npu_st.get("available"))
            selected_message = npu_st.get("message", "Hailo-8 NPU status checked.")
        else:
            selected_ready = bool(cpu_st.get("available") and npu_st.get("available"))
            if selected_ready:
                selected_message = "Both CPU and Hailo-8 NPU engines are ready for comparison."
            else:
                selected_message = (
                    f"Compare mode requires both engines; NPU status: {npu_st.get('status_code')} — {npu_st.get('message')}"
                )
        return {
            "selected_mode": mode,
            "selected_ready": selected_ready,
            "selected_message": selected_message,
            "auto_mode_supported": False,
            "cpu": cpu_st,
            "npu": npu_st,
        }

    def require_ready_for_mode(self, mode: str | None = None) -> str:
        target = self.validate_mode(mode or self.selected_mode)
        if target == EngineMode.CPU:
            if not self.cpu_engine.is_available():
                st = self.cpu_engine.get_status()
                raise RuntimeError(st.get("error") or f"CPU engine unavailable ({st.get('status_code')})")
        elif target == EngineMode.NPU:
            if not self.npu_engine.is_available():
                st = self.npu_engine.get_status()
                raise RuntimeError(f"{st.get('status_code')}: {st.get('message')}")
        else:  # compare
            if not self.cpu_engine.is_available():
                st = self.cpu_engine.get_status()
                raise RuntimeError(st.get("error") or "CPU engine unavailable for comparison")
            if not self.npu_engine.is_available():
                st = self.npu_engine.get_status()
                raise RuntimeError(f"Compare mode unavailable — {st.get('status_code')}: {st.get('message')}")
        return target
