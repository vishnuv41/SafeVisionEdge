"""Inference engine abstraction layer for SafeVision Edge (CPU, NPU, Compare)."""

from engines.base_engine import (
    ALLOWED_ENGINE_MODES,
    BaseInferenceEngine,
    EngineMode,
    EngineResult,
    detection_to_dict,
    dict_to_detection,
)
from engines.cpu_engine import CPUInferenceEngine
from engines.engine_manager import EngineManager
from engines.hailo_engine import HailoInferenceEngine

__all__ = [
    "ALLOWED_ENGINE_MODES",
    "BaseInferenceEngine",
    "CPUInferenceEngine",
    "EngineManager",
    "EngineMode",
    "EngineResult",
    "HailoInferenceEngine",
    "detection_to_dict",
    "dict_to_detection",
]
