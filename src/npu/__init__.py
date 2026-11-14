"""Hailo-8 NPU inference and diagnostics package for SafeVision Edge."""

from npu.hailo_decoder import decode_hailo_output
from npu.hailo_device import (
    HailoDeviceSession,
    HailoInferenceError,
    HailoModelNotReadyError,
    HailoUnavailableError,
)
from npu.hailo_inference import HailoInferenceRunner
from npu.hailo_postprocess import detections_to_normalized, filter_and_scale_detections
from npu.hailo_preprocess import preprocess_for_hailo
from npu.hailo_status import HailoStatusCode, check_hailo_status, read_hailo_temperature_c

__all__ = [
    "HailoDeviceSession",
    "HailoInferenceError",
    "HailoInferenceRunner",
    "HailoModelNotReadyError",
    "HailoStatusCode",
    "HailoUnavailableError",
    "check_hailo_status",
    "decode_hailo_output",
    "detections_to_normalized",
    "filter_and_scale_detections",
    "preprocess_for_hailo",
    "read_hailo_temperature_c",
]
