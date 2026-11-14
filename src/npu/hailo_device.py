"""HailoRT device session lifecycle and HEF execution wrapper."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from npu.hailo_status import HailoStatusCode, check_hailo_status


class HailoUnavailableError(RuntimeError):
    """Raised when Hailo-8 hardware, driver, or HailoRT runtime is unavailable."""


class HailoModelNotReadyError(RuntimeError):
    """Raised when the compiled `.hef` model file is missing or invalid."""


class HailoInferenceError(RuntimeError):
    """Raised when Hailo-8 inference fails during execution."""


class HailoDeviceSession:
    """Manages HailoRT VDevice, HEF loading, InferVStreams, and clean resource release."""

    def __init__(self, hef_path: Path, hailort_module: Any | None = None) -> None:
        self.hef_path = Path(hef_path)
        self._hailort_override = hailort_module
        self._lock = threading.Lock()
        self._loaded = False

        self._vdevice: Any = None
        self._hef: Any = None
        self._network_group: Any = None
        self._network_group_ctx: Any = None
        self._infer_pipeline: Any = None
        self._infer_ctx: Any = None
        self._input_vstream_name: str = ""
        self._input_shape: tuple[int, ...] = (640, 640, 3)
        self._input_format_type: str = "uint8"
        self._output_names: list[str] = []
        self._output_shapes: dict[str, tuple[int, ...]] = {}

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def input_layout(self) -> str:
        # Hailo vstream input shapes are typically (H, W, C) or (C, H, W)
        if len(self._input_shape) == 3 and self._input_shape[0] == 3 and self._input_shape[2] != 3:
            return "NCHW"
        return "NHWC"

    @property
    def input_dtype(self) -> str:
        return self._input_format_type

    def _resolve_hailort(self) -> Any:
        if self._hailort_override is not None:
            return self._hailort_override
        status = check_hailo_status(self.hef_path)
        code = status["status_code"]
        if code == HailoStatusCode.NPU_HARDWARE_UNAVAILABLE:
            raise HailoUnavailableError(status["message"])
        if code == HailoStatusCode.HAILORT_UNAVAILABLE:
            raise HailoUnavailableError(status["message"])
        if code == HailoStatusCode.NPU_MODEL_NOT_READY:
            raise HailoModelNotReadyError(status["message"])
        import hailo_platform  # type: ignore[import-not-found]

        return hailo_platform

    def load(self) -> None:
        """Open Hailo VDevice, configure HEF network group, and activate InferVStreams."""
        with self._lock:
            if self._loaded:
                return
            if not self.hef_path.is_file():
                raise HailoModelNotReadyError(
                    f"Compiled Hailo HEF model not found: {self.hef_path} ({HailoStatusCode.NPU_MODEL_NOT_READY})"
                )
            hp = self._resolve_hailort()
            try:
                self._hef = hp.HEF(str(self.hef_path))
                self._vdevice = hp.VDevice()
                cfg_params = hp.ConfigureParams.create_from_hef(
                    hef=self._hef, interface=hp.HailoStreamInterface.PCIe
                )
                network_groups = self._vdevice.configure(self._hef, cfg_params)
                self._network_group = network_groups[0]
                ng_params = self._network_group.create_params()

                in_infos = self._hef.get_input_vstream_infos()
                out_infos = self._hef.get_output_vstream_infos()
                if not in_infos or not out_infos:
                    raise HailoModelNotReadyError("HEF contains no input or output vstreams")

                self._input_vstream_name = str(in_infos[0].name)
                self._input_shape = tuple(int(x) for x in in_infos[0].shape)
                fmt_str = str(getattr(in_infos[0].format, "type", "UINT8")).lower()
                self._input_format_type = "float32" if "float" in fmt_str else "uint8"
                self._output_names = [str(o.name) for o in out_infos]
                self._output_shapes = {str(o.name): tuple(int(x) for x in o.shape) for o in out_infos}

                in_type = hp.FormatType.FLOAT32 if self._input_format_type == "float32" else hp.FormatType.UINT8
                in_params = hp.InputVStreamParams.make(self._network_group, quantized=(in_type == hp.FormatType.UINT8), format_type=in_type)
                out_params = hp.OutputVStreamParams.make(self._network_group, quantized=False, format_type=hp.FormatType.FLOAT32)

                self._network_group_ctx = self._network_group.activate(ng_params)
                self._network_group_ctx.__enter__()
                self._infer_ctx = hp.InferVStreams(self._network_group, in_params, out_params)
                self._infer_pipeline = self._infer_ctx.__enter__()
                self._loaded = True
            except (HailoUnavailableError, HailoModelNotReadyError):
                self.close()
                raise
            except Exception as exc:
                self.close()
                raise HailoInferenceError(f"Failed to initialize Hailo-8 HEF session: {exc}") from exc

    def infer(self, input_tensor: np.ndarray) -> dict[str, Any]:
        """Run synchronous inference on the Hailo-8 device and return output dict."""
        if not self._loaded or self._infer_pipeline is None:
            self.load()
        assert self._infer_pipeline is not None
        try:
            with self._lock:
                outputs = self._infer_pipeline.infer({self._input_vstream_name: input_tensor})
            if not isinstance(outputs, dict):
                raise HailoInferenceError(f"Unexpected Hailo InferVStreams return type: {type(outputs)}")
            return outputs
        except HailoInferenceError:
            raise
        except Exception as exc:
            raise HailoInferenceError(f"Hailo-8 inference execution failed: {exc}") from exc

    def metadata(self) -> dict[str, Any]:
        return {
            "loaded": self._loaded,
            "hef_path": str(self.hef_path),
            "input_vstream": self._input_vstream_name,
            "input_shape": list(self._input_shape),
            "input_layout": self.input_layout,
            "input_dtype": self._input_format_type,
            "output_vstreams": self._output_names,
            "output_shapes": {k: list(v) for k, v in self._output_shapes.items()},
        }

    def close(self) -> None:
        """Release InferVStreams, network group activation, and VDevice cleanly."""
        if self._infer_ctx is not None:
            try:
                self._infer_ctx.__exit__(None, None, None)
            except Exception:
                pass
        if self._network_group_ctx is not None:
            try:
                self._network_group_ctx.__exit__(None, None, None)
            except Exception:
                pass
        if self._vdevice is not None and hasattr(self._vdevice, "release"):
            try:
                self._vdevice.release()
            except Exception:
                pass
        self._infer_ctx = None
        self._infer_pipeline = None
        self._network_group_ctx = None
        self._network_group = None
        self._hef = None
        self._vdevice = None
        self._loaded = False

    def __enter__(self) -> "HailoDeviceSession":
        self.load()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
