"""End-to-end Hailo-8 inference runner combining preprocess, device NPU call, and decode."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from decode import Detection
from npu.hailo_decoder import decode_hailo_output
from npu.hailo_device import HailoDeviceSession
from npu.hailo_postprocess import filter_and_scale_detections
from npu.hailo_preprocess import preprocess_for_hailo


class HailoInferenceRunner:
    """Executes single-frame YOLOv11-S inference on Hailo-8 via HailoRT."""

    def __init__(
        self,
        hef_path: Path,
        class_names: dict[int, str],
        prep_cfg: dict[str, Any],
        session: HailoDeviceSession | None = None,
    ) -> None:
        self.hef_path = Path(hef_path)
        self.class_names = dict(class_names)
        self.prep_cfg = dict(prep_cfg)
        self.imgsz = int(prep_cfg.get("input_shape", [1, 3, 640, 640])[2])
        pad_rgb = prep_cfg.get("letterbox_pad_rgb", [114, 114, 114])
        self.pad_bgr = (int(pad_rgb[2]), int(pad_rgb[1]), int(pad_rgb[0]))
        self.default_conf = float(prep_cfg.get("default_conf_threshold", 0.25))
        self.default_iou = float(prep_cfg.get("default_iou_threshold", 0.45))
        self.max_det = int(prep_cfg.get("max_detections", 300))
        self.session = session or HailoDeviceSession(self.hef_path)

    def load(self) -> None:
        self.session.load()

    def predict_frame(
        self,
        img_bgr: np.ndarray,
        conf_thres: float | None = None,
        iou_thres: float | None = None,
    ) -> tuple[list[Detection], float, dict[str, float]]:
        """Run preprocess -> HailoRT infer -> decode/NMS; return `(detections, inf_ms, timings)`."""
        conf = self.default_conf if conf_thres is None else float(conf_thres)
        iou = self.default_iou if iou_thres is None else float(iou_thres)

        if not self.session.is_loaded:
            self.session.load()

        t_pre0 = time.perf_counter()
        inp_arr, scale, pad = preprocess_for_hailo(
            img_bgr,
            imgsz=self.imgsz,
            pad_value=self.pad_bgr,
            layout=self.session.input_layout,
            dtype=self.session.input_dtype,
        )
        preprocess_ms = (time.perf_counter() - t_pre0) * 1000.0

        t_inf0 = time.perf_counter()
        raw_outputs = self.session.infer(inp_arr)
        inference_ms = (time.perf_counter() - t_inf0) * 1000.0

        t_post0 = time.perf_counter()
        dets = decode_hailo_output(
            outputs=raw_outputs,
            orig_shape=img_bgr.shape[:2],
            scale=scale,
            pad=pad,
            class_names=self.class_names,
            imgsz=self.imgsz,
            conf_thres=conf,
            iou_thres=iou,
            max_det=self.max_det,
        )
        dets = filter_and_scale_detections(dets, img_bgr.shape[:2], conf_thres=conf, max_det=self.max_det)
        postprocess_ms = (time.perf_counter() - t_post0) * 1000.0

        timings = {
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": preprocess_ms + inference_ms + postprocess_ms,
        }
        return dets, inference_ms, timings

    def close(self) -> None:
        self.session.close()
