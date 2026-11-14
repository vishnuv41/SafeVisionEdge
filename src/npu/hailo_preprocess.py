"""Letterbox preprocessing for Hailo-8 NPU inference matching CPU geometry."""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_for_hailo(
    img_bgr: np.ndarray,
    imgsz: int = 640,
    pad_value: tuple[int, int, int] = (114, 114, 114),
    layout: str = "NHWC",
    dtype: str = "uint8",
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Letterbox BGR image identically to CPU `preprocess_bgr` and format for Hailo vstream."""
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Input frame is empty")

    h, w = img_bgr.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    dw = (imgsz - new_w) / 2.0
    dh = (imgsz - new_h) / 2.0

    if (w, h) != (new_w, new_h):
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = img_bgr

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    if dtype == "float32":
        arr = rgb.astype(np.float32) * np.float32(1.0 / 255.0)
    else:
        arr = np.ascontiguousarray(rgb, dtype=np.uint8)

    if layout.upper() == "NCHW":
        arr = np.ascontiguousarray(arr.transpose(2, 0, 1))[None, ...]
    else:
        arr = np.ascontiguousarray(arr)[None, ...]

    return arr, float(scale), (float(dw), float(dh))
