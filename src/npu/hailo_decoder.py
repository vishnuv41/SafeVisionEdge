"""Hailo-8 output tensor decoding supporting raw YOLOv11 tensors and Hailo NMS outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from decode import Detection, decode_predictions


def _looks_like_hailo_nms(raw_output: Any, num_classes: int) -> bool:
    """Detect HailoRT on-chip NMS structure (list of per-class Nx5 arrays)."""
    if isinstance(raw_output, (list, tuple)) and len(raw_output) == num_classes:
        return True
    if isinstance(raw_output, np.ndarray) and raw_output.dtype == object and raw_output.shape[0] == num_classes:
        return True
    return False


def _decode_hailo_nms(
    per_class_boxes: Any,
    orig_shape: tuple[int, int],
    imgsz: int,
    scale: float,
    pad: tuple[float, float],
    class_names: dict[int, str],
    conf_thres: float,
    max_det: int,
) -> list[Detection]:
    """Decode Hailo NMS `[y_min, x_min, y_max, x_max, score]` normalized to [0,1] on letterboxed input."""
    dw, dh = pad
    orig_h, orig_w = orig_shape
    candidates: list[Detection] = []

    for cid, cls_arr in enumerate(per_class_boxes):
        arr = np.asarray(cls_arr, dtype=np.float32)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 5:
            continue
        for row in arr:
            score = float(row[4])
            if score < conf_thres:
                continue
            # Hailo NMS outputs [ymin, xmin, ymax, xmax] normalized in [0, 1] wrt model imgsz
            ymin, xmin, ymax, xmax = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            if max(ymin, xmin, ymax, xmax) <= 1.5:
                xmin *= imgsz
                xmax *= imgsz
                ymin *= imgsz
                ymax *= imgsz
            x1 = max(0.0, min(float(orig_w), (xmin - dw) / scale))
            y1 = max(0.0, min(float(orig_h), (ymin - dh) / scale))
            x2 = max(0.0, min(float(orig_w), (xmax - dw) / scale))
            y2 = max(0.0, min(float(orig_h), (ymax - dh) / scale))
            if x2 <= x1 or y2 <= y1:
                continue
            candidates.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=score,
                    class_id=int(cid),
                    class_name=class_names.get(int(cid), str(cid)),
                )
            )

    candidates.sort(key=lambda d: d.confidence, reverse=True)
    return candidates[:max_det]


def decode_hailo_output(
    outputs: dict[str, Any] | np.ndarray | list[Any],
    orig_shape: tuple[int, int],
    scale: float,
    pad: tuple[float, float],
    class_names: dict[int, str],
    imgsz: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> list[Detection]:
    """Decode Hailo InferVStreams output into standard `Detection` objects."""
    if isinstance(outputs, dict):
        if not outputs:
            return []
        values = list(outputs.values())
        primary = values[0]
    else:
        primary = outputs

    num_classes = len(class_names)

    # Unwrap batch=1 list if HailoRT returns [batch_0_per_class]
    if isinstance(primary, list) and len(primary) == 1 and _looks_like_hailo_nms(primary[0], num_classes):
        primary = primary[0]

    if _looks_like_hailo_nms(primary, num_classes):
        return _decode_hailo_nms(
            per_class_boxes=primary,
            orig_shape=orig_shape,
            imgsz=imgsz,
            scale=scale,
            pad=pad,
            class_names=class_names,
            conf_thres=conf_thres,
            max_det=max_det,
        )

    arr = np.asarray(primary, dtype=np.float32)
    # Handle NHWC or (1, 8400, 11) -> decode_predictions handles (1, 11, 8400) or (8400, 11)
    if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 1:
        arr = arr.reshape(arr.shape[2], arr.shape[3])
    return decode_predictions(
        raw_output=arr,
        orig_shape=orig_shape,
        scale=scale,
        pad=pad,
        class_names=class_names,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
    )
