"""Standalone offline preprocessing, YOLOv11 box decoding, and class-wise NMS."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """Single decoded object detection in original image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


def preprocess_bgr(
    img_bgr: np.ndarray,
    imgsz: int = 640,
    pad_value: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Letterbox BGR uint8 frame to NCHW float32 RGB [0, 1] without Ultralytics dependency."""
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
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) * np.float32(1.0 / 255.0)
    nchw = chw[None, ...]
    return nchw, float(scale), (float(dw), float(dh))


def _nms_single_class(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Greedy IoU non-maximum suppression for Nx4 xyxy boxes and N confidence scores."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = np.maximum(1e-6, areas[i] + areas[rest] - inter)
        ious = inter / union
        order = rest[ious <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


def decode_predictions(
    raw_output: np.ndarray,
    orig_shape: tuple[int, int],
    scale: float,
    pad: tuple[float, float],
    class_names: dict[int, str],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> list[Detection]:
    """Decode `(1, 4+nc, 8400)` ExecuTorch output into scaled `Detection` objects via class-wise NMS."""
    arr = np.asarray(raw_output, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    # arr shape: (4 + nc, N) -> transpose to (N, 4 + nc)
    if arr.shape[0] == 4 + len(class_names):
        preds = arr.T
    else:
        preds = arr

    boxes_cxcywh = preds[:, :4]
    cls_scores = preds[:, 4:]
    cls_ids = np.argmax(cls_scores, axis=1)
    confs = cls_scores[np.arange(len(cls_scores)), cls_ids]

    mask = confs >= conf_thres
    if not np.any(mask):
        return []

    boxes_cxcywh = boxes_cxcywh[mask]
    cls_ids = cls_ids[mask]
    confs = confs[mask]

    # Pre-NMS topk cap for speed on dense scenes
    if len(confs) > 3000:
        topk = np.argpartition(confs, -3000)[-3000:]
        boxes_cxcywh = boxes_cxcywh[topk]
        cls_ids = cls_ids[topk]
        confs = confs[topk]

    cx, cy, bw, bh = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    xyxy = np.stack([cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5], axis=1)

    # Class-wise NMS via coordinate offset trick or per-class loop
    keep_indices: list[int] = []
    for cid in np.unique(cls_ids):
        cls_idx = np.flatnonzero(cls_ids == cid)
        kept = _nms_single_class(xyxy[cls_idx], confs[cls_idx], iou_thres)
        keep_indices.extend(cls_idx[kept].tolist())

    if not keep_indices:
        return []

    keep_arr = np.asarray(keep_indices, dtype=np.int64)
    keep_arr = keep_arr[np.argsort(confs[keep_arr])[::-1]][:max_det]

    dw, dh = pad
    orig_h, orig_w = orig_shape
    sel_xyxy = xyxy[keep_arr].copy()
    sel_xyxy[:, [0, 2]] = (sel_xyxy[:, [0, 2]] - dw) / scale
    sel_xyxy[:, [1, 3]] = (sel_xyxy[:, [1, 3]] - dh) / scale
    sel_xyxy[:, [0, 2]] = np.clip(sel_xyxy[:, [0, 2]], 0.0, float(orig_w))
    sel_xyxy[:, [1, 3]] = np.clip(sel_xyxy[:, [1, 3]], 0.0, float(orig_h))

    detections: list[Detection] = []
    for box, conf, cid in zip(sel_xyxy, confs[keep_arr], cls_ids[keep_arr]):
        cid_i = int(cid)
        detections.append(
            Detection(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
                confidence=float(conf),
                class_id=cid_i,
                class_name=class_names.get(cid_i, str(cid_i)),
            )
        )
    return detections
