"""Prepare Hailo Dataflow Compiler NHWC calibration dataset from images or NCHW .npy."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox_rgb_uint8(img_bgr: np.ndarray, imgsz: int = 640) -> np.ndarray:
    """Letterbox BGR image to (imgsz, imgsz, 3) RGB uint8 matching SafeVision CPU geometry."""
    h, w = img_bgr.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    dw = (imgsz - new_w) / 2.0
    dh = (imgsz - new_h) / 2.0

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR) if (w, h) != (new_w, new_h) else img_bgr
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)


def from_nchw_npy(src_npy: Path, out_npy: Path, max_samples: int = 256) -> tuple[int, ...]:
    """Convert existing NCHW float32 [0,1] calibration array (e.g. calib_set_1024x640x640.npy) to NHWC uint8."""
    arr = np.load(str(src_npy), mmap_mode="r")
    n = min(max_samples, int(arr.shape[0]))
    out = np.empty((n, int(arr.shape[2]), int(arr.shape[3]), 3), dtype=np.uint8)
    for i in range(n):
        sample = arr[i]
        if sample.dtype != np.uint8:
            # Assume [0, 1] float32 or [0, 255] float32
            scale = 255.0 if float(sample.max()) <= 1.05 else 1.0
            sample = np.clip(np.round(sample * scale), 0, 255).astype(np.uint8)
        if sample.shape[0] == 3:
            sample = np.transpose(sample, (1, 2, 0))
        out[i] = sample
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_npy), out)
    return tuple(out.shape)


def from_image_dir(img_dir: Path, out_npy: Path, imgsz: int = 640, max_samples: int = 256) -> tuple[int, ...]:
    """Build NHWC uint8 calibration tensor from a directory of road images."""
    paths = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)[:max_samples]
    if not paths:
        raise FileNotFoundError(f"No calibration images found in {img_dir}")
    samples: list[np.ndarray] = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is not None:
            samples.append(letterbox_rgb_uint8(im, imgsz=imgsz))
    out = np.stack(samples, axis=0).astype(np.uint8)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_npy), out)
    return tuple(out.shape)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare NHWC uint8 calibration .npy for Hailo Dataflow Compiler.")
    parser.add_argument("--source", type=Path, required=True, help="Image directory OR existing NCHW .npy calibration file.")
    parser.add_argument("--out", type=Path, default=Path("conversion/calib_nhwc_uint8.npy"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-samples", type=int, default=256)
    args = parser.parse_args()

    if args.source.is_file() and args.source.suffix.lower() == ".npy":
        shape = from_nchw_npy(args.source, args.out, max_samples=args.max_samples)
    elif args.source.is_dir():
        shape = from_image_dir(args.source, args.out, imgsz=args.imgsz, max_samples=args.max_samples)
    else:
        raise FileNotFoundError(f"Unsupported calibration source: {args.source}")

    print(f"Saved Hailo NHWC uint8 calibration dataset: {args.out} shape={shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
