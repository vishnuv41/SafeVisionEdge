"""Validate compiled Hailo-8 `.hef` against CPU ExecuTorch `.pte` baseline on fixed validation images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

PKG_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PKG_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmarks.comparison_runner import compute_detection_agreement  # noqa: E402
from engines.cpu_engine import CPUInferenceEngine  # noqa: E402
from engines.hailo_engine import HailoInferenceEngine  # noqa: E402
from infer import IMAGE_EXTS, load_configs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Hailo-8 .hef predictions vs CPU .pte baseline.")
    parser.add_argument("--pte", type=Path, default=PKG_ROOT / "models" / "custom_model.pte")
    parser.add_argument("--hef", type=Path, default=PKG_ROOT / "models" / "custom_model.hef")
    parser.add_argument("--val-dir", type=Path, default=PKG_ROOT / "tests" / "fixtures")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--out", type=Path, default=PKG_ROOT / "outputs" / "benchmark" / "hef_validation.json")
    args = parser.parse_args()

    class_names, prep_cfg, _ = load_configs(PKG_ROOT / "config")
    cpu = CPUInferenceEngine(args.pte, class_names, prep_cfg)
    npu = HailoInferenceEngine(args.hef, class_names, prep_cfg)

    npu_status = npu.get_status()
    if not npu_status.get("available"):
        print(json.dumps({"validated": False, "npu_status": npu_status}, indent=2))
        return 1

    imgs = sorted(p for p in args.val_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not imgs:
        raise FileNotFoundError(f"No validation images found in {args.val_dir}")

    cpu.load()
    npu.load()
    try:
        cpu_frames = []
        npu_frames = []
        for p in imgs:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            c_dets, _ = cpu.predict_frame(bgr, conf_thres=args.conf, iou_thres=args.iou)
            n_dets, _ = npu.predict_frame(bgr, conf_thres=args.conf, iou_thres=args.iou)
            cpu_frames.append(c_dets)
            npu_frames.append(n_dets)

        report: dict[str, Any] = {
            "validated": True,
            "pte_model": str(args.pte),
            "hef_model": str(args.hef),
            "images_tested": [p.name for p in imgs],
            "agreement": compute_detection_agreement(cpu_frames, npu_frames),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    finally:
        npu.close()


if __name__ == "__main__":
    raise SystemExit(main())
