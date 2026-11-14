#!/usr/bin/env python3
"""SafeVision Edge — Standalone Hailo-8 NPU Inference Verification Script."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np

from npu.hailo_device import HailoDeviceSession
from npu.hailo_inference import HailoInferenceRunner
from npu.hailo_status import HailoStatusCode, check_hailo_status
from infer import load_configs


def main() -> int:
    parser = argparse.ArgumentParser(description="Test physical Hailo-8 inference on road images.")
    parser.add_argument("--hef", type=Path, default=ROOT_DIR / "models" / "npu" / "custom_model.hef")
    parser.add_argument("--img", type=Path, default=ROOT_DIR / "tests" / "fixtures" / "sample_road_1.jpg")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    hef_path = args.hef

    print("==================================================================")
    print(" SafeVision Edge — Hailo-8 Physical Hardware Inference Test")
    print("==================================================================")
    print(f"Target HEF Model : {hef_path}")
    print(f"Test Road Image  : {args.img}")

    # Check hardware & model status first
    status = check_hailo_status(hef_path)
    print(f"Hailo Status Code: {status['status_code']}")
    print(f"Status Message   : {status['message']}")
    print(f"Device Connected : {status['device_connected']}")
    print(f"HailoRT Python   : {status['hailort_available']}")
    print(f"Model Exists     : {status['model_exists']}")

    if not status["available"]:
        print(f"\n[BLOCKED] NPU is not in READY state: {status['status_code']}.")
        print("Detailed diagnostics:")
        for k, v in status.items():
            print(f"  {k}: {v}")
        return 1

    if not args.img.is_file():
        print(f"ERROR: Test image missing: {args.img}")
        return 1

    img_bgr = cv2.imread(str(args.img))
    if img_bgr is None:
        print(f"ERROR: Failed to decode image: {args.img}")
        return 1

    class_names, prep_cfg, _ = load_configs(ROOT_DIR / "config")
    session = HailoDeviceSession(hef_path)
    runner = HailoInferenceRunner(
        hef_path=hef_path,
        class_names=class_names,
        prep_cfg=prep_cfg,
        session=session,
    )

    print("\nInitializing Hailo-8 session...")
    t0 = time.perf_counter()
    session.load()
    t_load = (time.perf_counter() - t0) * 1000
    print(f"[OK] Session loaded in {t_load:.1f} ms.")
    print("Session Metadata:", session.metadata())

    try:
        print("\nExecuting inference on test image...")
        dets, timings = runner.predict(img_bgr, conf_thres=args.conf, iou_thres=args.iou)
        print("=== INFERENCE RESULTS ===")
        print(f"Detections Count : {len(dets)}")
        print(f"Timing Breakdown : Preprocess={timings['preprocess_ms']:.1f} ms | "
              f"Inference={timings['inference_ms']:.1f} ms | "
              f"Postprocess={timings['postprocess_ms']:.1f} ms | "
              f"Total={timings['total_ms']:.1f} ms")

        counts = {}
        for d in dets:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        print("Counts by Class  :", counts)

        print("\nTop Detections:")
        for i, d in enumerate(dets[:10]):
            print(f"  [{i}] {d.class_name} (conf={d.confidence:.3f}) at [{d.x1:.1f}, {d.y1:.1f}, {d.x2:.1f}, {d.y2:.1f}]")

        print("\n[SUCCESS] Physical Hailo-8 inference executed and verified.")
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
