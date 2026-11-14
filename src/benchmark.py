"""Edge CPU / Raspberry Pi 5 benchmark for SafeVision ExecuTorch (.pte) model."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch

SRC_DIR = Path(__file__).resolve().parent
PKG_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from decode import decode_predictions, preprocess_bgr  # noqa: E402
from infer import load_configs  # noqa: E402
from safety_pulse import analyze_frame, load_safety_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark SafeVision ExecuTorch .pte model.")
    parser.add_argument("--model", type=Path, default=PKG_ROOT / "models" / "custom_model.pte")
    parser.add_argument("--config-dir", type=Path, default=PKG_ROOT / "config")
    parser.add_argument("--image", type=Path, default=None, help="Optional real road image for end-to-end timing")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--out-json", type=Path, default=PKG_ROOT / "outputs" / "benchmark.json")
    args = parser.parse_args()

    from executorch.runtime import Runtime

    class_names, prep_cfg, zone_analyzer = load_configs(args.config_dir)
    safety_cfg = load_safety_config(args.config_dir / "safety_config.json")
    imgsz = int(prep_cfg["input_shape"][2])
    pad_rgb = prep_cfg.get("letterbox_pad_rgb", [114, 114, 114])
    pad_bgr = (int(pad_rgb[2]), int(pad_rgb[1]), int(pad_rgb[0]))

    if args.image and args.image.exists():
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise RuntimeError(f"Could not read image: {args.image}")
    else:
        frame = np.full((720, 1280, 3), 120, dtype=np.uint8)

    rt = Runtime.get()
    prog = rt.load_program(str(args.model))
    method = prog.load_method("forward")

    proc = psutil.Process()
    proc.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None)

    # 1. Pure ExecuTorch forward pass benchmark (warmup discarded)
    inp_np, _, _ = preprocess_bgr(frame, imgsz=imgsz, pad_value=pad_bgr)
    inp_t = torch.from_numpy(inp_np).contiguous()
    for _ in range(args.warmup):
        _ = method.execute([inp_t])

    fwd_times: list[float] = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        _ = method.execute([inp_t])
        fwd_times.append((time.perf_counter() - t0) * 1000.0)

    # 2. Full end-to-end (preprocess + runtime forward + decode + NMS + zone + Safety Pulse)
    e2e_times: list[float] = []
    last_safety: dict[str, object] = {}
    for _ in range(args.runs):
        t0 = time.perf_counter()
        x_np, sc, pd = preprocess_bgr(frame, imgsz=imgsz, pad_value=pad_bgr)
        raw = method.execute([torch.from_numpy(x_np).contiguous()])[0].detach().cpu().numpy()
        dets = decode_predictions(
            raw,
            frame.shape[:2],
            sc,
            pd,
            class_names,
            conf_thres=float(prep_cfg["default_conf_threshold"]),
            iou_thres=float(prep_cfg["default_iou_threshold"]),
            max_det=int(prep_cfg["max_detections"]),
        )
        last_safety = analyze_frame(dets, frame.shape[:2], zone_analyzer, cfg=safety_cfg)
        e2e_times.append((time.perf_counter() - t0) * 1000.0)

    rss_mb = proc.memory_info().rss / (1024 ** 2)
    proc_cpu = proc.cpu_percent(interval=None)
    sys_cpu = psutil.cpu_percent(interval=None)
    e2e_mean = float(np.mean(e2e_times))
    fwd_mean = float(np.mean(fwd_times))

    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "is_raspberry_pi": "aarch64" in platform.machine().lower() or "arm" in platform.machine().lower(),
        "model_path": str(args.model),
        "model_size_mb": round(args.model.stat().st_size / (1024 ** 2), 3),
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "forward_only_ms": {
            "mean": round(fwd_mean, 2),
            "median": round(float(np.median(fwd_times)), 2),
            "p95": round(float(np.percentile(fwd_times, 95)), 2),
            "fps": round(1000.0 / fwd_mean, 2),
        },
        "end_to_end_ms": {
            "mean": round(e2e_mean, 2),
            "median": round(float(np.median(e2e_times)), 2),
            "p95": round(float(np.percentile(e2e_times, 95)), 2),
            "fps": round(1000.0 / e2e_mean, 2),
        },
        "process_rss_mb": round(rss_mb, 1),
        "process_cpu_percent": round(proc_cpu, 1),
        "system_cpu_percent": round(sys_cpu, 1),
        "sample_safety_snapshot": {
            "counts": last_safety.get("counts", {}),
            "density_class": last_safety.get("density_class"),
            "co_presence": last_safety.get("co_presence"),
            "vru_share": last_safety.get("vru_share"),
            "safety_pulse": last_safety.get("safety_pulse"),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
