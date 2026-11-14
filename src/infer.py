"""Offline SafeVision Edge inference CLI using ExecuTorch (.pte) runtime."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import torch

SRC_DIR = Path(__file__).resolve().parent
PKG_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from counter import (  # noqa: E402
    blur_persons,
    build_csv_row,
    render_annotations,
    write_counts_csv,
)
from decode import decode_predictions, preprocess_bgr  # noqa: E402
from safety_pulse import (  # noqa: E402
    PedestrianHeatmapAccumulator,
    analyze_frame,
    generate_safety_brief,
    load_safety_config,
)
from zone_analysis import ZoneAnalyzer  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


class EdgeDetector:
    """Offline ExecuTorch XNNPACK detector wrapper."""

    def __init__(self, pte_path: Path, class_names: dict[int, str], prep_cfg: dict[str, Any]) -> None:
        from executorch.runtime import Runtime

        self.pte_path = Path(pte_path)
        if not self.pte_path.is_file():
            raise FileNotFoundError(f"ExecuTorch model not found: {self.pte_path}")

        self.class_names = class_names
        self.imgsz = int(prep_cfg["input_shape"][2])
        pad_rgb = prep_cfg.get("letterbox_pad_rgb", [114, 114, 114])
        # OpenCV expects BGR tuple
        self.pad_bgr = (int(pad_rgb[2]), int(pad_rgb[1]), int(pad_rgb[0]))
        self.default_conf = float(prep_cfg.get("default_conf_threshold", 0.25))
        self.default_iou = float(prep_cfg.get("default_iou_threshold", 0.45))
        self.max_det = int(prep_cfg.get("max_detections", 300))

        self.runtime = Runtime.get()
        self.program = self.runtime.load_program(str(self.pte_path))
        self.method = self.program.load_method("forward")

        # Warm-up one dummy static frame
        dummy = torch.zeros((1, 3, self.imgsz, self.imgsz), dtype=torch.float32)
        _ = self.method.execute([dummy])

    def predict_frame(
        self, img_bgr: np.ndarray, conf_thres: float | None = None, iou_thres: float | None = None
    ) -> tuple[list[Any], float]:
        """Run preprocess -> ExecuTorch forward (timed separately) -> decode/NMS on one BGR frame."""
        conf = self.default_conf if conf_thres is None else conf_thres
        iou = self.default_iou if iou_thres is None else iou_thres

        inp_np, scale, pad = preprocess_bgr(img_bgr, imgsz=self.imgsz, pad_value=self.pad_bgr)
        inp_tensor = torch.from_numpy(inp_np).contiguous()

        t_inf0 = time.perf_counter()
        raw_out = self.method.execute([inp_tensor])[0].detach().cpu().numpy()
        inference_ms = (time.perf_counter() - t_inf0) * 1000.0

        dets = decode_predictions(
            raw_output=raw_out,
            orig_shape=img_bgr.shape[:2],
            scale=scale,
            pad=pad,
            class_names=self.class_names,
            conf_thres=conf,
            iou_thres=iou,
            max_det=self.max_det,
        )
        return dets, inference_ms


def load_configs(
    config_dir: Path,
) -> tuple[dict[int, str], dict[str, Any], ZoneAnalyzer | None]:
    """Load class names, preprocessing config, and polygonal zone analyzer."""
    names_raw = json.loads((config_dir / "class_names.json").read_text(encoding="utf-8"))
    class_names = {int(k): str(v) for k, v in names_raw.items()}
    prep_cfg = json.loads((config_dir / "preprocessing_config.json").read_text(encoding="utf-8"))
    zones_path = config_dir / "zones.json"
    zone_analyzer = ZoneAnalyzer.from_json(zones_path) if zones_path.exists() else None
    return class_names, prep_cfg, zone_analyzer


def _process_single_frame(
    detector: EdgeDetector,
    frame_bgr: np.ndarray,
    frame_id: str | int,
    frame_label: str,
    zone_analyzer: ZoneAnalyzer | None,
    safety_cfg: dict[str, Any],
    heatmap: PedestrianHeatmapAccumulator,
    conf_thres: float | None,
    iou_thres: float | None,
    blur_privacy: bool,
) -> tuple[np.ndarray, dict[str, object], dict[str, Any], float, float]:
    """Run unified full-frame pipeline shared identically by image and video modes."""
    t_total0 = time.perf_counter()
    dets, inference_ms = detector.predict_frame(frame_bgr, conf_thres=conf_thres, iou_thres=iou_thres)
    safety = analyze_frame(dets, frame_bgr.shape[:2], zone_analyzer, cfg=safety_cfg)
    heatmap.update(dets, frame_bgr.shape[:2])

    # Blur person ROIs BEFORE drawing zone overlays/boxes so annotations stay crisp on saved outputs
    vis = blur_persons(frame_bgr, dets) if blur_privacy else frame_bgr.copy()
    if zone_analyzer is not None:
        vis = zone_analyzer.draw_zones(vis, safety["zone_counts"])

    # Approximate total_ms prior to banner text render so banner matches CSV total_ms closely
    total_ms_pre_draw = (time.perf_counter() - t_total0) * 1000.0
    vis = render_annotations(
        vis,
        dets,
        safety,
        inference_ms,
        total_ms_pre_draw,
        frame_label=frame_label,
        blur_privacy=False,
    )
    total_ms = max(inference_ms, (time.perf_counter() - t_total0) * 1000.0)
    row = build_csv_row(frame_id, safety, inference_ms, total_ms)
    return vis, row, safety["zone_counts"], inference_ms, total_ms


def _finalize_outputs(
    run_name: str,
    detector: EdgeDetector,
    rows: list[dict[str, object]],
    zone_logs: list[dict[str, Any]],
    inf_latencies: list[float],
    tot_latencies: list[float],
    out_csv_path: Path,
    out_root: Path,
    heatmap: PedestrianHeatmapAccumulator,
    safety_cfg: dict[str, Any],
    proc: psutil.Process,
) -> dict[str, Any]:
    """Write CSV, zone log, 3x3 heatmap PNG, and Safety Brief MD/JSON reports."""
    write_counts_csv(rows, out_csv_path)
    if zone_logs:
        out_csv_path.with_suffix(".zones.json").write_text(
            json.dumps(zone_logs, indent=2), encoding="utf-8"
        )

    heatmap_path = out_root / "heatmaps" / f"{run_name}_pedestrian_heatmap.png"
    heatmap.save_heatmap(heatmap_path)

    rss_mb = proc.memory_info().rss / (1024 ** 2)
    cpu_pct = proc.cpu_percent(interval=None)
    brief_md_path = out_root / "briefs" / f"{run_name}_safety_brief.md"
    brief_json_path = out_root / "briefs" / f"{run_name}_safety_brief.json"
    brief = generate_safety_brief(
        rows=rows,
        zone_logs=zone_logs,
        model_path=detector.pte_path,
        brief_md_path=brief_md_path,
        brief_json_path=brief_json_path,
        rss_mb=rss_mb,
        cpu_percent=cpu_pct,
        heatmap_path=heatmap_path,
        disclaimer=str(safety_cfg.get("disclaimer", "")),
    )

    avg_inf = float(np.mean(inf_latencies)) if inf_latencies else 0.0
    avg_tot = float(np.mean(tot_latencies)) if tot_latencies else 0.0
    return {
        "frames_processed": len(rows),
        "avg_inference_ms": round(avg_inf, 2),
        "avg_total_ms": round(avg_tot, 2),
        "effective_fps": round(1000.0 / avg_tot, 2) if avg_tot > 0 else 0.0,
        "model_fps": round(1000.0 / avg_inf, 2) if avg_inf > 0 else 0.0,
        "csv_path": str(out_csv_path),
        "heatmap_path": str(heatmap_path),
        "safety_brief_md": str(brief_md_path),
        "safety_brief_json": str(brief_json_path),
        "peak_safety_pulse": brief["peak_safety_pulse_frame"],
    }


def process_images(
    detector: EdgeDetector,
    image_paths: list[Path],
    out_img_dir: Path,
    out_csv_path: Path,
    zone_analyzer: ZoneAnalyzer | None,
    conf_thres: float | None,
    iou_thres: float | None,
    safety_cfg: dict[str, Any] | None = None,
    blur_privacy: bool | None = None,
    out_root: Path | None = None,
) -> dict[str, Any]:
    """Process one or more BGR image files cleanly offline."""
    cfg = safety_cfg if safety_cfg is not None else load_safety_config(None)
    do_blur = bool(cfg.get("privacy_blur_persons", True)) if blur_privacy is None else blur_privacy
    root = out_root if out_root is not None else out_csv_path.parent.parent

    out_img_dir.mkdir(parents=True, exist_ok=True)
    heatmap = PedestrianHeatmapAccumulator()
    rows: list[dict[str, object]] = []
    zone_logs: list[dict[str, Any]] = []
    inf_latencies: list[float] = []
    tot_latencies: list[float] = []
    proc = psutil.Process()
    proc.cpu_percent(interval=None)

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        vis, row, zc, inf_ms, tot_ms = _process_single_frame(
            detector=detector,
            frame_bgr=frame,
            frame_id=img_path.name,
            frame_label=img_path.stem,
            zone_analyzer=zone_analyzer,
            safety_cfg=cfg,
            heatmap=heatmap,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            blur_privacy=do_blur,
        )
        rows.append(row)
        zone_logs.append({"frame_id": img_path.name, "zones": zc})
        inf_latencies.append(inf_ms)
        tot_latencies.append(tot_ms)
        cv2.imwrite(str(out_img_dir / img_path.name), vis)

    run_name = out_csv_path.stem[:-7] if out_csv_path.stem.endswith("_counts") else out_csv_path.stem
    summary = _finalize_outputs(
        run_name=run_name,
        detector=detector,
        rows=rows,
        zone_logs=zone_logs,
        inf_latencies=inf_latencies,
        tot_latencies=tot_latencies,
        out_csv_path=out_csv_path,
        out_root=root,
        heatmap=heatmap,
        safety_cfg=cfg,
        proc=proc,
    )
    summary["annotated_dir"] = str(out_img_dir)
    return summary


def process_video(
    detector: EdgeDetector,
    video_path: Path,
    out_video_path: Path,
    out_csv_path: Path,
    zone_analyzer: ZoneAnalyzer | None,
    conf_thres: float | None,
    iou_thres: float | None,
    max_frames: int = 0,
    safety_cfg: dict[str, Any] | None = None,
    blur_privacy: bool | None = None,
    out_root: Path | None = None,
) -> dict[str, Any]:
    """Process a video stream cleanly offline frame-by-frame."""
    cfg = safety_cfg if safety_cfg is not None else load_safety_config(None)
    do_blur = bool(cfg.get("privacy_blur_persons", True)) if blur_privacy is None else blur_privacy
    root = out_root if out_root is not None else out_csv_path.parent.parent

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    heatmap = PedestrianHeatmapAccumulator()
    rows: list[dict[str, object]] = []
    zone_logs: list[dict[str, Any]] = []
    inf_latencies: list[float] = []
    tot_latencies: list[float] = []
    proc = psutil.Process()
    proc.cpu_percent(interval=None)
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vis, row, zc, inf_ms, tot_ms = _process_single_frame(
                detector=detector,
                frame_bgr=frame,
                frame_id=frame_idx,
                frame_label=f"frame {frame_idx}",
                zone_analyzer=zone_analyzer,
                safety_cfg=cfg,
                heatmap=heatmap,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                blur_privacy=do_blur,
            )
            rows.append(row)
            zone_logs.append({"frame_id": frame_idx, "zones": zc})
            inf_latencies.append(inf_ms)
            tot_latencies.append(tot_ms)
            writer.write(vis)
            frame_idx += 1
            if max_frames > 0 and frame_idx >= max_frames:
                break
    finally:
        cap.release()
        writer.release()

    run_name = out_csv_path.stem[:-7] if out_csv_path.stem.endswith("_counts") else out_csv_path.stem
    summary = _finalize_outputs(
        run_name=run_name,
        detector=detector,
        rows=rows,
        zone_logs=zone_logs,
        inf_latencies=inf_latencies,
        tot_latencies=tot_latencies,
        out_csv_path=out_csv_path,
        out_root=root,
        heatmap=heatmap,
        safety_cfg=cfg,
        proc=proc,
    )
    summary["video_fps_source"] = round(float(fps), 2)
    summary["annotated_video"] = str(out_video_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="SafeVision Edge offline ExecuTorch inference.")
    parser.add_argument("--source", type=Path, required=True, help="Image file, folder of images, or video file.")
    parser.add_argument("--model", type=Path, default=PKG_ROOT / "models" / "custom_model.pte")
    parser.add_argument("--config-dir", type=Path, default=PKG_ROOT / "config")
    parser.add_argument("--out-dir", type=Path, default=PKG_ROOT / "outputs")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--no-zones", action="store_true")
    parser.add_argument("--no-blur", action="store_true", help="Disable privacy Gaussian blur on person boxes.")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    class_names, prep_cfg, zone_analyzer = load_configs(args.config_dir)
    safety_cfg = load_safety_config(args.config_dir / "safety_config.json")
    if args.no_zones:
        zone_analyzer = None
    blur_privacy = False if args.no_blur else bool(safety_cfg.get("privacy_blur_persons", True))
    detector = EdgeDetector(args.model, class_names, prep_cfg)

    src = args.source
    if not src.exists():
        raise FileNotFoundError(f"Input source not found: {src}")

    if src.is_dir():
        imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        summary = process_images(
            detector,
            imgs,
            args.out_dir / "annotated" / src.name,
            args.out_dir / "csv" / f"{src.name}_counts.csv",
            zone_analyzer,
            args.conf,
            args.iou,
            safety_cfg=safety_cfg,
            blur_privacy=blur_privacy,
            out_root=args.out_dir,
        )
    elif src.suffix.lower() in VIDEO_EXTS:
        summary = process_video(
            detector,
            src,
            args.out_dir / "annotated" / f"{src.stem}_annotated.mp4",
            args.out_dir / "csv" / f"{src.stem}_counts.csv",
            zone_analyzer,
            args.conf,
            args.iou,
            max_frames=args.max_frames,
            safety_cfg=safety_cfg,
            blur_privacy=blur_privacy,
            out_root=args.out_dir,
        )
    elif src.suffix.lower() in IMAGE_EXTS:
        summary = process_images(
            detector,
            [src],
            args.out_dir / "annotated",
            args.out_dir / "csv" / f"{src.stem}_counts.csv",
            zone_analyzer,
            args.conf,
            args.iou,
            safety_cfg=safety_cfg,
            blur_privacy=blur_privacy,
            out_root=args.out_dir,
        )
    else:
        raise ValueError(f"Unsupported input type: {src}")

    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
