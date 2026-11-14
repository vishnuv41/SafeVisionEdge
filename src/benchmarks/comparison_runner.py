"""Controlled paired CPU vs Hailo-8 NPU comparison benchmark runner."""

from __future__ import annotations

import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmarks.hardware_monitor import HardwareMonitor
from decode import Detection
from engines.base_engine import BaseInferenceEngine, EngineMode
from safety_pulse import ALL_CLASSES


def _bbox_iou(a: Detection, b: Detection) -> float:
    xx1 = max(a.x1, b.x1)
    yy1 = max(a.y1, b.y1)
    xx2 = min(a.x2, b.x2)
    yy2 = min(a.y2, b.y2)
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def compute_detection_agreement(
    cpu_frames: list[list[Detection]],
    npu_frames: list[list[Detection]],
    iou_match_threshold: float = 0.50,
    ground_truth_frames: list[list[Detection]] | None = None,
) -> dict[str, Any]:
    """Compare CPU vs NPU detection counts, confidence distributions, and box IoU similarity."""
    def _summarize_side(frames_dets: list[list[Detection]]) -> dict[str, Any]:
        counts = {c: 0 for c in ALL_CLASSES}
        confs: list[float] = []
        total = 0
        for dets in frames_dets:
            for d in dets:
                total += 1
                counts[d.class_name] = counts.get(d.class_name, 0) + 1
                confs.append(float(d.confidence))
        return {
            "total_detections": total,
            "class_counts": counts,
            "confidence_distribution": {
                "mean": round(float(np.mean(confs)), 4) if confs else None,
                "median": round(float(np.median(confs)), 4) if confs else None,
                "min": round(float(min(confs)), 4) if confs else None,
                "max": round(float(max(confs)), 4) if confs else None,
            },
        }

    cpu_summary = _summarize_side(cpu_frames)
    npu_summary = _summarize_side(npu_frames)

    matched_ious: list[float] = []
    matched_count = 0
    cpu_total = cpu_summary["total_detections"]
    npu_total = npu_summary["total_detections"]

    for c_dets, n_dets in zip(cpu_frames, npu_frames):
        used_n: set[int] = set()
        for c_det in sorted(c_dets, key=lambda d: d.confidence, reverse=True):
            best_idx = -1
            best_iou = 0.0
            for n_idx, n_det in enumerate(n_dets):
                if n_idx in used_n or n_det.class_id != c_det.class_id:
                    continue
                iou = _bbox_iou(c_det, n_det)
                if iou >= iou_match_threshold and iou > best_iou:
                    best_iou = iou
                    best_idx = n_idx
            if best_idx >= 0:
                used_n.add(best_idx)
                matched_count += 1
                matched_ious.append(best_iou)

    cpu_only = max(0, cpu_total - matched_count)
    npu_only = max(0, npu_total - matched_count)

    agreement: dict[str, Any] = {
        "iou_threshold": iou_match_threshold,
        "cpu": cpu_summary,
        "npu": npu_summary,
        "cross_engine_similarity": {
            "matched_detections": matched_count,
            "cpu_only_detections": cpu_only,
            "npu_only_detections": npu_only,
            "mean_matched_iou": round(float(np.mean(matched_ious)), 4) if matched_ious else None,
            "agreement_precision_vs_cpu": round(matched_count / npu_total, 4) if npu_total > 0 else None,
            "agreement_recall_vs_cpu": round(matched_count / cpu_total, 4) if cpu_total > 0 else None,
            "note": "Cross-engine agreement treats CPU detections as reference baseline, not labelled ground truth.",
        },
        "ground_truth_metrics": {
            "available": False,
            "note": "No labelled ground truth provided; precision/recall/FP/FN require labelled validation ground truth.",
        },
    }

    if ground_truth_frames is not None:
        def _eval_gt(pred_frames: list[list[Detection]]) -> dict[str, Any]:
            tp = 0
            fp = 0
            fn = 0
            for preds, gts in zip(pred_frames, ground_truth_frames):
                used_gt: set[int] = set()
                for p in sorted(preds, key=lambda d: d.confidence, reverse=True):
                    best_g = -1
                    best_iou = 0.0
                    for g_idx, g in enumerate(gts):
                        if g_idx in used_gt or g.class_id != p.class_id:
                            continue
                        iou = _bbox_iou(p, g)
                        if iou >= iou_match_threshold and iou > best_iou:
                            best_iou = iou
                            best_g = g_idx
                    if best_g >= 0:
                        used_gt.add(best_g)
                        tp += 1
                    else:
                        fp += 1
                fn += max(0, len(gts) - len(used_gt))
            prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
            rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
            return {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": prec,
                "recall": rec,
            }

        agreement["ground_truth_metrics"] = {
            "available": True,
            "cpu": _eval_gt(cpu_frames),
            "npu": _eval_gt(npu_frames),
        }

    return agreement


def _summarize_engine_timings(
    engine_id: str,
    status_info: dict[str, Any],
    inf_times: list[float],
    tot_times: list[float],
    processed: int,
    failed: int,
    errors: list[str],
    power_watts: float | None = None,
) -> dict[str, Any]:
    if inf_times:
        mean_inf = float(np.mean(inf_times))
        med_inf = float(np.median(inf_times))
        p95_inf = float(np.percentile(inf_times, 95))
        inf_fps = round(1000.0 / mean_inf, 2) if mean_inf > 0 else 0.0
    else:
        mean_inf = med_inf = p95_inf = 0.0
        inf_fps = 0.0

    if tot_times:
        mean_tot = float(np.mean(tot_times))
        med_tot = float(np.median(tot_times))
        p95_tot = float(np.percentile(tot_times, 95))
        e2e_fps = round(1000.0 / mean_tot, 2) if mean_tot > 0 else 0.0
    else:
        mean_tot = med_tot = p95_tot = 0.0
        e2e_fps = 0.0

    power_eff: dict[str, Any] = {"available": False, "fps_per_watt": None, "energy_per_frame_mj": None}
    if power_watts and power_watts > 0 and e2e_fps > 0:
        power_eff = {
            "available": True,
            "avg_watts": round(power_watts, 3),
            "fps_per_watt": round(e2e_fps / power_watts, 3),
            "energy_per_frame_mj": round(power_watts * mean_tot, 2),
        }

    return {
        "engine_id": engine_id,
        "available": bool(status_info.get("available")),
        "status_code": status_info.get("status_code"),
        "model_name": status_info.get("model_name"),
        "runtime": status_info.get("runtime"),
        "backend": status_info.get("backend"),
        "processed_frames": processed,
        "failed_frames": failed,
        "errors": errors[:20],
        "mean_inference_ms": round(mean_inf, 2) if inf_times else None,
        "median_inference_ms": round(med_inf, 2) if inf_times else None,
        "p95_inference_ms": round(p95_inf, 2) if inf_times else None,
        "inference_fps": inf_fps if inf_times else None,
        "mean_total_ms": round(mean_tot, 2) if tot_times else None,
        "median_total_ms": round(med_tot, 2) if tot_times else None,
        "p95_total_ms": round(p95_tot, 2) if tot_times else None,
        "end_to_end_fps": e2e_fps if tot_times else None,
        "power_efficiency": power_eff,
    }


class ComparisonRunner:
    """Controlled paired/interleaved CPU vs Hailo-8 NPU comparison benchmark."""

    def __init__(
        self,
        cpu_engine: BaseInferenceEngine,
        npu_engine: BaseInferenceEngine,
        hef_path: Path | None = None,
    ) -> None:
        self.cpu_engine = cpu_engine
        self.npu_engine = npu_engine
        self.hef_path = hef_path

    def run(
        self,
        frames_bgr: list[np.ndarray],
        *,
        mode: str = EngineMode.COMPARE,
        warmup_runs: int = 3,
        measured_runs: int = 10,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        ground_truth_frames: list[list[Detection]] | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        if not frames_bgr:
            raise ValueError("At least one input frame is required for benchmarking")

        run_cpu = mode in (EngineMode.CPU, EngineMode.COMPARE)
        run_npu = mode in (EngineMode.NPU, EngineMode.COMPARE)

        cpu_status = self.cpu_engine.get_status()
        npu_status = self.npu_engine.get_status()

        cpu_ready = False
        npu_ready = False
        cpu_init_error: str | None = None
        npu_init_error: str | None = None

        if run_cpu:
            try:
                self.cpu_engine.load()
                cpu_ready = True
                cpu_status = self.cpu_engine.get_status()
            except Exception as exc:
                cpu_init_error = str(exc)

        if run_npu:
            try:
                if self.npu_engine.is_available():
                    self.npu_engine.load()
                    npu_ready = True
                    npu_status = self.npu_engine.get_status()
                else:
                    npu_init_error = f"{npu_status.get('status_code')}: {npu_status.get('message')}"
            except Exception as exc:
                npu_init_error = str(exc)

        # Warm-up passes excluded from measured metrics
        if warmup_runs > 0:
            for w in range(warmup_runs):
                frame = frames_bgr[w % len(frames_bgr)]
                if cpu_ready:
                    try:
                        self.cpu_engine.predict_frame_detailed(frame, conf_thres=conf_thres, iou_thres=iou_thres)
                    except Exception:
                        pass
                if npu_ready:
                    try:
                        self.npu_engine.predict_frame_detailed(frame, conf_thres=conf_thres, iou_thres=iou_thres)
                    except Exception:
                        pass

        monitor = HardwareMonitor(hef_path=self.hef_path, sample_interval_sec=0.2)
        monitor.start()

        cpu_inf: list[float] = []
        cpu_tot: list[float] = []
        cpu_dets_all: list[list[Detection]] = []
        cpu_errors: list[str] = [cpu_init_error] if cpu_init_error else []
        cpu_failed = 0

        npu_inf: list[float] = []
        npu_tot: list[float] = []
        npu_dets_all: list[list[Detection]] = []
        npu_errors: list[str] = [npu_init_error] if npu_init_error else []
        npu_failed = 0

        t_bench0 = time.perf_counter()
        try:
            for idx in range(measured_runs):
                frame = frames_bgr[idx % len(frames_bgr)]
                # Interleave order on alternating iterations to avoid thermal ordering bias
                order = ("cpu", "npu") if idx % 2 == 0 else ("npu", "cpu")
                for target in order:
                    if target == "cpu" and run_cpu:
                        if not cpu_ready:
                            cpu_failed += 1
                            cpu_dets_all.append([])
                        else:
                            try:
                                res = self.cpu_engine.predict_frame_detailed(
                                    frame, conf_thres=conf_thres, iou_thres=iou_thres
                                )
                                cpu_inf.append(res.inference_ms)
                                cpu_tot.append(res.total_ms)
                                cpu_dets_all.append(res.detections)
                            except Exception as exc:
                                cpu_failed += 1
                                cpu_errors.append(f"iter {idx}: {exc}")
                                cpu_dets_all.append([])
                    elif target == "npu" and run_npu:
                        if not npu_ready:
                            npu_failed += 1
                            npu_dets_all.append([])
                        else:
                            try:
                                res = self.npu_engine.predict_frame_detailed(
                                    frame, conf_thres=conf_thres, iou_thres=iou_thres
                                )
                                npu_inf.append(res.inference_ms)
                                npu_tot.append(res.total_ms)
                                npu_dets_all.append(res.detections)
                            except Exception as exc:
                                npu_failed += 1
                                npu_errors.append(f"iter {idx}: {exc}")
                                npu_dets_all.append([])

                if progress_cb is not None:
                    progress_cb(idx + 1, measured_runs, "running")
        finally:
            hw_summary = monitor.stop()

        total_duration_s = round(time.perf_counter() - t_bench0, 3)
        avg_watts = hw_summary.get("power", {}).get("avg_watts")

        cpu_report = _summarize_engine_timings(
            "cpu",
            cpu_status,
            cpu_inf,
            cpu_tot,
            len(cpu_inf),
            cpu_failed,
            cpu_errors,
            power_watts=avg_watts,
        )
        npu_report = _summarize_engine_timings(
            "npu",
            npu_status,
            npu_inf,
            npu_tot,
            len(npu_inf),
            npu_failed,
            npu_errors,
            power_watts=avg_watts,
        )

        # Speedup calculation ONLY when both engines actually ran measured frames
        speedup: dict[str, Any] = {"available": False, "inference_speedup_x": None, "end_to_end_speedup_x": None}
        if cpu_report["mean_inference_ms"] and npu_report["mean_inference_ms"]:
            speedup = {
                "available": True,
                "inference_speedup_x": round(cpu_report["mean_inference_ms"] / npu_report["mean_inference_ms"], 2),
                "end_to_end_speedup_x": round(cpu_report["mean_total_ms"] / npu_report["mean_total_ms"], 2)
                if cpu_report["mean_total_ms"] and npu_report["mean_total_ms"]
                else None,
            }

        gt_aligned = (
            [ground_truth_frames[i % len(ground_truth_frames)] for i in range(measured_runs)]
            if ground_truth_frames
            else None
        )
        accuracy = compute_detection_agreement(
            cpu_frames=cpu_dets_all if run_cpu else [],
            npu_frames=npu_dets_all if run_npu else [],
            ground_truth_frames=gt_aligned,
        )

        sample_shape = list(frames_bgr[0].shape[:2])
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": mode,
            "interleaved": True,
            "warmup_runs_excluded": warmup_runs,
            "measured_runs": measured_runs,
            "unique_input_frames": len(frames_bgr),
            "input_resolution_hw": sample_shape,
            "thresholds": {"confidence": conf_thres, "iou": iou_thres},
            "duration_sec": total_duration_s,
            "platform": {
                "os": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "is_raspberry_pi": "aarch64" in platform.machine().lower() or "arm" in platform.machine().lower(),
            },
            "cpu": cpu_report,
            "npu": npu_report,
            "speedup": speedup,
            "accuracy_comparison": accuracy,
            "hardware_monitoring": hw_summary,
        }
