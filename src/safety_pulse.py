"""Stateless per-frame SafeVision Safety Pulse intelligence, 3x3 pedestrian heatmap, and Safety Brief."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from decode import Detection
from zone_analysis import ZoneAnalyzer

DEFAULT_SAFETY_CONFIG: dict[str, Any] = {
    "vehicle_classes": ["bicycle", "motorcycle", "auto_rickshaw", "car", "bus", "truck"],
    "vru_classes": ["person", "bicycle", "motorcycle"],
    "density_thresholds": {"low_max": 4, "moderate_max": 9},
    "weights": {
        "density_low": 0,
        "density_moderate": 1,
        "density_high": 2,
        "co_presence": 2,
        "high_vru_share": 1,
    },
    "vru_share_threshold": 0.5,
    "safety_pulse_bands": {"low_max": 1, "medium_max": 3},
    "privacy_blur_persons": True,
    "disclaimer": (
        "Safety Pulse is an explainable, configurable rule-based operational indicator (0-5) "
        "and is NOT a validated accident-prediction or statistical crash-risk model."
    ),
}

ALL_CLASSES = ("person", "bicycle", "motorcycle", "auto_rickshaw", "car", "bus", "truck")


def load_safety_config(path: Path | None) -> dict[str, Any]:
    """Load JSON safety configuration or return defaults if absent."""
    cfg = json.loads(json.dumps(DEFAULT_SAFETY_CONFIG))
    if path is not None and path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        for k, v in loaded.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def classify_density(vehicle_count: int, cfg: dict[str, Any]) -> tuple[str, int]:
    """Classify per-frame vehicle count into Low (0), Moderate (1), or High (2)."""
    thr = cfg.get("density_thresholds", DEFAULT_SAFETY_CONFIG["density_thresholds"])
    w = cfg.get("weights", DEFAULT_SAFETY_CONFIG["weights"])
    low_max = int(thr.get("low_max", 4))
    mod_max = int(thr.get("moderate_max", 9))
    if vehicle_count <= low_max:
        return "Low", int(w.get("density_low", 0))
    if vehicle_count <= mod_max:
        return "Moderate", int(w.get("density_moderate", 1))
    return "High", int(w.get("density_high", 2))


def map_safety_pulse_label(score: int, cfg: dict[str, Any]) -> str:
    """Map integer Safety Pulse score (0-5) to Low, Medium, or High."""
    bands = cfg.get("safety_pulse_bands", DEFAULT_SAFETY_CONFIG["safety_pulse_bands"])
    if score <= int(bands.get("low_max", 1)):
        return "Low"
    if score <= int(bands.get("medium_max", 3)):
        return "Medium"
    return "High"


def analyze_frame(
    detections: list[Detection],
    img_shape: tuple[int, int],
    zone_analyzer: ZoneAnalyzer | None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute stateless per-frame class counts, zone occupancy, and Safety Pulse."""
    safety_cfg = cfg if cfg is not None else DEFAULT_SAFETY_CONFIG
    vehicle_set = set(safety_cfg.get("vehicle_classes", DEFAULT_SAFETY_CONFIG["vehicle_classes"]))
    vru_set = set(safety_cfg.get("vru_classes", DEFAULT_SAFETY_CONFIG["vru_classes"]))

    counts: dict[str, int] = {name: 0 for name in ALL_CLASSES}
    for det in detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1

    total_detections = len(detections)
    total_vehicles = sum(counts.get(vcls, 0) for vcls in vehicle_set)
    vru_count = sum(counts.get(rcls, 0) for rcls in vru_set)
    vru_share = float(vru_count / total_detections) if total_detections > 0 else 0.0

    zone_counts: dict[str, dict[str, int]] = {}
    assignments: list[str | None] = [None] * len(detections)
    unassigned_counts: dict[str, int] = {name: 0 for name in ALL_CLASSES}
    unassigned_counts["total"] = len(detections)
    unassigned_counts["vehicles"] = total_vehicles
    boundary_assignments = 0
    zone_types_by_name: dict[str, str] = {}
    calibrated_densities: dict[str, Any] = {"calibrated": False, "physical_density_available": False, "per_zone": {}}

    if zone_analyzer is not None:
        if hasattr(zone_analyzer, "assign_zones_detailed"):
            detailed = zone_analyzer.assign_zones_detailed(detections, img_shape)
            zone_counts = detailed["zone_counts"]
            assignments = detailed["assignments"]
            boundary_assignments = int(detailed.get("boundary_assignments", 0))
            raw_unassigned = detailed.get("unassigned_counts", {})
            unassigned_counts = {name: int(raw_unassigned.get(name, 0)) for name in ALL_CLASSES}
            unassigned_counts["total"] = int(raw_unassigned.get("total", 0))
            unassigned_counts["vehicles"] = int(raw_unassigned.get("vehicles", 0))
            calibrated_densities = zone_analyzer.compute_calibrated_densities(zone_counts)
        else:
            zone_counts, assignments = zone_analyzer.assign_zones(detections, img_shape)
            unassigned_counts["total"] = sum(1 for a in assignments if a is None)
        for z in getattr(zone_analyzer, "zones", []) or []:
            zname = str(z.get("name", ""))
            ztype = str(z.get("type") or zname).strip().lower()
            if ztype not in {"road", "footpath", "crossing"}:
                low = zname.strip().lower()
                if "cross" in low:
                    ztype = "crossing"
                elif "foot" in low or "sidewalk" in low or "walk" in low:
                    ztype = "footpath"
                elif "road" in low or "lane" in low:
                    ztype = "road"
            zone_types_by_name[zname] = ztype

    def _matching_zone_names(target_type: str) -> list[str]:
        matched = [zn for zn, zt in zone_types_by_name.items() if zt == target_type]
        if matched:
            return matched
        return [target_type] if target_type in zone_counts else []

    def _zone_persons(ztype: str) -> int:
        return int(sum(int(zone_counts.get(zn, {}).get("person", 0)) for zn in _matching_zone_names(ztype)))

    def _zone_vehicles(ztype: str) -> int:
        total = 0
        for zn in _matching_zone_names(ztype):
            zmap = zone_counts.get(zn, {})
            total += sum(int(zmap.get(vcls, 0)) for vcls in vehicle_set)
        return int(total)

    def _category_breakdown(ztype: str) -> dict[str, int]:
        bd = {name: 0 for name in ALL_CLASSES}
        for zn in _matching_zone_names(ztype):
            zmap = zone_counts.get(zn, {})
            for cname in ALL_CLASSES:
                bd[cname] += int(zmap.get(cname, 0))
        bd["vehicles"] = int(sum(bd[vc] for vc in vehicle_set if vc in bd))
        bd["total"] = int(sum(bd[cname] for cname in ALL_CLASSES))
        return bd

    category_class_counts = {
        "road": _category_breakdown("road"),
        "crossing": _category_breakdown("crossing"),
        "footpath": _category_breakdown("footpath"),
    }

    persons_footpath = _zone_persons("footpath")
    persons_road = _zone_persons("road")
    persons_crossing = _zone_persons("crossing")
    vehicles_footpath = _zone_vehicles("footpath")
    vehicles_road = _zone_vehicles("road")
    vehicles_crossing = _zone_vehicles("crossing")
    unassigned_count = sum(1 for a in assignments if a is None)

    density_class, density_score = classify_density(total_vehicles, safety_cfg)

    has_copresence = persons_crossing >= 1 and vehicles_crossing >= 1
    co_presence = "YES" if has_copresence else "NO"
    co_weight = int(safety_cfg.get("weights", {}).get("co_presence", 2))
    co_presence_score = co_weight if has_copresence else 0

    vru_thresh = float(safety_cfg.get("vru_share_threshold", 0.5))
    vru_weight = int(safety_cfg.get("weights", {}).get("high_vru_share", 1))
    vru_score = vru_weight if (total_detections > 0 and vru_share >= vru_thresh) else 0

    safety_pulse = int(density_score + co_presence_score + vru_score)
    safety_pulse_label = map_safety_pulse_label(safety_pulse, safety_cfg)

    return {
        "counts": counts,
        "total_detections": total_detections,
        "total_vehicles": total_vehicles,
        "zone_counts": zone_counts,
        "category_class_counts": category_class_counts,
        "unassigned_counts": unassigned_counts,
        "boundary_assignments": boundary_assignments,
        "calibrated_densities": calibrated_densities,
        "persons_footpath": persons_footpath,
        "persons_road": persons_road,
        "persons_crossing": persons_crossing,
        "vehicles_footpath": vehicles_footpath,
        "vehicles_road": vehicles_road,
        "vehicles_crossing": vehicles_crossing,
        "unassigned_count": unassigned_count,
        "density_class": density_class,
        "density_score": density_score,
        "co_presence": co_presence,
        "co_presence_score": co_presence_score,
        "vru_count": vru_count,
        "vru_share": round(vru_share, 4),
        "vru_score": vru_score,
        "safety_pulse": safety_pulse,
        "safety_pulse_label": safety_pulse_label,
        "disclaimer": str(safety_cfg.get("disclaimer", DEFAULT_SAFETY_CONFIG["disclaimer"])),
    }


def compute_temporal_occupancy(api_frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-video/batch occupancy averages, peaks, and presence rates without claiming unique tracking IDs."""
    n = len(api_frames)
    if n == 0:
        return {
            "frames_analyzed": 0,
            "pedestrian_presence_frames": 0,
            "pedestrian_presence_pct": 0.0,
            "co_presence_frames": 0,
            "co_presence_pct": 0.0,
            "avg_occupancy": {"road": 0.0, "crossing": 0.0, "footpath": 0.0, "unassigned": 0.0, "total": 0.0},
            "peak_occupancy": {"road": 0, "crossing": 0, "footpath": 0, "unassigned": 0, "total": 0},
            "note": "No frames analyzed.",
        }

    cats = ("road", "crossing", "footpath", "unassigned")
    cat_series: dict[str, list[int]] = {c: [] for c in cats}
    tot_series: list[int] = []
    ped_presence = 0
    co_presence = 0

    for f in api_frames:
        zones = f.get("zones") or {}
        for c in cats:
            cat_series[c].append(int((zones.get(c) or {}).get("total", 0)))
        tot_series.append(int(f.get("total", 0)))
        if int((f.get("counts") or {}).get("person", 0)) > 0:
            ped_presence += 1
        if bool(f.get("co_presence")):
            co_presence += 1

    return {
        "frames_analyzed": n,
        "pedestrian_presence_frames": ped_presence,
        "pedestrian_presence_pct": round(ped_presence / n * 100.0, 1),
        "co_presence_frames": co_presence,
        "co_presence_pct": round(co_presence / n * 100.0, 1),
        "avg_occupancy": {
            **{c: round(float(np.mean(cat_series[c])), 2) for c in cats},
            "total": round(float(np.mean(tot_series)), 2),
        },
        "peak_occupancy": {
            **{c: int(max(cat_series[c])) for c in cats},
            "total": int(max(tot_series)),
        },
        "note": "Aggregated per-frame zone occupancy (averages and peaks across sampled frames; does not represent unique tracked object IDs).",
    }


class PedestrianHeatmapAccumulator:
    """Accumulate pedestrian bottom-center points into a 3x3 spatial grid across frames."""

    def __init__(self, rows: int = 3, cols: int = 3) -> None:
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=np.int64)
        self.total_points = 0

    def update(self, detections: list[Detection], img_shape: tuple[int, int]) -> None:
        """Add person detections from one frame using (center_x, y_bottom)."""
        h, w = img_shape[:2]
        if h <= 0 or w <= 0:
            return
        for det in detections:
            if det.class_name != "person" and det.class_id != 0:
                continue
            cx = (det.x1 + det.x2) * 0.5
            cy = det.y2
            r = min(self.rows - 1, max(0, int((cy / float(h)) * self.rows)))
            c = min(self.cols - 1, max(0, int((cx / float(w)) * self.cols)))
            self.grid[r, c] += 1
            self.total_points += 1

    def render_image(self, cell_px: int = 180) -> np.ndarray:
        """Render the accumulated 3x3 pedestrian density grid to a BGR PNG canvas."""
        header_h = 56
        grid_w = self.cols * cell_px
        grid_h = self.rows * cell_px
        canvas = np.full((grid_h + header_h, grid_w, 3), 24, dtype=np.uint8)

        max_val = int(self.grid.max())
        norm = (self.grid.astype(np.float32) / max_val * 255.0).astype(np.uint8) if max_val > 0 else np.zeros((self.rows, self.cols), dtype=np.uint8)
        small_color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        heat_body = cv2.resize(small_color, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)
        canvas[header_h:, :] = heat_body

        cv2.putText(
            canvas,
            f"SafeVision 3x3 Pedestrian Heatmap (points={self.total_points})",
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        for r in range(self.rows):
            for c in range(self.cols):
                x0 = c * cell_px
                y0 = header_h + r * cell_px
                cv2.rectangle(canvas, (x0, y0), (x0 + cell_px - 1, y0 + cell_px - 1), (230, 230, 230), 2)
                cnt = int(self.grid[r, c])
                pct = (cnt / self.total_points * 100.0) if self.total_points > 0 else 0.0
                cv2.putText(canvas, f"R{r}C{c}: {cnt}", (x0 + 18, y0 + cell_px // 2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"({pct:.1f}%)", (x0 + 18, y0 + cell_px // 2 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        return canvas

    def save_heatmap(self, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), self.render_image())
        return out_path


def generate_safety_brief(
    rows: list[dict[str, object]],
    zone_logs: list[dict[str, Any]],
    model_path: Path,
    brief_md_path: Path,
    brief_json_path: Path,
    rss_mb: float,
    cpu_percent: float,
    heatmap_path: Path | None = None,
    disclaimer: str = DEFAULT_SAFETY_CONFIG["disclaimer"],
) -> dict[str, Any]:
    """Generate factual Markdown + JSON Safety Brief directly from processed frame records."""
    total_frames = len(rows)
    if total_frames > 0:
        peak_row = max(rows, key=lambda r: (int(r["safety_pulse"]), int(r["persons_crossing"]) + int(r["vehicles_crossing"])))
        copres_frames = sum(1 for r in rows if str(r["co_presence"]) == "YES")
        copres_pct = round(copres_frames / total_frames * 100.0, 2)
        avg_inf_ms = round(float(np.mean([float(r["inference_ms"]) for r in rows])), 2)
        avg_total_ms = round(float(np.mean([float(r["total_ms"]) for r in rows])), 2)
        peak_info = {
            "frame_id": peak_row["frame_id"],
            "safety_pulse": int(peak_row["safety_pulse"]),
            "safety_pulse_label": map_safety_pulse_label(int(peak_row["safety_pulse"]), DEFAULT_SAFETY_CONFIG),
            "density_class": str(peak_row["density_class"]),
            "co_presence": str(peak_row["co_presence"]),
            "vru_share": float(peak_row["vru_share"]),
        }
    else:
        copres_frames = 0
        copres_pct = 0.0
        avg_inf_ms = 0.0
        avg_total_ms = 0.0
        peak_info = None

    zone_totals: dict[str, int] = {}
    for entry in zone_logs:
        for zname, zstats in entry.get("zones", {}).items():
            zone_totals[zname] = zone_totals.get(zname, 0) + int(zstats.get("total", 0))
    if zone_totals:
        busiest_name = max(zone_totals, key=lambda k: zone_totals[k])
        busiest_zone = {"name": busiest_name, "cumulative_detections": zone_totals[busiest_name], "all_zones": zone_totals}
    else:
        busiest_zone = {"name": "N/A", "cumulative_detections": 0, "all_zones": {}}

    inf_fps = round(1000.0 / avg_inf_ms, 2) if avg_inf_ms > 0 else 0.0
    e2e_fps = round(1000.0 / avg_total_ms, 2) if avg_total_ms > 0 else 0.0
    model_size_mb = round(model_path.stat().st_size / (1024 ** 2), 2) if model_path.exists() else 0.0

    summary = {
        "total_frames_processed": total_frames,
        "peak_safety_pulse_frame": peak_info,
        "co_presence_alert_frames": copres_frames,
        "co_presence_alert_percent": copres_pct,
        "busiest_zone": busiest_zone,
        "avg_inference_ms": avg_inf_ms,
        "avg_total_ms": avg_total_ms,
        "approx_inference_fps": inf_fps,
        "approx_end_to_end_fps": e2e_fps,
        "model": {"name": model_path.name, "path": str(model_path), "size_mb": model_size_mb, "classes": list(ALL_CLASSES)},
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "process_rss_mb": round(float(rss_mb), 1),
            "process_cpu_percent": round(float(cpu_percent), 1),
            "is_raspberry_pi": "aarch64" in platform.machine().lower() or "arm" in platform.machine().lower(),
        },
        "heatmap_path": str(heatmap_path) if heatmap_path else None,
        "indicator_disclaimer": disclaimer,
    }

    brief_json_path.parent.mkdir(parents=True, exist_ok=True)
    brief_md_path.parent.mkdir(parents=True, exist_ok=True)
    brief_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    peak_line = (
        f"`{peak_info['frame_id']}` (score `{peak_info['safety_pulse']}` / `{peak_info['safety_pulse_label']}`, "
        f"density=`{peak_info['density_class']}`, co_presence=`{peak_info['co_presence']}`, vru_share=`{peak_info['vru_share']}`)"
        if peak_info
        else "None (0 frames)"
    )
    hw_note = (
        "Executed on ARM/Raspberry Pi target."
        if summary["hardware"]["is_raspberry_pi"]
        else "Executed on workstation x86-64 CPU (NOT measured on physical Raspberry Pi 5 hardware)."
    )

    md_lines = [
        "# SafeVision Edge — Offline Safety Brief",
        "",
        f"> **Indicator Notice:** {disclaimer}",
        "",
        "## Operational Summary",
        f"- **Total processed frames:** `{total_frames}`",
        f"- **Peak Safety Pulse frame:** {peak_line}",
        f"- **Co-presence alert frames:** `{copres_frames} / {total_frames}` (`{copres_pct:.2f}%`)",
        f"- **Busiest zone:** `{busiest_zone['name']}` (`{busiest_zone['cumulative_detections']}` cumulative detections; zone totals: `{busiest_zone['all_zones']}`)",
        "",
        "## Performance & Resource Telemetry",
        f"- **Average ExecuTorch inference latency:** `{avg_inf_ms:.2f} ms` (`{inf_fps:.2f} FPS` model-only)",
        f"- **Average end-to-end latency:** `{avg_total_ms:.2f} ms` (`{e2e_fps:.2f} FPS` full pipeline)",
        f"- **Model:** `{model_path.name}` (`{model_size_mb:.2f} MiB`, `7` classes: `{', '.join(ALL_CLASSES)}`)",
        f"- **Host platform:** `{summary['hardware']['platform']}` | `{summary['hardware']['machine']}` | `{summary['hardware']['processor']}`",
        f"- **Process telemetry:** RSS `{summary['hardware']['process_rss_mb']} MiB`, CPU `{summary['hardware']['process_cpu_percent']}%`",
        f"- **Hardware scope:** {hw_note}",
    ]
    if heatmap_path is not None:
        md_lines.append(f"- **Pedestrian 3x3 heatmap:** `{heatmap_path}`")

    brief_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return summary
