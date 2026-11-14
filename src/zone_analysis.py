"""Road zone occupancy, exclusive boundary assignment, and optional homography calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from decode import Detection

VEHICLE_CLASS_NAMES = frozenset({"bicycle", "motorcycle", "auto_rickshaw", "car", "bus", "truck"})
ZONE_TYPE_PRIORITY = {
    "crossing": 0,
    "footpath": 1,
    "road": 2,
}


def infer_zone_type_from_def(zone_def: dict[str, Any]) -> str:
    explicit = str(zone_def.get("type") or "").strip().lower()
    if explicit in ZONE_TYPE_PRIORITY:
        return explicit
    name = str(zone_def.get("name") or "").strip().lower()
    if "cross" in name:
        return "crossing"
    if "foot" in name or "sidewalk" in name or "walk" in name:
        return "footpath"
    if "road" in name or "lane" in name or "carriage" in name:
        return "road"
    return "other"


def polygon_area_px(poly: np.ndarray) -> float:
    """Compute signed-magnitude polygon area using Shoelace formula."""
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0].astype(np.float64)
    y = poly[:, 1].astype(np.float64)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def compute_homography_calibration(
    image_points_norm: list[list[float]],
    world_points_m: list[list[float]],
    frame_width: int = 1280,
    frame_height: int = 720,
) -> dict[str, Any]:
    """Fit a ground-plane homography from >=4 normalized image points to metric ground coordinates (metres)."""
    if len(image_points_norm) < 4 or len(image_points_norm) != len(world_points_m):
        raise ValueError("Homography calibration requires >= 4 matching image_points_norm and world_points_m pairs")

    for i, (ip, wp) in enumerate(zip(image_points_norm, world_points_m)):
        if len(ip) != 2 or not (0.0 <= float(ip[0]) <= 1.0 and 0.0 <= float(ip[1]) <= 1.0):
            raise ValueError(f"Calibration image point {i} out of [0, 1] bounds: {ip}")
        if len(wp) != 2:
            raise ValueError(f"Calibration world point {i} must be [X_m, Y_m]: {wp}")

    src_norm = np.array(image_points_norm, dtype=np.float64)
    dst_m = np.array(world_points_m, dtype=np.float64)

    # Fit directly in normalized [0..1] image coordinates so matrix applies consistently across resolutions
    H, mask = cv2.findHomography(src_norm, dst_m, method=0)
    if H is None or np.linalg.matrix_rank(H) < 3:
        raise ValueError("Selected reference points are collinear or degenerate; cannot compute ground-plane homography")

    projected = cv2.perspectiveTransform(src_norm.reshape(-1, 1, 2), H).reshape(-1, 2)
    reproj_errors = np.linalg.norm(projected - dst_m, axis=1)
    rmse_m = float(np.sqrt(np.mean(reproj_errors ** 2)))

    return {
        "calibrated": True,
        "method": "homography_4pt",
        "point_count": len(image_points_norm),
        "reference_frame_size": [int(frame_width), int(frame_height)],
        "image_points_norm": [[round(float(p[0]), 5), round(float(p[1]), 5)] for p in image_points_norm],
        "world_points_m": [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in world_points_m],
        "homography_norm_to_m": [[float(v) for v in row] for row in H.tolist()],
        "reprojection_rmse_m": round(rmse_m, 4),
    }


def project_zone_area_m2(zone_def: dict[str, Any], H_norm_to_m: np.ndarray) -> float | None:
    """Project normalized zone polygon(s) onto the calibrated ground plane (in m^2)."""
    raw_polys: list[list[list[float]]] = []
    primary = zone_def.get("polygon_norm") or zone_def.get("points")
    if primary:
        raw_polys.append(primary)
    for extra in zone_def.get("extra_polygons_norm", []) or []:
        if extra:
            raw_polys.append(extra)
    if not raw_polys:
        return None
    total_area = 0.0
    for poly in raw_polys:
        if len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.float64).reshape(-1, 1, 2)
        world_pts = cv2.perspectiveTransform(pts, H_norm_to_m).reshape(-1, 2)
        if not np.all(np.isfinite(world_pts)):
            return None
        total_area += polygon_area_px(world_pts)
    return round(float(total_area), 2) if total_area > 0 else None


class ZoneAnalyzer:
    """Classify detections into configured polygonal road zones using JSON definitions."""

    def __init__(self, zones_cfg: dict[str, Any]) -> None:
        self.anchor_mode = str(zones_cfg.get("anchor_mode", "bottom_center"))
        self.zones = list(zones_cfg.get("zones", []))
        self.calibration = zones_cfg.get("calibration")

    @classmethod
    def from_json(cls, path: Path) -> "ZoneAnalyzer":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def _anchor_point(self, det: Detection) -> tuple[float, float]:
        cx = (det.x1 + det.x2) * 0.5
        if self.anchor_mode == "center":
            cy = (det.y1 + det.y2) * 0.5
        else:
            cy = det.y2
        return float(cx), float(cy)

    @staticmethod
    def _zone_polys(zone_def: dict[str, Any], w: int, h: int, dtype: Any = np.float32) -> list[np.ndarray]:
        raw_polys: list[list[list[float]]] = []
        primary = zone_def.get("polygon_norm") or zone_def.get("points")
        if primary:
            raw_polys.append(primary)
        for extra in zone_def.get("extra_polygons_norm", []) or []:
            if extra:
                raw_polys.append(extra)
        if dtype is np.int32:
            return [
                np.array([[int(round(px * w)), int(round(py * h))] for px, py in poly], dtype=np.int32)
                for poly in raw_polys
            ]
        return [
            np.array([[float(px) * float(w), float(py) * float(h)] for px, py in poly], dtype=np.float32)
            for poly in raw_polys
        ]

    def assign_zones_detailed(
        self, detections: list[Detection], img_shape: tuple[int, int]
    ) -> dict[str, Any]:
        """Assign each detection to at most one zone via deterministic safety priority and return full diagnostics.

        Priority for overlapping/shared-boundary zones: crossing > footpath > road > other.
        Point-in-polygon uses `cv2.pointPolygonTest(...) >= 0` so points exactly on boundaries
        are never dropped and are deterministically resolved to the highest-priority zone.
        """
        h, w = img_shape[:2]
        zone_counts: dict[str, dict[str, int]] = {}
        priority_entries: list[tuple[int, int, str, str, list[np.ndarray]]] = []

        for idx, z in enumerate(self.zones):
            zname = str(z["name"])
            ztype = infer_zone_type_from_def(z)
            zone_counts.setdefault(zname, {"total": 0})
            polys = self._zone_polys(z, w, h, dtype=np.float32)
            prio = ZONE_TYPE_PRIORITY.get(ztype, 3)
            priority_entries.append((prio, idx, zname, ztype, polys))

        priority_entries.sort(key=lambda item: (item[0], item[1]))

        assignments: list[str | None] = []
        assignment_records: list[dict[str, Any]] = []
        unassigned_counts: dict[str, int] = {"total": 0, "person": 0, "vehicles": 0}
        boundary_hits = 0

        for det in detections:
            px, py = self._anchor_point(det)
            # Clamp slightly outside-border boxes strictly when finite inside frame checks?
            # No: keep raw anchor so out-of-frame/outside-zone anchors stay honestly unassigned.
            pt = (float(px), float(py))
            matched_zone: str | None = None
            matched_type: str | None = None
            on_boundary = False

            for _prio, _idx, zname, ztype, polys in priority_entries:
                hit_inside = False
                hit_edge = False
                for poly in polys:
                    if len(poly) < 3:
                        continue
                    dist_sign = cv2.pointPolygonTest(poly, pt, False)
                    if dist_sign > 0:
                        hit_inside = True
                        break
                    if dist_sign == 0:
                        hit_edge = True
                if hit_inside or hit_edge:
                    matched_zone = zname
                    matched_type = ztype
                    on_boundary = (not hit_inside) and hit_edge
                    if on_boundary:
                        boundary_hits += 1
                    zone_counts[zname]["total"] += 1
                    zone_counts[zname][det.class_name] = zone_counts[zname].get(det.class_name, 0) + 1
                    break

            if matched_zone is None:
                unassigned_counts["total"] += 1
                if det.class_name in VEHICLE_CLASS_NAMES:
                    unassigned_counts["vehicles"] += 1
                unassigned_counts[det.class_name] = unassigned_counts.get(det.class_name, 0) + 1

            assignments.append(matched_zone)
            assignment_records.append(
                {
                    "class_name": det.class_name,
                    "confidence": round(float(det.confidence), 4),
                    "anchor_px": [round(float(px), 2), round(float(py), 2)],
                    "anchor_norm": [
                        round(float(px) / float(w), 5) if w > 0 else 0.0,
                        round(float(py) / float(h), 5) if h > 0 else 0.0,
                    ],
                    "assigned": matched_zone is not None,
                    "zone_name": matched_zone,
                    "zone_type": matched_type,
                    "on_boundary": on_boundary,
                }
            )

        return {
            "zone_counts": zone_counts,
            "assignments": assignments,
            "assignment_records": assignment_records,
            "unassigned_counts": unassigned_counts,
            "boundary_assignments": boundary_hits,
            "policy": "exclusive_priority:crossing>footpath>road; boundary=included",
        }

    def assign_zones(
        self, detections: list[Detection], img_shape: tuple[int, int]
    ) -> tuple[dict[str, dict[str, int]], list[str | None]]:
        """Assign each detection exclusively to a zone, returning `(zone_counts, assignments)`."""
        detailed = self.assign_zones_detailed(detections, img_shape)
        return detailed["zone_counts"], detailed["assignments"]

    def analyze(
        self, detections: list[Detection], img_shape: tuple[int, int]
    ) -> dict[str, dict[str, int]]:
        """Return per-zone class and total counts."""
        zone_counts, _ = self.assign_zones(detections, img_shape)
        return zone_counts

    def compute_calibrated_densities(
        self, zone_counts: dict[str, dict[str, int]]
    ) -> dict[str, Any]:
        """Compute physical m^2 densities strictly when homography calibration is present."""
        cal = self.calibration
        if not isinstance(cal, dict) or not cal.get("calibrated") or not cal.get("homography_norm_to_m"):
            return {
                "calibrated": False,
                "physical_density_available": False,
                "note": "Scene has no ground-plane homography calibration; displaying raw occupancy counts only.",
                "per_zone": {},
            }

        H = np.array(cal["homography_norm_to_m"], dtype=np.float64)
        per_zone: dict[str, Any] = {}
        for z in self.zones:
            zname = str(z["name"])
            area_m2 = project_zone_area_m2(z, H)
            counts = zone_counts.get(zname, {})
            persons = int(counts.get("person", 0))
            vehicles = int(sum(int(counts.get(vc, 0)) for vc in VEHICLE_CLASS_NAMES))
            if area_m2 and area_m2 > 0:
                per_zone[zname] = {
                    "area_m2": area_m2,
                    "pedestrians_per_m2": round(persons / area_m2, 4),
                    "vehicles_per_m2": round(vehicles / area_m2, 4),
                }
        return {
            "calibrated": True,
            "physical_density_available": bool(per_zone),
            "reprojection_rmse_m": cal.get("reprojection_rmse_m"),
            "per_zone": per_zone,
        }

    def draw_zones(
        self, img_bgr: np.ndarray, zone_counts: dict[str, dict[str, int]] | None = None
    ) -> np.ndarray:
        """Render configured zone polygons and per-zone occupancy labels."""
        vis = img_bgr.copy()
        h, w = vis.shape[:2]
        for z in self.zones:
            zname = str(z["name"])
            color = tuple(int(c) for c in z.get("color_bgr", [0, 220, 255]))
            polys = self._zone_polys(z, w, h, dtype=np.int32)
            for idx, pts in enumerate(polys):
                cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
                if idx == 0:
                    label = zname
                    if zone_counts and zname in zone_counts:
                        label = f"{label}: {zone_counts[zname].get('total', 0)}"
                    top_pt = tuple(pts[0])
                    cv2.putText(
                        vis,
                        label,
                        (top_pt[0] + 6, max(45, top_pt[1] + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
        return vis
