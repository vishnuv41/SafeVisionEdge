"""Lightweight, confidence-aware CV + geometry scene zone initializer.

No extra deep-learning segmentation model is added (keeping Raspberry Pi 5 RAM/CPU
overhead minimal). Uses edge orientation convergence, asphalt HSV uniformity cues,
periodic bright crosswalk stripe texture, and optional detection bottom-center
anchors strictly as auxiliary evidence — never claiming semantic road understanding
from bounding boxes alone, and never auto-saving uncertain suggestions.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from decode import Detection
from zone_store import DEFAULT_COLORS_BGR


def _conf_level(score: float) -> str:
    if score >= 0.72:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _clamp01(v: float) -> float:
    return round(max(0.0, min(1.0, float(v))), 4)


def _estimate_horizon_and_corridor(gray: np.ndarray, hsv: np.ndarray) -> dict[str, Any]:
    """Extract oblique road corridor edges and low-saturation asphalt mask in the lower scene."""
    h, w = gray.shape[:2]
    # Work at compact 320px width for fast, deterministic execution on Pi 5
    scale = 320.0 / max(1, w)
    sw = max(64, int(round(w * scale)))
    sh = max(48, int(round(h * scale)))
    g_small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    hsv_small = cv2.resize(hsv, (sw, sh), interpolation=cv2.INTER_AREA)

    y_start = int(sh * 0.28)
    roi_g = cv2.GaussianBlur(g_small[y_start:, :], (5, 5), 0)
    edges = cv2.Canny(roi_g, 50, 140)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(18, sw // 12),
        minLineLength=max(16, sw // 9),
        maxLineGap=12,
    )

    left_x_bot: list[float] = []
    left_x_top: list[float] = []
    right_x_bot: list[float] = []
    right_x_top: list[float] = []

    roi_h = sh - y_start
    if lines is not None and len(lines) > 0:
        for seg in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = [float(v) for v in seg]
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 4.0 or abs(dy) < 4.0:
                continue
            slope = dy / dx
            # Oblique road edges typically fall in |slope| between 0.35 and 3.0
            if not (0.35 <= abs(slope) <= 3.2):
                continue
            # Project line to top of ROI (y=0) and bottom of ROI (y=roi_h)
            x_at_top = x1 + (0.0 - y1) / slope
            x_at_bot = x1 + (float(roi_h) - y1) / slope
            if not (-0.3 * sw <= x_at_top <= 1.3 * sw and -0.4 * sw <= x_at_bot <= 1.4 * sw):
                continue
            if slope < 0 and x_at_bot < 0.52 * sw:
                left_x_top.append(x_at_top / sw)
                left_x_bot.append(x_at_bot / sw)
            elif slope > 0 and x_at_bot > 0.48 * sw:
                right_x_top.append(x_at_top / sw)
                right_x_bot.append(x_at_bot / sw)

    # Low-saturation neutral surface (typical asphalt/concrete carriageway in lower 55%)
    lower_hsv = hsv_small[int(sh * 0.45) :, :]
    asphalt_mask = (lower_hsv[:, :, 1] < 72) & (lower_hsv[:, :, 2] >= 28) & (lower_hsv[:, :, 2] <= 210)
    col_density = asphalt_mask.mean(axis=0)
    asphalt_cols = np.where(col_density > 0.35)[0]
    asphalt_left = float(asphalt_cols[0] / sw) if len(asphalt_cols) >= 8 else 0.14
    asphalt_right = float(asphalt_cols[-1] / sw) if len(asphalt_cols) >= 8 else 0.86
    asphalt_coverage = float(asphalt_mask.mean())

    has_left_edge = len(left_x_bot) >= 2
    has_right_edge = len(right_x_bot) >= 2

    road_top_l = float(np.median(left_x_top)) if has_left_edge else max(0.18, min(0.36, asphalt_left + 0.12))
    road_top_r = float(np.median(right_x_top)) if has_right_edge else min(0.82, max(0.64, asphalt_right - 0.12))
    road_bot_l = float(np.median(left_x_bot)) if has_left_edge else max(0.06, min(0.24, asphalt_left))
    road_bot_r = float(np.median(right_x_bot)) if has_right_edge else min(0.94, max(0.76, asphalt_right))

    # Enforce sane non-inverted corridor widths
    road_top_l = max(0.10, min(0.45, road_top_l))
    road_top_r = min(0.90, max(0.55, road_top_r))
    if road_top_r - road_top_l < 0.22:
        road_top_l, road_top_r = 0.28, 0.72
    road_bot_l = max(0.03, min(0.40, road_bot_l))
    road_bot_r = min(0.97, max(0.60, road_bot_r))
    if road_bot_r - road_bot_l < 0.35:
        road_bot_l, road_bot_r = 0.14, 0.86

    # Confidence scoring grounded strictly in real cue counts
    edge_score = 0.25 + (0.22 if has_left_edge else 0.0) + (0.22 if has_right_edge else 0.0)
    surf_bonus = min(0.20, asphalt_coverage * 0.35)
    road_conf = round(min(0.86, edge_score + surf_bonus), 2)

    return {
        "y_top": 0.35,
        "y_cross": 0.70,
        "y_bot": 0.94,
        "road_top_l": _clamp01(road_top_l),
        "road_top_r": _clamp01(road_top_r),
        "road_bot_l": _clamp01(road_bot_l),
        "road_bot_r": _clamp01(road_bot_r),
        "has_left_edge": has_left_edge,
        "has_right_edge": has_right_edge,
        "asphalt_coverage": round(asphalt_coverage, 3),
        "road_conf": road_conf,
    }


def _detect_crossing_band(gray: np.ndarray, x_left: float, x_right: float) -> dict[str, Any]:
    """Look for horizontal bright-bar periodic stripe energy indicative of a zebra crossing."""
    h, w = gray.shape[:2]
    scale = 320.0 / max(1, w)
    sw = max(64, int(round(w * scale)))
    sh = max(48, int(round(h * scale)))
    g_small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)

    cx0 = max(0, int(x_left * sw))
    cx1 = min(sw, int(x_right * sw))
    if cx1 - cx0 < 24:
        return {"detected": False, "y1": 0.70, "y2": 0.92, "confidence": 0.28, "evidence": "Corridor too narrow to sample markings"}

    # Scan candidate horizontal bands in bottom half [0.55..0.95]
    best_score = 0.0
    best_band = (0.70, 0.92)
    band_h = max(8, int(0.14 * sh))
    step = max(3, band_h // 3)
    for y0 in range(int(0.55 * sh), max(int(0.56 * sh), sh - band_h), step):
        patch = g_small[y0 : y0 + band_h, cx0:cx1]
        if patch.size == 0:
            continue
        col_mean = patch.mean(axis=0)
        # Detrend along X (horizontal blur kernel width=..., height=1 on 1xW row vector)
        k_w = max(9, (cx1 - cx0) // 6)
        smooth = cv2.blur(col_mean.reshape(1, -1), (k_w, 1)).flatten()
        highpass = col_mean - smooth
        # Count zero-crossings above contrast threshold (paint stripes create regular positive/negative swings)
        std = float(np.std(highpass))
        if std < 8.0:
            continue
        thresh = max(6.0, 0.45 * std)
        binary = (highpass > thresh).astype(np.int8)
        transitions = int(np.sum(np.abs(np.diff(binary))))
        # 6..22 transitions across road width ≈ 3..11 zebra stripes
        if 6 <= transitions <= 24:
            band_score = min(1.0, (std / 28.0) * min(1.0, transitions / 12.0))
            if band_score > best_score:
                best_score = band_score
                best_band = (y0 / float(sh), min(0.95, (y0 + band_h + int(0.04 * sh)) / float(sh)))

    detected = best_score >= 0.42
    conf = round(0.32 + 0.50 * best_score, 2) if detected else round(0.24 + 0.18 * best_score, 2)
    evidence = (
        f"Periodic horizontal contrast stripes found in y=[{best_band[0]:.2f}, {best_band[1]:.2f}]"
        if detected
        else "No reliable zebra-crossing markings detected — template placed; draw or adjust manually."
    )
    return {
        "detected": detected,
        "y1": _clamp01(best_band[0]),
        "y2": _clamp01(best_band[1]),
        "confidence": conf,
        "evidence": evidence,
    }


def estimate_scene_zones(
    img_bgr: np.ndarray | None = None,
    detections: list[Detection] | None = None,
) -> dict[str, Any]:
    """Generate confidence-aware candidate polygons for Main Road, Crossing, and Left/Right Footpaths."""
    if img_bgr is None or img_bgr.size == 0:
        warnings = [
            "No reference preview image was loaded for visual inspection; starter perspective geometry returned with LOW confidence.",
            "Upload an image/video frame and manually align or redraw all four zones before saving.",
        ]
        zones = _build_candidate_list(
            y_top=0.36,
            y_cross_top=0.70,
            y_bot=0.94,
            road_top_l=0.26,
            road_top_r=0.74,
            road_mid_l=0.18,
            road_mid_r=0.82,
            road_bot_l=0.12,
            road_bot_r=0.88,
            road_conf=0.25,
            cross_conf=0.20,
            left_fp_conf=0.22,
            right_fp_conf=0.22,
            road_evidence="Default geometric template (no image frame provided).",
            cross_evidence="No image analyzed — crossing position unverified.",
            left_fp_evidence="Default side strip template (no curb cues inspected).",
            right_fp_evidence="Default side strip template (no curb cues inspected).",
        )
        return {
            "method": "geometric_template_unverified",
            "requires_confirmation": True,
            "overall_confidence": 0.22,
            "warnings": warnings,
            "zones": zones,
        }

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    corridor = _estimate_horizon_and_corridor(gray, hsv)
    cross_info = _detect_crossing_band(gray, corridor["road_bot_l"], corridor["road_bot_r"])

    y_top = corridor["y_top"]
    y_cross = cross_info["y1"] if cross_info["detected"] else corridor["y_cross"]
    y_bot = cross_info["y2"] if cross_info["detected"] else corridor["y_bot"]

    # Linearly interpolate road corridor edges at crossing top y_cross
    frac_cross = max(0.1, min(0.95, (y_cross - y_top) / max(0.1, y_bot - y_top)))
    road_top_l = corridor["road_top_l"]
    road_top_r = corridor["road_top_r"]
    road_bot_l = corridor["road_bot_l"]
    road_bot_r = corridor["road_bot_r"]
    road_mid_l = _clamp01(road_top_l + frac_cross * (road_bot_l - road_top_l))
    road_mid_r = _clamp01(road_top_r + frac_cross * (road_bot_r - road_top_r))

    road_conf = float(corridor["road_conf"])
    det_note = ""
    if detections:
        # Auxiliary sanity check only: count vehicles inside candidate road
        veh_hits = 0
        for d in detections:
            if d.class_name != "person":
                ax = ((d.x1 + d.x2) * 0.5) / max(1, w)
                ay = d.y2 / max(1, h)
                if road_bot_l <= ax <= road_bot_r and y_top <= ay <= y_bot:
                    veh_hits += 1
        if veh_hits > 0:
            det_note = f" ({veh_hits} vehicle anchors corroborate corridor)"

    edge_cues = []
    if corridor["has_left_edge"]:
        edge_cues.append("left curb/edge lines")
    if corridor["has_right_edge"]:
        edge_cues.append("right curb/edge lines")
    if corridor["asphalt_coverage"] > 0.2:
        edge_cues.append(f"neutral road texture ({int(corridor['asphalt_coverage']*100)}%)")
    road_evidence = (
        ("Detected " + ", ".join(edge_cues) + det_note)
        if edge_cues
        else "Weak edge cues — using central perspective road corridor; please verify curb boundaries."
    )

    left_fp_conf = round(0.32 + (0.28 if corridor["has_left_edge"] and road_mid_l >= 0.06 else 0.0), 2)
    right_fp_conf = round(0.32 + (0.28 if corridor["has_right_edge"] and road_mid_r <= 0.94 else 0.0), 2)

    zones = _build_candidate_list(
        y_top=y_top,
        y_cross_top=y_cross,
        y_bot=y_bot,
        road_top_l=road_top_l,
        road_top_r=road_top_r,
        road_mid_l=road_mid_l,
        road_mid_r=road_mid_r,
        road_bot_l=road_bot_l,
        road_bot_r=road_bot_r,
        road_conf=road_conf,
        cross_conf=float(cross_info["confidence"]),
        left_fp_conf=left_fp_conf,
        right_fp_conf=right_fp_conf,
        road_evidence=road_evidence,
        cross_evidence=cross_info["evidence"],
        left_fp_evidence=(
            "Left margin inferred outside detected left road edge — verify sidewalk boundary."
            if corridor["has_left_edge"]
            else "No clear left curb detected — please draw or adjust Left Footpath."
        ),
        right_fp_evidence=(
            "Right margin inferred outside detected right road edge — verify sidewalk boundary."
            if corridor["has_right_edge"]
            else "No clear right curb detected — please draw or adjust Right Footpath."
        ),
    )

    warnings: list[str] = []
    if not cross_info["detected"]:
        warnings.append(
            "Pedestrian Crossing could not be reliably identified from image markings. Draw or adjust the crossing polygon manually."
        )
    if not corridor["has_left_edge"] or not corridor["has_right_edge"]:
        warnings.append(
            "Road curb edges were only partially visible; please verify Left/Right Footpath and Main Road boundaries before saving."
        )

    overall_conf = round(float(np.mean([z["confidence"] for z in zones])), 2)
    return {
        "method": "cv_geometry_perspective",
        "requires_confirmation": True,
        "overall_confidence": overall_conf,
        "warnings": warnings,
        "zones": zones,
    }


def _build_candidate_list(
    *,
    y_top: float,
    y_cross_top: float,
    y_bot: float,
    road_top_l: float,
    road_top_r: float,
    road_mid_l: float,
    road_mid_r: float,
    road_bot_l: float,
    road_bot_r: float,
    road_conf: float,
    cross_conf: float,
    left_fp_conf: float,
    right_fp_conf: float,
    road_evidence: str,
    cross_evidence: str,
    left_fp_evidence: str,
    right_fp_evidence: str,
) -> list[dict[str, Any]]:
    road_pts = [
        [road_top_l, y_top],
        [road_top_r, y_top],
        [road_mid_r, y_cross_top],
        [road_mid_l, y_cross_top],
    ]
    cross_pts = [
        [road_mid_l, y_cross_top],
        [road_mid_r, y_cross_top],
        [road_bot_r, y_bot],
        [road_bot_l, y_bot],
    ]
    left_fp_pts = [
        [0.0, y_top],
        [road_top_l, y_top],
        [road_mid_l, y_cross_top],
        [road_bot_l, y_bot],
        [0.0, y_bot],
    ]
    right_fp_pts = [
        [road_top_r, y_top],
        [1.0, y_top],
        [1.0, y_bot],
        [road_bot_r, y_bot],
        [road_mid_r, y_cross_top],
    ]

    items = [
        ("suggest_crossing", "Pedestrian Crossing", "crossing", cross_pts, cross_conf, cross_evidence),
        ("suggest_road", "Main Road", "road", road_pts, road_conf, road_evidence),
        ("suggest_footpath_left", "Left Footpath", "footpath", left_fp_pts, left_fp_conf, left_fp_evidence),
        ("suggest_footpath_right", "Right Footpath", "footpath", right_fp_pts, right_fp_conf, right_fp_evidence),
    ]
    out: list[dict[str, Any]] = []
    for zid, name, ztype, pts, conf, ev in items:
        level = _conf_level(conf)
        out.append(
            {
                "zone_id": zid,
                "name": name,
                "type": ztype,
                "color_bgr": list(DEFAULT_COLORS_BGR[ztype]),
                "points": pts,
                "polygon_norm": pts,
                "confidence": round(conf, 2),
                "confidence_level": level,
                "uncertain": level == "low",
                "needs_confirmation": True,
                "evidence": ev,
            }
        )
    return out
