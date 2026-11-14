"""Scene-specific polygonal zone persistence and representative video/image frame extraction."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

ALLOWED_ZONE_TYPES = ("road", "footpath", "crossing")
DEFAULT_COLORS_BGR: dict[str, list[int]] = {
    "crossing": [0, 220, 255],
    "road": [255, 180, 0],
    "footpath": [100, 230, 100],
}


def sanitize_scene_id(scene_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(scene_id or "").strip())
    if not cleaned:
        raise ValueError("scene_id cannot be empty")
    return cleaned


def infer_zone_type(name: str, explicit_type: str | None = None) -> str:
    if explicit_type:
        t = str(explicit_type).strip().lower()
        if t not in ALLOWED_ZONE_TYPES:
            raise ValueError(f"Invalid zone type '{explicit_type}'. Allowed: {list(ALLOWED_ZONE_TYPES)}")
        return t
    low = str(name or "").strip().lower()
    if "cross" in low:
        return "crossing"
    if "foot" in low or "sidewalk" in low or "walk" in low:
        return "footpath"
    return "road"


def _polygon_area_norm(points: list[list[float]]) -> float:
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _segments_intersect(a1: list[float], a2: list[float], b1: list[float], b2: list[float]) -> bool:
    def _ccw(p1: list[float], p2: list[float], p3: list[float]) -> float:
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    d1 = _ccw(b1, b2, a1)
    d2 = _ccw(b1, b2, a2)
    d3 = _ccw(a1, a2, b1)
    d4 = _ccw(a1, a2, b2)
    if ((d1 > 1e-9 and d2 < -1e-9) or (d1 < -1e-9 and d2 > 1e-9)) and (
        (d3 > 1e-9 and d4 < -1e-9) or (d3 < -1e-9 and d4 > 1e-9)
    ):
        return True
    return False


def _is_simple_polygon(points: list[list[float]]) -> bool:
    n = len(points)
    if n < 4:
        return True
    for i in range(n):
        a1 = points[i]
        a2 = points[(i + 1) % n]
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            b1 = points[j]
            b2 = points[(j + 1) % n]
            if _segments_intersect(a1, a2, b1, b2):
                return False
    return True


def validate_points(points: list[list[float]], label: str = "zone") -> list[list[float]]:
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError(f"{label} must have >= 3 polygon vertices")
    cleaned: list[list[float]] = []
    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise ValueError(f"{label} vertex must be [x, y]")
        x = float(pt[0])
        y = float(pt[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"{label} has coordinates outside [0, 1]: [{x}, {y}]")
        cleaned.append([round(x, 5), round(y, 5)])
    if _polygon_area_norm(cleaned) < 1e-5:
        raise ValueError(f"{label} polygon area is too small / collinear")
    if not _is_simple_polygon(cleaned):
        raise ValueError(f"{label} polygon edges self-intersect; adjust vertex order")
    return cleaned


def normalize_zone_item(raw: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Normalize zone dict to support both `points` and `polygon_norm` formats."""
    name = str(raw.get("name") or f"Zone {index + 1}").strip()
    if not name:
        raise ValueError("Zone name cannot be empty")
    ztype = infer_zone_type(name, raw.get("type"))
    pts_raw = raw.get("points") if raw.get("points") is not None else raw.get("polygon_norm")
    if pts_raw is None:
        raise ValueError(f"Zone '{name}' missing points / polygon_norm")
    pts = validate_points(pts_raw, label=f"Zone '{name}'")
    zone_id = str(raw.get("zone_id") or f"zone_{index + 1:03d}").strip()
    color = raw.get("color_bgr")
    if not isinstance(color, list) or len(color) != 3:
        color = list(DEFAULT_COLORS_BGR[ztype])
    out: dict[str, Any] = {
        "zone_id": zone_id,
        "name": name,
        "type": ztype,
        "color_bgr": [int(c) for c in color],
        "points": pts,
        "polygon_norm": pts,
    }
    if raw.get("extra_polygons_norm"):
        out["extra_polygons_norm"] = [
            validate_points(ep, label=f"Zone '{name}' extra polygon")
            for ep in raw["extra_polygons_norm"]
        ]
    return out


class ZoneStore:
    """Manages default and media/scene-specific zone configurations and preview frames."""

    def __init__(self, config_dir: Path, previews_dir: Path) -> None:
        self.config_dir = Path(config_dir)
        self.default_zones_path = self.config_dir / "zones.json"
        self.scenes_dir = self.config_dir / "zone_scenes"
        self.previews_dir = Path(previews_dir)
        self.scenes_dir.mkdir(parents=True, exist_ok=True)
        self.previews_dir.mkdir(parents=True, exist_ok=True)

    def _scene_path(self, scene_id: str) -> Path:
        sid = sanitize_scene_id(scene_id)
        if sid == "default":
            return self.default_zones_path
        return self.scenes_dir / f"{sid}.json"

    def has_scene_config(self, scene_id: str) -> bool:
        sid = sanitize_scene_id(scene_id)
        if sid == "default":
            return self.default_zones_path.is_file()
        return (self.scenes_dir / f"{sid}.json").is_file()

    def load_zones(self, scene_id: str | None = None) -> dict[str, Any]:
        sid = sanitize_scene_id(scene_id) if scene_id else "default"
        path = self._scene_path(sid)
        is_scene_specific = sid != "default" and path.is_file()
        if not path.is_file():
            path = self.default_zones_path
        if not path.is_file():
            return {"scene_id": sid, "anchor_mode": "bottom_center", "is_scene_specific": False, "zones": []}
        raw = json.loads(path.read_text(encoding="utf-8"))
        zones = [normalize_zone_item(z, idx) for idx, z in enumerate(raw.get("zones", []))]
        return {
            "scene_id": sid,
            "anchor_mode": str(raw.get("anchor_mode", "bottom_center")),
            "is_scene_specific": is_scene_specific,
            "updated_at": raw.get("updated_at"),
            "calibration": raw.get("calibration"),
            "zones": zones,
        }

    def save_zones(
        self,
        zones: list[dict[str, Any]],
        *,
        scene_id: str | None = None,
        anchor_mode: str = "bottom_center",
        calibration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not zones:
            raise ValueError("At least one zone polygon is required")
        sid = sanitize_scene_id(scene_id) if scene_id else "default"
        existing_cal = None
        target = self._scene_path(sid)
        if target.is_file():
            try:
                existing_cal = json.loads(target.read_text(encoding="utf-8")).get("calibration")
            except Exception:
                existing_cal = None
        final_cal = calibration if calibration is not None else existing_cal

        normalized: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for idx, z in enumerate(zones):
            item = normalize_zone_item(z, idx)
            if item["zone_id"] in seen_ids:
                item["zone_id"] = f"zone_{uuid.uuid4().hex[:6]}"
            seen_ids.add(item["zone_id"])
            normalized.append(item)

        payload: dict[str, Any] = {
            "anchor_mode": anchor_mode,
            "zones": normalized,
        }
        if final_cal:
            payload["calibration"] = final_cal
        if sid != "default":
            payload["scene_id"] = sid
            payload["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return {
            "scene_id": sid,
            "anchor_mode": anchor_mode,
            "is_scene_specific": sid != "default",
            "updated_at": payload.get("updated_at"),
            "calibration": final_cal,
            "zones": normalized,
        }

    def save_calibration(self, scene_id: str, calibration: dict[str, Any] | None) -> dict[str, Any]:
        cfg = self.load_zones(scene_id)
        sid = sanitize_scene_id(scene_id)
        target = self._scene_path(sid)
        payload: dict[str, Any] = {
            "anchor_mode": cfg["anchor_mode"],
            "zones": cfg["zones"],
        }
        if calibration:
            payload["calibration"] = calibration
        if sid != "default":
            payload["scene_id"] = sid
            payload["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return {
            "scene_id": sid,
            "anchor_mode": cfg["anchor_mode"],
            "is_scene_specific": sid != "default",
            "updated_at": payload.get("updated_at"),
            "calibration": calibration,
            "zones": cfg["zones"],
        }

    def reset_scene_zones(self, scene_id: str | None = None) -> dict[str, Any]:
        sid = sanitize_scene_id(scene_id) if scene_id else "default"
        if sid != "default":
            scene_file = self.scenes_dir / f"{sid}.json"
            if scene_file.exists():
                scene_file.unlink()
            return self.load_zones(sid)
        # Reset default config/zones.json cleanly back to canonical road/crossing/footpath starter layout
        starter = [
            {
                "zone_id": "zone_road",
                "name": "road",
                "type": "road",
                "color_bgr": list(DEFAULT_COLORS_BGR["road"]),
                "polygon_norm": [[0.25, 0.35], [0.75, 0.35], [0.85, 0.70], [0.15, 0.70]],
            },
            {
                "zone_id": "zone_crossing",
                "name": "crossing",
                "type": "crossing",
                "color_bgr": list(DEFAULT_COLORS_BGR["crossing"]),
                "polygon_norm": [[0.15, 0.70], [0.85, 0.70], [0.90, 0.92], [0.10, 0.92]],
            },
            {
                "zone_id": "zone_footpath",
                "name": "footpath",
                "type": "footpath",
                "color_bgr": list(DEFAULT_COLORS_BGR["footpath"]),
                "polygon_norm": [[0.00, 0.35], [0.25, 0.35], [0.15, 0.70], [0.10, 0.92], [0.00, 0.92]],
                "extra_polygons_norm": [
                    [[0.75, 0.35], [1.00, 0.35], [1.00, 0.92], [0.90, 0.92], [0.85, 0.70]]
                ],
            },
        ]
        return self.save_zones(starter, scene_id="default")

    def list_scenes(self) -> list[dict[str, Any]]:
        scenes: list[dict[str, Any]] = []
        default_cfg = self.load_zones("default")
        scenes.append(
            {
                "scene_id": "default",
                "zone_count": len(default_cfg["zones"]),
                "calibrated": bool(default_cfg.get("calibration")),
                "is_scene_specific": False,
            }
        )
        for p in sorted(self.scenes_dir.glob("*.json")):
            sid = p.stem
            try:
                cfg = self.load_zones(sid)
                scenes.append(
                    {
                        "scene_id": sid,
                        "zone_count": len(cfg["zones"]),
                        "calibrated": bool(cfg.get("calibration")),
                        "is_scene_specific": True,
                        "updated_at": cfg.get("updated_at"),
                    }
                )
            except Exception:
                continue
        return scenes

    def add_zone(self, zone_data: dict[str, Any], *, scene_id: str | None = None) -> dict[str, Any]:
        cfg = self.load_zones(scene_id)
        zones = list(cfg["zones"])
        if not zone_data.get("zone_id"):
            zone_data = {**zone_data, "zone_id": f"zone_{len(zones) + 1:03d}_{uuid.uuid4().hex[:4]}"}
        new_item = normalize_zone_item(zone_data, len(zones))
        # Replace if same zone_id exists, otherwise append
        replaced = False
        for i, existing in enumerate(zones):
            if existing["zone_id"] == new_item["zone_id"]:
                zones[i] = new_item
                replaced = True
                break
        if not replaced:
            zones.append(new_item)
        saved = self.save_zones(zones, scene_id=scene_id, anchor_mode=cfg["anchor_mode"])
        return {"zone": new_item, "config": saved}

    def update_zone(self, zone_id: str, patch: dict[str, Any], *, scene_id: str | None = None) -> dict[str, Any]:
        cfg = self.load_zones(scene_id)
        zones = list(cfg["zones"])
        target_idx = next((i for i, z in enumerate(zones) if z["zone_id"] == zone_id), None)
        if target_idx is None:
            raise KeyError(f"Zone '{zone_id}' not found in scene '{cfg['scene_id']}'")
        merged = {**zones[target_idx], **{k: v for k, v in patch.items() if v is not None}, "zone_id": zone_id}
        # If points were updated in patch, keep polygon_norm aligned (and vice versa)
        if patch.get("points") is not None:
            merged["polygon_norm"] = patch["points"]
        elif patch.get("polygon_norm") is not None:
            merged["points"] = patch["polygon_norm"]
        # Update default color if type changed and color wasn't explicitly passed
        if patch.get("type") is not None and patch.get("color_bgr") is None:
            t = infer_zone_type(merged["name"], patch["type"])
            merged["color_bgr"] = list(DEFAULT_COLORS_BGR[t])
        updated_item = normalize_zone_item(merged, target_idx)
        zones[target_idx] = updated_item
        saved = self.save_zones(zones, scene_id=scene_id, anchor_mode=cfg["anchor_mode"])
        return {"zone": updated_item, "config": saved}

    def delete_zone(self, zone_id: str, *, scene_id: str | None = None) -> dict[str, Any]:
        cfg = self.load_zones(scene_id)
        zones = [z for z in cfg["zones"] if z["zone_id"] != zone_id]
        if len(zones) == len(cfg["zones"]):
            raise KeyError(f"Zone '{zone_id}' not found in scene '{cfg['scene_id']}'")
        if not zones:
            raise ValueError("Cannot delete the last remaining zone; at least one zone is required")
        saved = self.save_zones(zones, scene_id=scene_id, anchor_mode=cfg["anchor_mode"])
        return {"deleted_zone_id": zone_id, "config": saved}

    def extract_preview_from_bytes(
        self,
        content: bytes,
        filename: str,
        *,
        is_video: bool,
    ) -> dict[str, Any]:
        """Save representative frame for uploaded image/video and derive deterministic scene_id."""
        digest = hashlib.sha256(content).hexdigest()[:12]
        stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(filename).stem)[:32] or "scene"
        scene_id = f"{stem}_{digest}"
        preview_jpg = self.previews_dir / f"{scene_id}.jpg"
        meta_json = self.previews_dir / f"{scene_id}.json"

        if is_video:
            tmp_vid = self.previews_dir / f"{scene_id}_src{Path(filename).suffix.lower() or '.mp4'}"
            tmp_vid.write_bytes(content)
            try:
                cap = cv2.VideoCapture(str(tmp_vid))
                if not cap.isOpened():
                    raise ValueError("Uploaded video cannot be opened by OpenCV")
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                target_idx = total_frames // 2 if total_frames > 2 else 0
                if target_idx > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    target_idx = 0
                cap.release()
                if not ok or frame is None:
                    raise ValueError("Could not decode representative frame from video")
                cv2.imwrite(str(preview_jpg), frame)
            finally:
                if tmp_vid.exists():
                    tmp_vid.unlink(missing_ok=True)
        else:
            import numpy as np

            arr = np.frombuffer(content, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Invalid or corrupted image: {filename}")
            cv2.imwrite(str(preview_jpg), frame)
            total_frames = 1
            fps = 0.0
            target_idx = 0

        h, w = frame.shape[:2]
        meta = {
            "scene_id": scene_id,
            "filename": Path(filename).name,
            "media_type": "video" if is_video else "image",
            "width": int(w),
            "height": int(h),
            "aspect_ratio": round(w / h, 5) if h > 0 else 1.0,
            "representative_frame_index": int(target_idx),
            "total_video_frames": int(total_frames),
            "video_fps": round(fps, 2),
            "preview_url": f"/api/v1/zones/preview/{scene_id}",
            "has_saved_zones": self.has_scene_config(scene_id),
        }
        meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def extract_preview_from_file(self, media_path: Path, scene_id: str, is_video: bool) -> dict[str, Any]:
        """Create a preview frame from an existing local image/video path (e.g. inspection input)."""
        sid = sanitize_scene_id(scene_id)
        preview_jpg = self.previews_dir / f"{sid}.jpg"
        meta_json = self.previews_dir / f"{sid}.json"
        if preview_jpg.is_file() and meta_json.is_file():
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            meta["has_saved_zones"] = self.has_scene_config(sid)
            return meta
        content = Path(media_path).read_bytes()
        meta = self.extract_preview_from_bytes(content, Path(media_path).name, is_video=is_video)
        # Also alias under requested sid if different
        if meta["scene_id"] != sid:
            alias_jpg = self.previews_dir / f"{sid}.jpg"
            alias_jpg.write_bytes((self.previews_dir / f"{meta['scene_id']}.jpg").read_bytes())
            meta["scene_id"] = sid
            meta["preview_url"] = f"/api/v1/zones/preview/{sid}"
            meta["has_saved_zones"] = self.has_scene_config(sid)
            (self.previews_dir / f"{sid}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def get_preview_path(self, scene_id: str) -> Path:
        sid = sanitize_scene_id(scene_id)
        return self.previews_dir / f"{sid}.jpg"

    @staticmethod
    def suggest_heuristic_zones() -> list[dict[str, Any]]:
        """Optional geometric starter suggestions (must be explicitly confirmed by user)."""
        return [
            {
                "zone_id": "suggest_crossing",
                "name": "Pedestrian Crossing",
                "type": "crossing",
                "color_bgr": list(DEFAULT_COLORS_BGR["crossing"]),
                "points": [[0.20, 0.70], [0.80, 0.70], [0.88, 0.96], [0.12, 0.96]],
                "polygon_norm": [[0.20, 0.70], [0.80, 0.70], [0.88, 0.96], [0.12, 0.96]],
                "ai_suggested": False,
                "heuristic_suggestion": True,
            },
            {
                "zone_id": "suggest_road",
                "name": "Main Road",
                "type": "road",
                "color_bgr": list(DEFAULT_COLORS_BGR["road"]),
                "points": [[0.26, 0.36], [0.74, 0.36], [0.80, 0.70], [0.20, 0.70]],
                "polygon_norm": [[0.26, 0.36], [0.74, 0.36], [0.80, 0.70], [0.20, 0.70]],
                "ai_suggested": False,
                "heuristic_suggestion": True,
            },
            {
                "zone_id": "suggest_footpath_left",
                "name": "Left Footpath",
                "type": "footpath",
                "color_bgr": list(DEFAULT_COLORS_BGR["footpath"]),
                "points": [[0.00, 0.36], [0.26, 0.36], [0.20, 0.70], [0.12, 0.96], [0.00, 0.96]],
                "polygon_norm": [[0.00, 0.36], [0.26, 0.36], [0.20, 0.70], [0.12, 0.96], [0.00, 0.96]],
                "ai_suggested": False,
                "heuristic_suggestion": True,
            },
            {
                "zone_id": "suggest_footpath_right",
                "name": "Right Footpath",
                "type": "footpath",
                "color_bgr": list(DEFAULT_COLORS_BGR["footpath"]),
                "points": [[0.74, 0.36], [1.00, 0.36], [1.00, 0.96], [0.88, 0.96], [0.80, 0.70]],
                "polygon_norm": [[0.74, 0.36], [1.00, 0.36], [1.00, 0.96], [0.88, 0.96], [0.80, 0.70]],
                "ai_suggested": False,
                "heuristic_suggestion": True,
            },
        ]
