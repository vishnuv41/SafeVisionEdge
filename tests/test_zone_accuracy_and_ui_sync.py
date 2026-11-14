"""Comprehensive validation tests for zone occupancy accuracy, calibration, confidence-aware suggestion, and UI sync."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

PKG_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PKG_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import api as api_module  # noqa: E402
from decode import Detection  # noqa: E402
from safety_pulse import analyze_frame, compute_temporal_occupancy  # noqa: E402
from zone_analysis import (  # noqa: E402
    ZoneAnalyzer,
    compute_homography_calibration,
    project_zone_area_m2,
)
from zone_store import validate_points  # noqa: E402
from zone_suggest import estimate_scene_zones  # noqa: E402


DEFAULT_4_ZONES = [
    {
        "zone_id": "crossing",
        "name": "Pedestrian Crossing",
        "type": "crossing",
        "points": [[0.20, 0.60], [0.80, 0.60], [0.90, 0.90], [0.10, 0.90]],
    },
    {
        "zone_id": "road",
        "name": "Main Road",
        "type": "road",
        "points": [[0.30, 0.40], [0.70, 0.40], [0.80, 0.60], [0.20, 0.60]],
    },
    {
        "zone_id": "left_footpath",
        "name": "Left Footpath",
        "type": "footpath",
        "points": [[0.00, 0.40], [0.20, 0.40], [0.10, 0.90], [0.00, 0.90]],
    },
    {
        "zone_id": "right_footpath",
        "name": "Right Footpath",
        "type": "footpath",
        "points": [[0.80, 0.40], [1.00, 0.40], [1.00, 0.90], [0.90, 0.90]],
    },
]


def _det(cid: int, cname: str, xyxy: list[float], conf: float = 0.9) -> Detection:
    return Detection(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(conf), int(cid), cname)


class ZoneAccuracyAndUiSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.safety_cfg = json.loads((PKG_ROOT / "config" / "safety_config.json").read_text(encoding="utf-8"))
        cls.zones_cfg = {
            "anchor_mode": "bottom_center",
            "zones": DEFAULT_4_ZONES,
        }
        cls.analyzer = ZoneAnalyzer(cls.zones_cfg)

    def test_01_ground_contact_bottom_center_vs_box_center(self) -> None:
        """A tall person standing in crossing whose bbox centre falls in road must assign to crossing."""
        det = _det(0, "person", [450.0, 250.0, 550.0, 750.0], 0.92)
        res = self.analyzer.assign_zones_detailed([det], (1000, 1000))
        rec = res["assignment_records"][0]
        self.assertEqual(rec["anchor_px"], [500.0, 750.0])
        self.assertEqual(rec["zone_name"], "Pedestrian Crossing")
        self.assertEqual(rec["zone_type"], "crossing")
        self.assertEqual(res["zone_counts"]["Pedestrian Crossing"].get("person", 0), 1)
        self.assertEqual(res["zone_counts"]["Main Road"].get("person", 0), 0)

    def test_02_boundary_inclusion_and_exclusive_priority(self) -> None:
        """Point directly on road/crossing boundary y=0.60 resolves deterministically once to crossing."""
        det = _det(4, "car", [460.0, 480.0, 540.0, 600.0], 0.88)
        res = self.analyzer.assign_zones_detailed([det], (1000, 1000))
        rec = res["assignment_records"][0]
        self.assertEqual(rec["zone_name"], "Pedestrian Crossing")
        self.assertEqual(rec["zone_type"], "crossing")
        self.assertTrue(rec["on_boundary"])
        # Zero double-counting: total across all zones == 1
        total_assigned = sum(v["total"] for v in res["zone_counts"].values())
        self.assertEqual(total_assigned, 1)
        self.assertEqual(res["zone_counts"]["Main Road"]["total"], 0)
        self.assertEqual(res["zone_counts"]["Pedestrian Crossing"]["total"], 1)
        self.assertEqual(res["boundary_assignments"], 1)

    def test_03_overlapping_polygons_never_double_count_and_footpath_splits_aggregate(self) -> None:
        """Overlapping custom polygons assign once by priority; Left+Right footpaths sum cleanly."""
        cfg = {
            "anchor_mode": "bottom_center",
            "zones": [
                {"zone_id": "road", "name": "Road", "type": "road", "points": [[0.2, 0.4], [0.8, 0.4], [0.8, 0.9], [0.2, 0.9]]},
                {"zone_id": "crossing", "name": "Crossing", "type": "crossing", "points": [[0.3, 0.5], [0.7, 0.5], [0.7, 0.8], [0.3, 0.8]]},
                {"zone_id": "left_fp", "name": "Left FP", "type": "footpath", "points": [[0.0, 0.4], [0.15, 0.4], [0.15, 0.9], [0.0, 0.9]]},
                {"zone_id": "right_fp", "name": "Right FP", "type": "footpath", "points": [[0.85, 0.4], [1.0, 0.4], [1.0, 0.9], [0.85, 0.9]]},
            ],
        }
        pulse = analyze_frame(
            [
                _det(0, "person", [480, 500, 520, 650], 0.9),     # inside crossing & road -> crossing once
                _det(0, "person", [50, 550, 90, 700], 0.85),      # inside Left FP
                _det(2, "motorcycle", [900, 550, 940, 700], 0.8), # inside Right FP
            ],
            (1000, 1000),
            ZoneAnalyzer(cfg),
            self.safety_cfg,
        )
        self.assertEqual(pulse["persons_crossing"], 1)
        self.assertEqual(pulse["persons_road"], 0)
        self.assertEqual(pulse["persons_footpath"], 1)
        self.assertEqual(pulse["vehicles_footpath"], 1)
        self.assertEqual(pulse["category_class_counts"]["footpath"]["person"], 1)
        self.assertEqual(pulse["category_class_counts"]["footpath"]["motorcycle"], 1)

    def test_04_seven_class_per_zone_and_unassigned_breakdown(self) -> None:
        """All 7 classes and outside-zone unassigned detections are tallied without silent loss."""
        dets = [
            _det(0, "person", [480, 600, 520, 750], 0.95),        # Pedestrian Crossing
            _det(1, "bicycle", [480, 350, 520, 500], 0.90),       # Main Road
            _det(2, "motorcycle", [430, 350, 470, 500], 0.88),    # Main Road
            _det(3, "auto_rickshaw", [530, 350, 570, 500], 0.87), # Main Road
            _det(4, "car", [580, 350, 620, 500], 0.91),           # Main Road
            _det(5, "bus", [380, 350, 420, 500], 0.89),           # Main Road
            _det(6, "truck", [400, 600, 440, 750], 0.86),         # Pedestrian Crossing
            _det(0, "person", [480, 50, 520, 150], 0.77),         # Unassigned
            _det(4, "car", [480, 920, 520, 980], 0.75),           # Unassigned
        ]
        res = self.analyzer.assign_zones_detailed(dets, (1000, 1000))
        road_cls = res["zone_counts"]["Main Road"]
        self.assertEqual(road_cls.get("bicycle", 0), 1)
        self.assertEqual(road_cls.get("motorcycle", 0), 1)
        self.assertEqual(road_cls.get("auto_rickshaw", 0), 1)
        self.assertEqual(road_cls.get("car", 0), 1)
        self.assertEqual(road_cls.get("bus", 0), 1)
        self.assertEqual(res["zone_counts"]["Pedestrian Crossing"].get("person", 0), 1)
        self.assertEqual(res["zone_counts"]["Pedestrian Crossing"].get("truck", 0), 1)
        self.assertEqual(res["unassigned_counts"]["total"], 2)
        self.assertEqual(res["unassigned_counts"]["person"], 1)
        self.assertEqual(res["unassigned_counts"]["vehicles"], 1)
        self.assertEqual(res["unassigned_counts"]["car"], 1)

    def test_05_normalized_resolution_invariance(self) -> None:
        """Scaling image and detections uniformly preserves exact zone assignment."""
        norm_boxes = [
            (0, "person", [0.48, 0.55, 0.52, 0.75]),
            (4, "car", [0.46, 0.35, 0.54, 0.50]),
            (0, "person", [0.03, 0.50, 0.08, 0.70]),
            (2, "motorcycle", [0.94, 0.50, 0.98, 0.70]),
        ]

        def _run_for(w: int, h: int):
            ds = [_det(cid, cname, [x1 * w, y1 * h, x2 * w, y2 * h], 0.9) for cid, cname, (x1, y1, x2, y2) in norm_boxes]
            return self.analyzer.assign_zones_detailed(ds, (h, w))

        low = _run_for(640, 360)
        high = _run_for(1920, 1080)
        self.assertEqual(low["assignments"], high["assignments"])
        self.assertEqual(low["zone_counts"], high["zone_counts"])

    def test_06_invalid_self_intersecting_and_collinear_polygon_rejection(self) -> None:
        """validate_points must reject bowtie self-intersecting and collinear zero-area polygons."""
        with self.assertRaises(ValueError):
            validate_points([[0.2, 0.2], [0.8, 0.8], [0.8, 0.2], [0.2, 0.8]])
        with self.assertRaises(ValueError):
            validate_points([[0.1, 0.5], [0.5, 0.5], [0.9, 0.5]])
        with self.assertRaises(ValueError):
            validate_points([[0.1, 0.2], [1.05, 0.2], [0.5, 0.8]])

    def test_07_homography_calibration_and_uncalibrated_honesty(self) -> None:
        """Uncalibrated scenes report no fake ped/m2; 4-pt calibration computes metric area and density."""
        pulse_uncal = analyze_frame(
            [_det(0, "person", [480, 550, 520, 750], 0.9)],
            (1000, 1000),
            self.analyzer,
            self.safety_cfg,
        )
        self.assertFalse(pulse_uncal["calibrated_densities"]["calibrated"])
        self.assertFalse(pulse_uncal["calibrated_densities"]["physical_density_available"])
        self.assertEqual(pulse_uncal["calibrated_densities"]["per_zone"], {})

        cal = compute_homography_calibration(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            frame_width=1000,
            frame_height=1000,
        )
        area_m2 = project_zone_area_m2(DEFAULT_4_ZONES[0], np.array(cal["homography_norm_to_m"], dtype=np.float64))
        self.assertIsNotNone(area_m2)
        self.assertAlmostEqual(float(area_m2), 21.0, places=2)

        analyzer_cal = ZoneAnalyzer({**self.zones_cfg, "calibration": cal})
        det_counts = analyzer_cal.assign_zones_detailed([_det(0, "person", [480, 550, 520, 750], 0.9)], (1000, 1000))
        dens = analyzer_cal.compute_calibrated_densities(det_counts["zone_counts"])
        self.assertTrue(dens["calibrated"])
        self.assertIn("Pedestrian Crossing", dens["per_zone"])
        self.assertAlmostEqual(dens["per_zone"]["Pedestrian Crossing"]["area_m2"], 21.0, places=2)
        self.assertAlmostEqual(dens["per_zone"]["Pedestrian Crossing"]["pedestrians_per_m2"], round(1.0 / 21.0, 4), places=4)

    def test_08_confidence_aware_zone_suggest_honesty(self) -> None:
        """Zone suggestion flags uncertainty on featureless frames and detects evidence on structured frames."""
        blank = np.full((360, 640, 3), 110, dtype=np.uint8)
        blank_res = estimate_scene_zones(blank)
        self.assertTrue(blank_res["requires_confirmation"])
        self.assertGreaterEqual(len(blank_res["warnings"]), 1)
        crossing_z = next(z for z in blank_res["zones"] if z["type"] == "crossing")
        self.assertTrue(crossing_z["uncertain"])
        self.assertLess(crossing_z["confidence"], 0.65)

        scene = np.full((360, 640, 3), 90, dtype=np.uint8)
        cv2.line(scene, (180, 140), (60, 355), (220, 220, 220), 4)
        cv2.line(scene, (460, 140), (580, 355), (220, 220, 220), 4)
        for x in range(160, 480, 36):
            cv2.rectangle(scene, (x, 240), (x + 18, 305), (240, 240, 240), -1)
        marked_res = estimate_scene_zones(scene)
        self.assertTrue(marked_res["requires_confirmation"])
        self.assertEqual(len(marked_res["zones"]), 4)
        marked_crossing = next(z for z in marked_res["zones"] if z["type"] == "crossing")
        self.assertGreater(marked_crossing["confidence"], crossing_z["confidence"])

    def test_09_temporal_occupancy_summary(self) -> None:
        """Multi-frame temporal summary computes exact averages, peaks, and percentages."""
        frames = [
            {
                "total": 8,
                "counts": {"person": 3, "car": 5},
                "co_presence": True,
                "zones": {
                    "road": {"total": 4},
                    "crossing": {"total": 3},
                    "footpath": {"total": 1},
                    "unassigned": {"total": 0},
                },
            },
            {
                "total": 6,
                "counts": {"person": 0, "car": 6},
                "co_presence": False,
                "zones": {
                    "road": {"total": 5},
                    "crossing": {"total": 1},
                    "footpath": {"total": 0},
                    "unassigned": {"total": 0},
                },
            },
        ]
        summary = compute_temporal_occupancy(frames)
        self.assertEqual(summary["frames_analyzed"], 2)
        self.assertEqual(summary["pedestrian_presence_frames"], 1)
        self.assertEqual(summary["pedestrian_presence_pct"], 50.0)
        self.assertEqual(summary["co_presence_frames"], 1)
        self.assertEqual(summary["co_presence_pct"], 50.0)
        self.assertEqual(summary["avg_occupancy"]["crossing"], 2.0)
        self.assertEqual(summary["peak_occupancy"]["crossing"], 3)
        self.assertEqual(summary["avg_occupancy"]["road"], 4.5)
        self.assertEqual(summary["peak_occupancy"]["road"], 5)

    def test_10_api_calibration_suggest_reset_and_frontend_bundle_sync(self) -> None:
        """End-to-end API checks for suggest, reset, 4-pt calibration, and UI route cleanup."""
        scene_id = "test_tmp_sync_scene"
        zones_dir = PKG_ROOT / "config" / "zone_scenes"
        try:
            with TestClient(api_module.app) as client:
                r_bad = client.post(
                    "/api/v1/zones",
                    json={
                        "scene_id": scene_id,
                        "name": "Bowtie",
                        "type": "road",
                        "points": [[0.2, 0.2], [0.8, 0.8], [0.8, 0.2], [0.2, 0.8]],
                    },
                )
                self.assertEqual(r_bad.status_code, 400)

                r_save = client.post(
                    "/api/v1/zones",
                    json={
                        "scene_id": scene_id,
                        "zones": [
                            {
                                "zone_id": "custom_r",
                                "name": "Custom Road",
                                "type": "road",
                                "points": [[0.2, 0.3], [0.8, 0.3], [0.85, 0.9], [0.15, 0.9]],
                            }
                        ],
                    },
                )
                self.assertEqual(r_save.status_code, 200)

                r_cal = client.post(
                    "/api/v1/zones/calibration",
                    json={
                        "scene_id": scene_id,
                        "image_points_norm": [[0.2, 0.3], [0.8, 0.3], [0.85, 0.9], [0.15, 0.9]],
                        "world_points_m": [[0.0, 0.0], [10.0, 0.0], [10.0, 12.0], [0.0, 12.0]],
                    },
                )
                self.assertEqual(r_cal.status_code, 200)
                cal_res = r_cal.json()
                self.assertIsNotNone(cal_res["calibration"])
                self.assertIn("homography_norm_to_m", cal_res["calibration"])

                r_del_cal = client.delete(f"/api/v1/zones/calibration?scene_id={scene_id}")
                self.assertEqual(r_del_cal.status_code, 200)
                self.assertIsNone(r_del_cal.json()["calibration"])

                r_reset = client.post(f"/api/v1/zones/reset?scene_id={scene_id}")
                self.assertEqual(r_reset.status_code, 200)
                self.assertEqual(len(r_reset.json()["zones"]), 4)

                r_sug = client.post(f"/api/v1/zones/suggest?scene_id={scene_id}")
                self.assertEqual(r_sug.status_code, 200)
                self.assertTrue(r_sug.json()["requires_confirmation"])
                self.assertEqual(len(client.get(f"/api/v1/zones?scene_id={scene_id}").json()["zones"]), 4)

                r_bundle = client.get("/assets/index-BBiwelqg.js")
                self.assertEqual(r_bundle.status_code, 200)
                js = r_bundle.text
                self.assertNotIn('{id:"zones",label:"Zone editor"}', js)
                self.assertNotIn('{id:"npu",label:"NPU & Benchmark"}', js)
                self.assertNotIn("Edit zones for scene", js)
                self.assertNotIn('V==="zones"', js)
                self.assertNotIn('V==="npu"', js)
                self.assertIn("Edit polygons", js)
                self.assertIn("ZoneEditorModal", js)

                r_ext = client.get("/assets/sv_extensions.js")
                self.assertEqual(r_ext.status_code, 200)
                self.assertIn("ZoneEditorModal", r_ext.text)
                self.assertIn("Edit polygons", r_ext.text)
        finally:
            (zones_dir / f"{scene_id}.json").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
