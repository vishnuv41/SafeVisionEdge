"""Automated functional, safety-intelligence, and real-.pte integration audit suite."""

from __future__ import annotations

import csv
import importlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PKG_ROOT / "src"
CONFIG_DIR = PKG_ROOT / "config"
MODEL_PATH = PKG_ROOT / "models" / "custom_model.pte"
FIXTURES_DIR = PKG_ROOT / "tests" / "fixtures"
SAMPLE_IMAGE = FIXTURES_DIR / "sample_road_1.jpg"
SAMPLE_IMAGE_2 = FIXTURES_DIR / "sample_road_2.jpg"


def _write_temp_mp4(out_path: Path, frame_paths: list[Path], num_frames: int = 12, fps: float = 10.0) -> Path:
    imgs = [cv2.imread(str(p)) for p in frame_paths]
    if any(im is None for im in imgs):
        raise RuntimeError("Failed to read fixture images for MP4 synthesis")
    h, w = imgs[0].shape[:2]
    resized = [cv2.resize(im, (w, h)) if im.shape[:2] != (h, w) else im for im in imgs]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")
    try:
        for i in range(num_frames):
            writer.write(resized[i % len(resized)])
    finally:
        writer.release()
    return out_path

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from decode import Detection, decode_predictions, preprocess_bgr  # noqa: E402
from zone_analysis import ZoneAnalyzer  # noqa: E402


def _det(cls_id: int, name: str, x1: float, y1: float, x2: float, y2: float, conf: float = 0.85) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, class_id=cls_id, class_name=name)


# Reference 1000x1000 normalized zones covering standard road, crossing, footpath layout:
# footpath: top/left band [0,0.4]x[0.5,1.0], road: [0.4,1.0]x[0.0,0.65], crossing: [0.4,1.0]x[0.65,1.0]
TEST_ZONES_CFG = {
    "anchor_mode": "bottom_center",
    "zones": [
        {
            "name": "crossing",
            "color_bgr": [0, 220, 255],
            "polygon_norm": [[0.40, 0.65], [1.00, 0.65], [1.00, 1.00], [0.40, 1.00]],
        },
        {
            "name": "road",
            "color_bgr": [255, 180, 0],
            "polygon_norm": [[0.40, 0.00], [1.00, 0.00], [1.00, 0.65], [0.40, 0.65]],
        },
        {
            "name": "footpath",
            "color_bgr": [100, 255, 100],
            "polygon_norm": [[0.00, 0.50], [0.40, 0.50], [0.40, 1.00], [0.00, 1.00]],
        },
    ],
}


def _import_safety_module():
    return importlib.import_module("safety_pulse")


class TestExistingPipelineAndOfflineCompliance(unittest.TestCase):
    """Audit existing preprocessing, decoding, offline safety, zones, and real .pte runtime."""

    def test_00a_class_names_has_seven_classes_including_auto_rickshaw(self) -> None:
        names = json.loads((CONFIG_DIR / "class_names.json").read_text(encoding="utf-8"))
        self.assertEqual(
            names,
            {
                "0": "person",
                "1": "bicycle",
                "2": "motorcycle",
                "3": "auto_rickshaw",
                "4": "car",
                "5": "bus",
                "6": "truck",
            },
        )

    def test_00b_preprocess_bgr_to_rgb_nchw_float32(self) -> None:
        # Pure blue pixel in BGR = (255, 0, 0) must become RGB (0, 0, 1.0) in channel 2
        img_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        img_bgr[:, :, 0] = 255  # Blue in BGR
        tensor, scale, pad = preprocess_bgr(img_bgr, imgsz=640)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertAlmostEqual(scale, 1.0, places=5)
        self.assertEqual(pad, (0.0, 80.0))
        # Center region should have R=0, G=0, B=1.0
        self.assertAlmostEqual(float(tensor[0, 0, 320, 320]), 0.0, places=5)
        self.assertAlmostEqual(float(tensor[0, 1, 320, 320]), 0.0, places=5)
        self.assertAlmostEqual(float(tensor[0, 2, 320, 320]), 1.0, places=5)

    def test_00c_decode_nms_suppresses_duplicates_and_handles_empty(self) -> None:
        names = {0: "person", 1: "bicycle", 2: "motorcycle", 3: "auto_rickshaw", 4: "car", 5: "bus", 6: "truck"}
        empty = np.zeros((1, 11, 8400), dtype=np.float32)
        self.assertEqual(decode_predictions(empty, (720, 1280), 0.5, (0.0, 140.0), names), [])

        # Two overlapping car predictions + one distinct auto_rickshaw prediction
        raw = np.zeros((1, 11, 8400), dtype=np.float32)
        raw[0, :4, 0] = [320, 320, 100, 100]
        raw[0, 4 + 4, 0] = 0.90  # car
        raw[0, :4, 1] = [322, 322, 100, 100]
        raw[0, 4 + 4, 1] = 0.70  # duplicate car
        raw[0, :4, 2] = [100, 320, 80, 80]
        raw[0, 4 + 3, 2] = 0.80  # auto_rickshaw
        dets = decode_predictions(raw, (640, 640), 1.0, (0.0, 0.0), names, conf_thres=0.25, iou_thres=0.45)
        self.assertEqual(len(dets), 2)
        self.assertEqual([d.class_name for d in dets], ["car", "auto_rickshaw"])

    def test_00d_offline_no_network_dependencies_in_src(self) -> None:
        forbidden = re.compile(r"\b(requests|urllib|http\.client|ultralytics|torch\.hub|huggingface_hub)\b")
        for py_file in sorted(SRC_DIR.glob("*.py")):
            text = py_file.read_text(encoding="utf-8")
            self.assertIsNone(forbidden.search(text), f"Network/ultralytics reference in {py_file.name}")

    def test_00e_configured_zones_include_road_crossing_footpath(self) -> None:
        from zone_analysis import infer_zone_type_from_def
        zones_cfg = json.loads((CONFIG_DIR / "zones.json").read_text(encoding="utf-8"))
        zone_types = {infer_zone_type_from_def(z) for z in zones_cfg.get("zones", [])}
        self.assertTrue({"road", "crossing", "footpath"}.issubset(zone_types), f"Actual zone types: {zone_types}")


class TestSafetyIntelligenceRules(unittest.TestCase):
    """18 deterministic synthetic safety-intelligence, zone, CSV, heatmap, and blur tests."""

    def test_01_empty_frame(self) -> None:
        sp = _import_safety_module()
        out = sp.analyze_frame([], (1000, 1000), ZoneAnalyzer(TEST_ZONES_CFG))
        self.assertEqual(out["total_detections"], 0)
        self.assertEqual(out["density_class"], "Low")
        self.assertEqual(out["density_score"], 0)
        self.assertEqual(out["co_presence"], "NO")
        self.assertEqual(out["co_presence_score"], 0)
        self.assertEqual(out["vru_share"], 0.0)
        self.assertEqual(out["vru_score"], 0)
        self.assertEqual(out["safety_pulse"], 0)
        self.assertEqual(out["safety_pulse_label"], "Low")

    def test_02_one_pedestrian_in_crossing(self) -> None:
        sp = _import_safety_module()
        dets = [_det(0, "person", 650, 700, 750, 900)]  # bottom-center (700, 900) in crossing
        out = sp.analyze_frame(dets, (1000, 1000), ZoneAnalyzer(TEST_ZONES_CFG))
        self.assertEqual(out["persons_crossing"], 1)
        self.assertEqual(out["vehicles_crossing"], 0)
        self.assertEqual(out["co_presence"], "NO")
        self.assertAlmostEqual(out["vru_share"], 1.0, places=4)
        self.assertEqual(out["safety_pulse"], 1)
        self.assertEqual(out["safety_pulse_label"], "Low")

    def test_03_one_car_in_crossing(self) -> None:
        sp = _import_safety_module()
        dets = [_det(4, "car", 600, 700, 800, 900)]  # bottom-center (700, 900) in crossing
        out = sp.analyze_frame(dets, (1000, 1000), ZoneAnalyzer(TEST_ZONES_CFG))
        self.assertEqual(out["persons_crossing"], 0)
        self.assertEqual(out["vehicles_crossing"], 1)
        self.assertEqual(out["co_presence"], "NO")
        self.assertEqual(out["vru_share"], 0.0)
        self.assertEqual(out["safety_pulse"], 0)

    def test_04_pedestrian_and_car_together_in_crossing_no_frame_bleed(self) -> None:
        sp = _import_safety_module()
        za = ZoneAnalyzer(TEST_ZONES_CFG)
        dets = [
            _det(0, "person", 500, 700, 560, 850),
            _det(4, "car", 650, 700, 850, 900),
        ]
        out1 = sp.analyze_frame(dets, (1000, 1000), za)
        self.assertEqual(out1["co_presence"], "YES")
        self.assertEqual(out1["co_presence_score"], 2)
        self.assertAlmostEqual(out1["vru_share"], 0.5, places=4)
        self.assertEqual(out1["vru_score"], 1)
        self.assertEqual(out1["safety_pulse"], 3)
        self.assertEqual(out1["safety_pulse_label"], "Medium")
        # Next frame empty -> must NOT carry over prior detections
        out2 = sp.analyze_frame([], (1000, 1000), za)
        self.assertEqual(out2["co_presence"], "NO")
        self.assertEqual(out2["safety_pulse"], 0)

    def test_05_pedestrian_on_footpath_and_vehicle_on_road(self) -> None:
        sp = _import_safety_module()
        dets = [
            _det(0, "person", 150, 600, 250, 800),  # bottom-center (200, 800) in footpath
            _det(4, "car", 600, 200, 800, 400),     # bottom-center (700, 400) in road
        ]
        out = sp.analyze_frame(dets, (1000, 1000), ZoneAnalyzer(TEST_ZONES_CFG))
        self.assertEqual(out["persons_footpath"], 1)
        self.assertEqual(out["vehicles_road"], 1)
        self.assertEqual(out["co_presence"], "NO")
        self.assertEqual(out["co_presence_score"], 0)
        self.assertAlmostEqual(out["vru_share"], 0.5, places=4)
        self.assertEqual(out["safety_pulse"], 1)

    def test_06_four_vehicles_low_density(self) -> None:
        sp = _import_safety_module()
        dets = [_det(4, "car", 600, 100, 700, 300) for _ in range(4)]
        out = sp.analyze_frame(dets, (1000, 1000), ZoneAnalyzer(TEST_ZONES_CFG))
        self.assertEqual(out["density_class"], "Low")
        self.assertEqual(out["density_score"], 0)

    def test_07_five_vehicles_moderate_density(self) -> None:
        sp = _import_safety_module()
        dets = [_det(4, "car", 600, 100, 700, 300) for _ in range(5)]
        out = sp.analyze_frame(dets, (1000, 1000), ZoneAnalyzer(TEST_ZONES_CFG))
        self.assertEqual(out["density_class"], "Moderate")
        self.assertEqual(out["density_score"], 1)

    def test_08_nine_vehicles_moderate_density(self) -> None:
        sp = _import_safety_module()
        dets = [_det(4, "car", 600, 100, 700, 300) for _ in range(9)]
        out = sp.analyze_frame(dets, (1000, 1000), ZoneAnalyzer(TEST_ZONES_CFG))
        self.assertEqual(out["density_class"], "Moderate")
        self.assertEqual(out["density_score"], 1)

    def test_09_ten_vehicles_high_density(self) -> None:
        sp = _import_safety_module()
        dets = [_det(4, "car", 600, 100, 700, 300) for _ in range(10)]
        out = sp.analyze_frame(dets, (1000, 1000), ZoneAnalyzer(TEST_ZONES_CFG))
        self.assertEqual(out["density_class"], "High")
        self.assertEqual(out["density_score"], 2)

    def test_10_vru_share_below_0_5(self) -> None:
        sp = _import_safety_module()
        dets = [_det(0, "person", 100, 600, 200, 800), _det(4, "car", 600, 100, 700, 300), _det(5, "bus", 600, 100, 700, 400)]
        out = sp.analyze_frame(dets, (1000, 1000), ZoneAnalyzer(TEST_ZONES_CFG))
        self.assertAlmostEqual(out["vru_share"], 1 / 3, places=4)
        self.assertEqual(out["vru_score"], 0)

    def test_11_vru_share_equal_0_5(self) -> None:
        sp = _import_safety_module()
        dets = [_det(2, "motorcycle", 500, 100, 600, 300), _det(4, "car", 600, 100, 700, 300)]
        out = sp.analyze_frame(dets, (1000, 1000), ZoneAnalyzer(TEST_ZONES_CFG))
        self.assertAlmostEqual(out["vru_share"], 0.5, places=4)
        self.assertEqual(out["vru_score"], 1)

    def test_12_vru_share_above_0_5(self) -> None:
        sp = _import_safety_module()
        dets = [
            _det(0, "person", 100, 600, 200, 800),
            _det(1, "bicycle", 150, 600, 250, 800),
            _det(2, "motorcycle", 500, 100, 600, 300),
            _det(4, "car", 600, 100, 700, 300),
        ]
        out = sp.analyze_frame(dets, (1000, 1000), ZoneAnalyzer(TEST_ZONES_CFG))
        self.assertAlmostEqual(out["vru_share"], 0.75, places=4)
        self.assertEqual(out["vru_score"], 1)

    def test_13_all_safety_pulse_scores_0_to_5(self) -> None:
        sp = _import_safety_module()
        za = ZoneAnalyzer(TEST_ZONES_CFG)
        cases = [
            (0, "Low", []),
            (1, "Low", [_det(0, "person", 100, 600, 200, 800)]),
            (2, "Medium", [_det(4, "car", 600, 100, 700, 300) for _ in range(10)]),
            (3, "Medium", [_det(0, "person", 500, 700, 560, 850), _det(4, "car", 650, 700, 850, 900)]),
            (4, "High", [_det(0, "person", 500, 700, 560, 850)] + [_det(4, "car", 650, 700, 850, 900) for _ in range(10)]),
            (5, "High", [_det(0, "person", 500, 700, 560, 850)] + [_det(2, "motorcycle", 500, 200, 560, 400) for _ in range(9)] + [_det(4, "car", 650, 700, 850, 900)]),
        ]
        for expected_score, expected_label, dets in cases:
            out = sp.analyze_frame(dets, (1000, 1000), za)
            self.assertEqual(out["safety_pulse"], expected_score)
            self.assertEqual(out["safety_pulse_label"], expected_label)

    def test_14_detection_on_zone_boundary(self) -> None:
        za = ZoneAnalyzer(TEST_ZONES_CFG)
        # Bottom-center exact (400, 650) sits on crossing/road/footpath vertex
        dets = [_det(0, "person", 350, 500, 450, 650)]
        zc = za.analyze(dets, (1000, 1000))
        total_assigned = sum(v["total"] for v in zc.values())
        self.assertEqual(total_assigned, 1)

    def test_15_detection_outside_every_zone(self) -> None:
        za = ZoneAnalyzer(TEST_ZONES_CFG)
        # Bottom-center (200, 200) is in top-left empty quadrant outside all 3 polygons
        dets = [_det(4, "car", 100, 50, 300, 200)]
        zc = za.analyze(dets, (1000, 1000))
        self.assertEqual(sum(v["total"] for v in zc.values()), 0)

    def test_16_auto_rickshaw_counting_and_vehicle_rules(self) -> None:
        sp = _import_safety_module()
        za = ZoneAnalyzer(TEST_ZONES_CFG)
        dets = [
            _det(0, "person", 500, 700, 560, 850),
            _det(3, "auto_rickshaw", 650, 700, 800, 900),
        ]
        out = sp.analyze_frame(dets, (1000, 1000), za)
        self.assertEqual(out["counts"]["auto_rickshaw"], 1)
        self.assertEqual(out["vehicles_crossing"], 1)
        self.assertEqual(out["co_presence"], "YES")

    def test_17_pedestrian_heatmap_3x3_accumulation(self) -> None:
        sp = _import_safety_module()
        acc = sp.PedestrianHeatmapAccumulator()
        # Frame (300x300): grid cells are 100x100 each. Add 2 pedestrians in top-left (50,50) and 1 in bottom-right (250,250)
        acc.update([_det(0, "person", 20, 10, 80, 50), _det(0, "person", 30, 10, 70, 50), _det(0, "person", 220, 180, 280, 250)], (300, 300))
        self.assertEqual(int(acc.grid[0, 0]), 2)
        self.assertEqual(int(acc.grid[2, 2]), 1)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "heatmap.png"
            acc.save_heatmap(out_path)
            self.assertTrue(out_path.exists())

    def test_18_person_blurring_privacy(self) -> None:
        counter_mod = importlib.import_module("counter")
        self.assertTrue(hasattr(counter_mod, "blur_persons"), "counter.blur_persons missing")
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        # Checkerboard inside person bbox & vehicle bbox
        img[20:120, 20:120:2] = 255
        img[20:120, 180:280:2] = 255
        dets = [_det(0, "person", 20, 20, 120, 120), _det(4, "car", 180, 20, 280, 120)]
        blurred = counter_mod.blur_persons(img, dets)
        # Person ROI must change significantly (high variance smoothed), vehicle ROI untouched
        self.assertGreater(float(np.abs(blurred[30:110, 30:110].astype(float) - img[30:110, 30:110].astype(float)).mean()), 5.0)
        self.assertTrue(np.array_equal(blurred[20:120, 180:280], img[20:120, 180:280]))


class TestCsvContractAndRealModelIntegration(unittest.TestCase):
    """Audit documented CSV schema, Safety Brief, and real .pte image/video execution."""

    REQUIRED_CSV_COLS = {
        "frame_id",
        "person",
        "bicycle",
        "motorcycle",
        "car",
        "bus",
        "truck",
        "persons_footpath",
        "persons_road",
        "persons_crossing",
        "vehicles_road",
        "vehicles_crossing",
        "density_class",
        "co_presence",
        "vru_share",
        "safety_pulse",
        "inference_ms",
        "total_ms",
    }

    def test_19_csv_schema_contains_documented_columns_and_auto_rickshaw(self) -> None:
        counter_mod = importlib.import_module("counter")
        cols = set(counter_mod.CSV_COLUMNS)
        missing = self.REQUIRED_CSV_COLS - cols
        self.assertFalse(missing, f"Missing CSV columns: {missing}")
        self.assertIn("auto_rickshaw", cols)

    def test_20_real_pte_image_and_video_end_to_end(self) -> None:
        if not MODEL_PATH.exists() or not SAMPLE_IMAGE.exists() or not SAMPLE_IMAGE_2.exists():
            self.skipTest("Model or sample image/video not present")
        infer_mod = importlib.import_module("infer")
        sp = _import_safety_module()
        class_names, prep_cfg, zone_analyzer = infer_mod.load_configs(CONFIG_DIR)
        safety_cfg = sp.load_safety_config(CONFIG_DIR / "safety_config.json")
        detector = infer_mod.EdgeDetector(MODEL_PATH, class_names, prep_cfg)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            # Constant gray/white blank frames -> 0 detections cleanly, no crash
            gray_blank = np.full((480, 640, 3), 120, dtype=np.uint8)
            dets_blank, blank_ms = detector.predict_frame(gray_blank)
            self.assertEqual(len(dets_blank), 0)
            self.assertGreater(blank_ms, 0.0)

            # Image inference end-to-end (real .pte)
            img_csv = tmp_dir / "csv" / "sample_counts.csv"
            ann_dir = tmp_dir / "annotated"
            res_img = infer_mod.process_images(
                detector,
                [SAMPLE_IMAGE],
                ann_dir,
                img_csv,
                zone_analyzer,
                None,
                None,
                safety_cfg=safety_cfg,
                blur_privacy=True,
                out_root=tmp_dir,
            )
            self.assertEqual(res_img["frames_processed"], 1)
            self.assertTrue(img_csv.exists())
            self.assertTrue(img_csv.with_suffix(".zones.json").exists())
            self.assertTrue(Path(res_img["heatmap_path"]).exists())
            self.assertTrue(Path(res_img["safety_brief_md"]).exists())
            self.assertTrue(Path(res_img["safety_brief_json"]).exists())

            with open(img_csv, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            row0 = rows[0]
            self.assertTrue(self.REQUIRED_CSV_COLS.issubset(row0.keys()), f"CSV keys: {list(row0.keys())}")
            self.assertIn("auto_rickshaw", row0)
            self.assertGreaterEqual(float(row0["total_ms"]), float(row0["inference_ms"]))
            # Person blurring actually altered saved annotated pixels inside person box
            orig = cv2.imread(str(SAMPLE_IMAGE))
            saved = cv2.imread(str(ann_dir / SAMPLE_IMAGE.name))
            self.assertIsNotNone(saved)
            self.assertFalse(np.array_equal(orig, saved))

            # Video inference end-to-end (5 frames from real Indian road clip)
            vid_csv = tmp_dir / "csv" / "vid_counts.csv"
            tmp_video = _write_temp_mp4(tmp_dir / "clip.mp4", [SAMPLE_IMAGE, SAMPLE_IMAGE_2], num_frames=12)
            res_vid = infer_mod.process_video(
                detector,
                tmp_video,
                tmp_dir / "annotated" / "clip.mp4",
                vid_csv,
                zone_analyzer,
                None,
                None,
                max_frames=5,
                safety_cfg=safety_cfg,
                blur_privacy=True,
                out_root=tmp_dir,
            )
            self.assertEqual(res_vid["frames_processed"], 5)
            self.assertTrue(vid_csv.exists())
            self.assertTrue(vid_csv.with_suffix(".zones.json").exists())
            self.assertTrue(Path(res_vid["heatmap_path"]).exists())
            brief_data = json.loads(Path(res_vid["safety_brief_json"]).read_text(encoding="utf-8"))
            self.assertEqual(brief_data["total_frames_processed"], 5)
            self.assertIn("NOT a validated accident-prediction", brief_data["indicator_disclaimer"])

    def test_21_json_zone_reconfiguration_without_code_change(self) -> None:
        sp = _import_safety_module()
        # Reconfigure crossing to occupy only top half [0,0.5] and verify bottom-left moves to road
        custom_cfg = {
            "anchor_mode": "bottom_center",
            "zones": [
                {"name": "crossing", "polygon_norm": [[0.0, 0.0], [1.0, 0.0], [1.0, 0.4], [0.0, 0.4]]},
                {"name": "road", "polygon_norm": [[0.0, 0.4], [1.0, 0.4], [1.0, 1.0], [0.0, 1.0]]},
                {"name": "footpath", "polygon_norm": [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]]},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / "zones.json"
            zpath.write_text(json.dumps(custom_cfg), encoding="utf-8")
            za = ZoneAnalyzer.from_json(zpath)
        dets = [_det(0, "person", 450, 600, 550, 800), _det(3, "auto_rickshaw", 400, 600, 600, 800)]
        out = sp.analyze_frame(dets, (1000, 1000), za)
        self.assertEqual(out["persons_road"], 1)
        self.assertEqual(out["vehicles_road"], 1)
        self.assertEqual(out["co_presence"], "NO")


if __name__ == "__main__":
    unittest.main(verbosity=2)
