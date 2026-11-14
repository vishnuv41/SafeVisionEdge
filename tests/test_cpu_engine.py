"""Tests for CPU Inference Engine (ExecuTorch INT8 YOLOv11-S)."""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
from engines.cpu_engine import CPUInferenceEngine
from infer import load_configs

EXPECTED_SHA256 = "e9c932aa70edbe3d8065612b576aaa25b4cd86840a3017864461a436d5959b56"


class TestCPUEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.pte_path = ROOT_DIR / "models" / "custom_model.pte"
        self.class_names, self.prep_cfg, _ = load_configs(ROOT_DIR / "config")
        self.engine = CPUInferenceEngine(
            model_path=self.pte_path,
            class_names=self.class_names,
            prep_cfg=self.prep_cfg,
        )

    def test_01_cpu_model_sha256_unmodified(self) -> None:
        self.assertTrue(self.pte_path.is_file(), f"Model not found: {self.pte_path}")
        h = hashlib.sha256(self.pte_path.read_bytes()).hexdigest()
        self.assertEqual(h.lower(), EXPECTED_SHA256.lower(), "CPU model hash mismatch!")

    def test_02_cpu_engine_status_and_classes(self) -> None:
        st = self.engine.get_status()
        self.assertEqual(st["engine_id"], "cpu")
        self.assertTrue(st["available"])
        self.assertEqual(len(st["classes"]), 7)
        self.assertEqual(
            st["classes"],
            ["person", "bicycle", "motorcycle", "auto_rickshaw", "car", "bus", "truck"],
        )

    def test_03_cpu_smoke_inference_on_road_frame(self) -> None:
        img_path = ROOT_DIR / "tests" / "fixtures" / "sample_road_1.jpg"
        self.assertTrue(img_path.is_file())
        bgr = cv2.imread(str(img_path))
        self.assertIsNotNone(bgr)

        dets, inference_ms = self.engine.predict_frame(bgr, conf_thres=0.25)
        self.assertGreater(len(dets), 10, "Expected robust road-user detections on sample road image")
        self.assertIsInstance(inference_ms, float)
        self.assertGreater(inference_ms, 0.0)

        # Verify all detected classes are valid 7 SafeVision classes
        allowed = set(self.class_names.values())
        for d in dets:
            self.assertIn(d.class_name, allowed)
            self.assertGreaterEqual(d.confidence, 0.25)


if __name__ == "__main__":
    unittest.main()
