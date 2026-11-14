"""Tests for Hailo-8 NPU Inference Engine and Decoder."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from engines.hailo_engine import HailoInferenceEngine
from npu.hailo_decoder import decode_hailo_output
from npu.hailo_status import HailoStatusCode, check_hailo_status
from infer import load_configs


class TestNPUEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.class_names, self.prep_cfg, _ = load_configs(ROOT_DIR / "config")
        self.hef_path = ROOT_DIR / "models" / "npu" / "custom_model.hef"
        self.engine = HailoInferenceEngine(
            hef_path=self.hef_path,
            class_names=self.class_names,
            prep_cfg=self.prep_cfg,
        )

    def test_01_npu_status_reports_truthfully_on_host(self) -> None:
        st = self.engine.get_status()
        self.assertEqual(st["engine_id"], "npu")
        self.assertIn("status_code", st)
        # On Windows host without physical PCIe Hailo device, status must be DEVICE_NOT_FOUND or MODEL_NOT_FOUND
        self.assertIn(
            st["status_code"],
            [
                HailoStatusCode.DEVICE_NOT_FOUND,
                HailoStatusCode.RUNTIME_NOT_INSTALLED,
                HailoStatusCode.MODEL_NOT_FOUND,
                HailoStatusCode.MODEL_INCOMPATIBLE,
            ],
        )
        self.assertFalse(st["available"])

    def test_02_hailo_decoder_7_classes(self) -> None:
        # Mock Hailo on-chip NMS output for 7 classes
        per_cls = [np.zeros((0, 5), dtype=np.float32) for _ in range(7)]
        # Class 3: auto_rickshaw
        per_cls[3] = np.array([[0.1, 0.2, 0.5, 0.6, 0.88]], dtype=np.float32)

        dets = decode_hailo_output(
            outputs=per_cls,
            orig_shape=(1080, 1920),
            scale=1.0,
            pad=(0.0, 0.0),
            class_names=self.class_names,
            conf_thres=0.25,
        )
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].class_name, "auto_rickshaw")
        self.assertEqual(dets[0].class_id, 3)
        self.assertAlmostEqual(dets[0].confidence, 0.88, places=2)


if __name__ == "__main__":
    unittest.main()
