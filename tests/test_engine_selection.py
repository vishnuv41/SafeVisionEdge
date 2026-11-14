"""Tests for Engine Selection: strictly cpu, npu, compare (no auto mode, no silent fallback)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from engines.base_engine import EngineMode
from engines.cpu_engine import CPUInferenceEngine
from engines.engine_manager import EngineManager, InvalidEngineModeError
from engines.hailo_engine import HailoInferenceEngine
from infer import load_configs


class TestEngineSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.class_names, self.prep_cfg, _ = load_configs(ROOT_DIR / "config")
        self.cpu = CPUInferenceEngine(
            model_path=ROOT_DIR / "models" / "custom_model.pte",
            class_names=self.class_names,
            prep_cfg=self.prep_cfg,
        )
        self.npu = HailoInferenceEngine(
            hef_path=ROOT_DIR / "models" / "npu" / "custom_model.hef",
            class_names=self.class_names,
            prep_cfg=self.prep_cfg,
        )
        self.mgr = EngineManager(cpu_engine=self.cpu, npu_engine=self.npu)

    def test_01_allowed_modes_strictly_three(self) -> None:
        info = self.mgr.get_engines()
        mode_ids = [m["id"] for m in info["allowed_modes"]]
        self.assertEqual(mode_ids, [EngineMode.CPU, EngineMode.NPU, EngineMode.COMPARE])
        self.assertNotIn("auto", mode_ids)

    def test_02_auto_mode_rejected(self) -> None:
        with self.assertRaises(InvalidEngineModeError):
            self.mgr.select_mode("auto")

    def test_03_no_silent_fallback_on_unavailable_npu(self) -> None:
        self.mgr.select_mode(EngineMode.NPU)
        self.assertEqual(self.mgr.selected_mode, EngineMode.NPU)
        # require_ready_for_mode must raise RuntimeError, never silently switch to CPU
        if not self.npu.is_available():
            with self.assertRaises(RuntimeError) as ctx:
                self.mgr.require_ready_for_mode(EngineMode.NPU)
            err_msg = str(ctx.exception).upper()
            self.assertTrue(
                any(x in err_msg for x in ["HAILO", "DEVICE_NOT_FOUND", "MODEL_NOT_FOUND", "NPU"]),
                f"Unexpected error message: {err_msg}",
            )


if __name__ == "__main__":
    unittest.main()
