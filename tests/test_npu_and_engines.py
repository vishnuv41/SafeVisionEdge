"""Unit & API integration tests for CPU/NPU EngineManager and Hailo-8 NPU adapter.

NOTE: Because physical Hailo-8 hardware, HailoRT driver, and a compiled `.hef` model
are not present on this test host, hardware-positive Hailo tests use explicit mocks
and are clearly labeled `(MOCKED HARDWARE)`. Missing-hardware checks run against
real/unmocked state or mocked probe states.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

PKG_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PKG_ROOT / "src"
FIXTURES_DIR = PKG_ROOT / "tests" / "fixtures"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import api as api_module  # noqa: E402
from engines.base_engine import ALLOWED_ENGINE_MODES, EngineMode, detection_to_dict  # noqa: E402
from engines.cpu_engine import CPUInferenceEngine  # noqa: E402
from engines.engine_manager import EngineManager, InvalidEngineModeError  # noqa: E402
from engines.hailo_engine import HailoInferenceEngine  # noqa: E402
from infer import load_configs  # noqa: E402
from npu.hailo_device import HailoDeviceSession, HailoModelNotReadyError, HailoUnavailableError  # noqa: E402
from npu.hailo_inference import HailoInferenceRunner  # noqa: E402
from npu.hailo_status import HailoStatusCode, check_hailo_status  # noqa: E402
import time  # noqa: E402


class _FakeHailoSession(HailoDeviceSession):
    """Explicit mocked HailoDeviceSession returning synthetic outputs without hardware."""

    def __init__(self, hef_path: Path, output_factory) -> None:
        super().__init__(hef_path=hef_path)
        self._output_factory = output_factory
        self.closed = False
        self.load_calls = 0

    def load(self) -> None:
        self._loaded = True
        self._input_layout = "NHWC"
        self._input_dtype = "uint8"
        self.load_calls += 1

    def infer(self, preprocessed_input: np.ndarray):
        return self._output_factory(preprocessed_input)

    def close(self) -> None:
        self._loaded = False
        self.closed = True


class NpuAndEnginesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.class_names, cls.prep_cfg, _ = load_configs(PKG_ROOT / "config")
        cls.sample_bgr = cv2.imread(str(FIXTURES_DIR / "sample_road_1.jpg"))
        assert cls.sample_bgr is not None

    def test_01_allowed_engine_modes_strictly_three_no_auto(self) -> None:
        self.assertEqual(ALLOWED_ENGINE_MODES, ("cpu", "npu", "compare"))
        self.assertNotIn("auto", ALLOWED_ENGINE_MODES)

        cpu_eng = CPUInferenceEngine(PKG_ROOT / "models" / "custom_model.pte", self.class_names, self.prep_cfg)
        npu_eng = HailoInferenceEngine(PKG_ROOT / "models" / "npu" / "custom_model.hef", self.class_names, self.prep_cfg)
        mgr = EngineManager(cpu_eng, npu_eng, initial_mode="cpu")

        for mode in ("cpu", "npu", "compare"):
            st = mgr.select_mode(mode)
            self.assertEqual(st["selected_mode"], mode)

        for bad in ("auto", "AUTO", "hybrid", "", "gpu"):
            with self.assertRaises(InvalidEngineModeError):
                mgr.select_mode(bad)

    def test_02_real_host_hailo_status_not_fabricated(self) -> None:
        # On a host without Hailo-8 or .hef, must never falsely report READY.
        nonexistent_hef = PKG_ROOT / "models" / "npu" / "custom_model.hef"
        st = check_hailo_status(hef_path=nonexistent_hef)
        if not nonexistent_hef.exists():
            self.assertFalse(st["available"])
            self.assertNotEqual(st["status_code"], HailoStatusCode.READY)
            self.assertIn(
                st["status_code"],
                {
                    HailoStatusCode.NPU_HARDWARE_UNAVAILABLE,
                    HailoStatusCode.HAILORT_UNAVAILABLE,
                    HailoStatusCode.NPU_MODEL_NOT_READY,
                },
            )

    def test_03_mocked_hailo_status_state_transitions(self) -> None:
        class _FakeHailoPlatform:
            __version__ = "4.20.0"

        with tempfile.TemporaryDirectory() as tmp:
            hef = Path(tmp) / "custom_model.hef"
            hef.write_bytes(b"HEF_TEST_PLACEHOLDER")
            hef_missing = Path(tmp) / "missing.hef"

            st_no_hw = check_hailo_status(
                hef_path=hef,
                device_nodes_fn=lambda: [],
                driver_loaded_fn=lambda: False,
                import_hailort_fn=lambda: _FakeHailoPlatform(),
                run_cli_fn=lambda _args: (1, "no device"),
            )
            self.assertEqual(st_no_hw["status_code"], HailoStatusCode.NPU_HARDWARE_UNAVAILABLE)
            self.assertFalse(st_no_hw["available"])

            st_no_rt = check_hailo_status(
                hef_path=hef,
                device_nodes_fn=lambda: ["/dev/hailo0"],
                driver_loaded_fn=lambda: True,
                import_hailort_fn=lambda: None,
                run_cli_fn=lambda args: (0, "0000:01:00.0") if args == ["scan"] else (0, "Device Architecture: HAILO8"),
            )
            self.assertEqual(st_no_rt["status_code"], HailoStatusCode.HAILORT_UNAVAILABLE)
            self.assertFalse(st_no_rt["available"])

            st_no_hef = check_hailo_status(
                hef_path=hef_missing,
                device_nodes_fn=lambda: ["/dev/hailo0"],
                driver_loaded_fn=lambda: True,
                import_hailort_fn=lambda: _FakeHailoPlatform(),
                run_cli_fn=lambda args: (0, "0000:01:00.0") if args == ["scan"] else (0, "Device Architecture: HAILO8"),
            )
            self.assertEqual(st_no_hef["status_code"], HailoStatusCode.NPU_MODEL_NOT_READY)
            self.assertFalse(st_no_hef["available"])

            st_ready = check_hailo_status(
                hef_path=hef,
                device_nodes_fn=lambda: ["/dev/hailo0"],
                driver_loaded_fn=lambda: True,
                import_hailort_fn=lambda: _FakeHailoPlatform(),
                run_cli_fn=lambda args: (0, "0000:01:00.0") if args == ["scan"] else (0, "Device Architecture: HAILO8\nFirmware Version: 4.20.0"),
            )
            self.assertEqual(st_ready["status_code"], HailoStatusCode.READY)
            self.assertTrue(st_ready["available"])
            self.assertTrue(st_ready["firmware_available"])

    def test_04_mocked_hailo_runner_decodes_nms_and_raw_tensors(self) -> None:
        """(MOCKED HARDWARE) Verify HailoNMS & raw [1,11,8400] decode + resource cleanup."""
        with tempfile.TemporaryDirectory() as tmp:
            hef = Path(tmp) / "fake.hef"
            hef.write_bytes(b"FAKEHEF")

            # Case A: HailoNMSByClass output format (list of 7 arrays with [ymin, xmin, ymax, xmax, score])
            def nms_factory(_inp):
                per_cls = [np.zeros((0, 5), dtype=np.float32) for _ in range(7)]
                # Class 0 = person in normalized [0..1] wrt 640x640 letterbox
                per_cls[0] = np.array([[0.25, 0.25, 0.75, 0.50, 0.91]], dtype=np.float32)
                return [per_cls]

            sess_nms = _FakeHailoSession(hef, nms_factory)
            runner_nms = HailoInferenceRunner(hef, self.class_names, self.prep_cfg, session=sess_nms)
            dets, inf_ms, timings = runner_nms.predict_frame(self.sample_bgr, conf_thres=0.30)
            self.assertEqual(len(dets), 1)
            self.assertEqual(dets[0].class_id, 0)
            self.assertEqual(dets[0].class_name, "person")
            self.assertAlmostEqual(dets[0].confidence, 0.91, places=2)
            self.assertGreaterEqual(inf_ms, 0.0)
            self.assertIn("total_ms", timings)
            norm = detection_to_dict(dets[0])
            self.assertEqual(set(norm.keys()), {"class_id", "class_name", "confidence", "bbox"})
            self.assertEqual(len(norm["bbox"]), 4)
            runner_nms.close()
            self.assertTrue(sess_nms.closed)

            # Case B: Raw YOLOv11 [1, 11, 8400] tensor output
            def raw_factory(_inp):
                out = np.zeros((1, 11, 8400), dtype=np.float32)
                # Put a car (class_id=4 => channel 4+4=8) at anchor 10
                out[0, 0, 10] = 320.0
                out[0, 1, 10] = 320.0
                out[0, 2, 10] = 120.0
                out[0, 3, 10] = 80.0
                out[0, 8, 10] = 0.88
                return {"yolov11/output": out}

            sess_raw = _FakeHailoSession(hef, raw_factory)
            runner_raw = HailoInferenceRunner(hef, self.class_names, self.prep_cfg, session=sess_raw)
            dets_raw, _, _ = runner_raw.predict_frame(self.sample_bgr, conf_thres=0.30)
            self.assertTrue(any(d.class_name == "car" for d in dets_raw))
            runner_raw.close()
            self.assertTrue(sess_raw.closed)

    def test_05_npu_unavailable_never_silently_falls_back_to_cpu(self) -> None:
        cpu_eng = CPUInferenceEngine(PKG_ROOT / "models" / "custom_model.pte", self.class_names, self.prep_cfg)
        npu_eng = HailoInferenceEngine(PKG_ROOT / "models" / "nonexistent_test.hef", self.class_names, self.prep_cfg)
        mgr = EngineManager(cpu_eng, npu_eng, initial_mode=EngineMode.NPU)

        st = mgr.get_status()
        self.assertFalse(st["selected_ready"])
        self.assertNotEqual(st["npu"]["status_code"], HailoStatusCode.READY)
        with self.assertRaises((HailoModelNotReadyError, HailoUnavailableError)):
            mgr.npu_engine.predict_frame(self.sample_bgr, conf_thres=0.25)

    def _poll_until_done(self, client: TestClient, inspection_id: str, timeout_s: float = 10.0) -> dict:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            st = client.get(f"/api/v1/inspections/{inspection_id}/status").json()
            if st.get("status") in {"completed", "failed"}:
                return st
            time.sleep(0.05)
        self.fail(f"Inspection {inspection_id} did not finish within {timeout_s}s")

    def test_06_api_engine_selection_and_mocked_compare_inspection(self) -> None:
        """Verify /api/v1/engines* rejects 'auto' and runs compare inspection with (MOCKED HARDWARE) NPU."""
        with TestClient(api_module.app) as client:
            r_list = client.get("/api/v1/engines")
            self.assertEqual(r_list.status_code, 200)
            mode_ids = [m["id"] for m in r_list.json()["allowed_modes"]]
            self.assertEqual(mode_ids, ["cpu", "npu", "compare"])

            r_bad = client.post("/api/v1/engines/select", json={"mode": "auto"})
            self.assertEqual(r_bad.status_code, 400)

            r_ok = client.post("/api/v1/engines/select", json={"mode": "cpu"})
            self.assertEqual(r_ok.status_code, 200)
            self.assertEqual(r_ok.json()["selected_mode"], "cpu")

            # Verify NPU mode explicitly fails inspection without falling back to CPU
            img_bytes = (FIXTURES_DIR / "sample_road_1.jpg").read_bytes()
            created_npu = client.post(
                "/api/v1/inspections",
                files={"files": ("sample_road_1.jpg", img_bytes, "image/jpeg")},
                data={"sample_count": "10", "confidence_threshold": "0.3", "engine_mode": "npu"},
            )
            self.assertEqual(created_npu.status_code, 200)
            npu_id = created_npu.json()["inspection_id"]
            try:
                proc = client.post(f"/api/v1/inspections/{npu_id}/process", json={"engine_mode": "npu"})
                self.assertEqual(proc.status_code, 200)
                st = self._poll_until_done(client, npu_id)
                self.assertEqual(st["status"], "failed")
                self.assertTrue(
                    any(
                        code in str(st["error"])
                        for code in (
                            "DEVICE_NOT_FOUND",
                            "RUNTIME_NOT_INSTALLED",
                            "MODEL_NOT_FOUND",
                            "MODEL_INCOMPATIBLE",
                            "NPU_HARDWARE_UNAVAILABLE",
                            "HAILORT_UNAVAILABLE",
                            "NPU_MODEL_NOT_READY",
                        )
                    ),
                    f"Unexpected error message: {st['error']}",
                )
            finally:
                shutil.rmtree(api_module.INSPECTIONS_DIR / npu_id, ignore_errors=True)

            # Now mock HailoInferenceEngine so Compare Both completes via API
            ctx = api_module.get_extended_context()
            real_npu = ctx["engine_manager"].npu_engine

            with tempfile.TemporaryDirectory() as tmp:
                hef = Path(tmp) / "mocked.hef"
                hef.write_bytes(b"MOCKED_HEF")

                def nms_factory(_inp):
                    per_cls = [np.zeros((0, 5), dtype=np.float32) for _ in range(7)]
                    per_cls[4] = np.array([[0.30, 0.20, 0.70, 0.60, 0.85]], dtype=np.float32)
                    return [per_cls]

                mock_sess = _FakeHailoSession(hef, nms_factory)
                mock_runner = HailoInferenceRunner(hef, self.class_names, self.prep_cfg, session=mock_sess)

                with (
                    patch.object(real_npu, "is_available", return_value=True),
                    patch.object(real_npu, "get_status", return_value={
                        "engine_id": "npu",
                        "engine_name": "Hailo-8 NPU",
                        "available": True,
                        "status_code": "READY",
                        "message": "Mocked Hailo-8 READY",
                        "model_name": "custom_model.hef (MOCKED)",
                        "runtime": "HailoRT (Mocked)",
                        "backend": "Hailo-8",
                    }),
                    patch.object(real_npu, "_ensure_runner", return_value=mock_runner),
                ):
                    created_cmp = client.post(
                        "/api/v1/inspections",
                        files={"files": ("sample_road_1.jpg", img_bytes, "image/jpeg")},
                        data={
                            "sample_count": "10",
                            "confidence_threshold": "0.3",
                            "engine_mode": "compare",
                            "generate_safety_brief": "false",
                        },
                    )
                    self.assertEqual(created_cmp.status_code, 200)
                    cmp_id = created_cmp.json()["inspection_id"]
                    try:
                        proc_cmp = client.post(
                            f"/api/v1/inspections/{cmp_id}/process",
                            json={"engine_mode": "compare", "generate_safety_brief": False},
                        )
                        self.assertEqual(proc_cmp.status_code, 200)
                        st_cmp = self._poll_until_done(client, cmp_id)
                        self.assertEqual(st_cmp["status"], "completed")

                        res_cmp = client.get(f"/api/v1/inspections/{cmp_id}/results").json()
                        self.assertEqual(res_cmp["engine_mode"], "compare")
                        self.assertIn("comparison", res_cmp)
                        self.assertIn("npu_comparison", res_cmp["frames"][0])
                    finally:
                        shutil.rmtree(api_module.INSPECTIONS_DIR / cmp_id, ignore_errors=True)
                        # Reset selected mode back to cpu cleanly
                        client.post("/api/v1/engines/select", json={"mode": "cpu"})


if __name__ == "__main__":
    unittest.main()
