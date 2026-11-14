"""Unit & API integration tests for Comparison Benchmark, HardwareMonitor, and Interactive Zone Editor."""

from __future__ import annotations

import csv
import json
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
from benchmarks.comparison_runner import ComparisonRunner, compute_detection_agreement  # noqa: E402
from benchmarks.hardware_monitor import HardwareMonitor, _read_throttling_status  # noqa: E402
from benchmarks.performance_report import load_latest_comparison, save_comparison_reports  # noqa: E402
from decode import Detection  # noqa: E402
from engines.base_engine import BaseInferenceEngine, EngineMode, EngineResult  # noqa: E402
from engines.cpu_engine import CPUInferenceEngine  # noqa: E402
from infer import load_configs  # noqa: E402
from safety_pulse import analyze_frame  # noqa: E402
from zone_analysis import ZoneAnalyzer  # noqa: E402
from zone_store import ZoneStore  # noqa: E402


class _RecordingMockEngine(BaseInferenceEngine):
    """Test engine recording call order (warmup vs measured) for interleaved verification."""

    def __init__(self, engine_id: str, call_log: list[str], dets: list[Detection]) -> None:
        self.engine_id = engine_id
        self.display_name = f"Mock {engine_id.upper()}"
        self._log = call_log
        self._dets = dets

    def is_available(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def get_status(self):
        return {
            "engine_id": self.engine_id,
            "available": True,
            "status_code": "READY",
            "model_name": f"{self.engine_id}_mock",
            "runtime": "MockRT",
            "backend": "MockBackend",
        }

    def predict_frame(self, img_bgr: np.ndarray, conf_thres: float | None = None, iou_thres: float | None = None):
        self._log.append(self.engine_id)
        return list(self._dets), (4.0 if self.engine_id == "npu" else 28.0)

    def predict_frame_detailed(
        self, img_bgr: np.ndarray, conf_thres: float | None = None, iou_thres: float | None = None
    ) -> EngineResult:
        self._log.append(self.engine_id)
        inf = 4.0 if self.engine_id == "npu" else 28.0
        return EngineResult(
            engine_id=self.engine_id,
            engine_name=self.display_name,
            model_name=f"{self.engine_id}_mock",
            detections=list(self._dets),
            inference_ms=inf,
            preprocess_ms=1.0,
            postprocess_ms=1.0,
            total_ms=inf + 2.0,
        )


class ComparisonAndZonesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.class_names, cls.prep_cfg, _ = load_configs(PKG_ROOT / "config")
        cls.safety_cfg = json.loads((PKG_ROOT / "config" / "safety_config.json").read_text(encoding="utf-8"))
        cls.sample_bgr = cv2.imread(str(FIXTURES_DIR / "sample_road_1.jpg"))
        assert cls.sample_bgr is not None

    def test_01_hardware_monitor_unavailable_sensors_not_fabricated(self) -> None:
        # Mock missing vcgencmd and thermal/throttle sysfs so throttling is verified Unavailable (not 'No throttling')
        with (
            patch("benchmarks.hardware_monitor.shutil.which", return_value=None),
            patch("benchmarks.hardware_monitor.Path.is_file", return_value=False),
        ):
            th = _read_throttling_status()
            self.assertFalse(th["supported"])
            self.assertEqual(th["status"], "Unavailable")
            self.assertIsNone(th["throttled"])

        mon = HardwareMonitor(sample_interval_sec=0.05)
        mon.start()
        summary = mon.stop()
        self.assertGreaterEqual(summary["sample_count"], 1)
        self.assertGreaterEqual(summary["duration_sec"], 0.0)
        self.assertGreater(summary["process_rss_mb"]["mean"], 0.0)
        self.assertGreater(summary["system_ram_percent"]["mean"], 0.0)
        self.assertFalse(summary["power"]["available"])
        self.assertIsNone(summary["power"]["avg_watts"])

    def test_02_comparison_runner_interleaved_warmup_excluded_and_reports(self) -> None:
        """Verify paired interleaving, warmup exclusion, IoU agreement, and JSON/CSV report writing."""
        call_log: list[str] = []
        det_a = [
            Detection(100, 100, 200, 220, 0.90, 0, "person"),
            Detection(250, 150, 420, 320, 0.85, 4, "car"),
        ]
        det_b = [
            Detection(102, 102, 198, 218, 0.87, 0, "person"),
            Detection(250, 150, 420, 320, 0.80, 4, "car"),
        ]
        cpu_mock = _RecordingMockEngine("cpu", call_log, det_a)
        npu_mock = _RecordingMockEngine("npu", call_log, det_b)

        runner = ComparisonRunner(cpu_mock, npu_mock)
        report = runner.run(
            [self.sample_bgr],
            mode=EngineMode.COMPARE,
            warmup_runs=2,
            measured_runs=4,
            conf_thres=0.25,
            iou_thres=0.45,
        )

        # Total calls = (2 warmup + 4 measured) * 2 engines = 12 interleaved calls:
        # Warmup: (cpu, npu) x2; Measured alternates (cpu, npu) / (npu, cpu) to avoid thermal bias
        self.assertEqual(
            call_log,
            ["cpu", "npu", "cpu", "npu", "cpu", "npu", "npu", "cpu", "cpu", "npu", "npu", "cpu"],
        )

        self.assertEqual(report["cpu"]["processed_frames"], 4)
        self.assertEqual(report["npu"]["processed_frames"], 4)
        self.assertEqual(report["cpu"]["failed_frames"], 0)
        self.assertAlmostEqual(report["cpu"]["mean_inference_ms"], 28.0, places=1)
        self.assertAlmostEqual(report["npu"]["mean_inference_ms"], 4.0, places=1)
        self.assertGreater(report["accuracy_comparison"]["cross_engine_similarity"]["matched_detections"], 0)

        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            paths = save_comparison_reports(report, out_root)
            self.assertTrue(Path(paths["comparison_json"]).is_file())
            self.assertTrue(Path(paths["comparison_csv"]).is_file())
            self.assertTrue(Path(paths["cpu_json"]).is_file())
            self.assertTrue(Path(paths["npu_json"]).is_file())

            loaded = load_latest_comparison(out_root)
            self.assertTrue(loaded["available"])
            self.assertEqual(loaded["cpu"]["processed_frames"], 4)
            with open(paths["comparison_csv"], "r", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)
            self.assertEqual({r["engine_id"] for r in rows}, {"cpu", "npu"})

    def test_03_detection_agreement_with_ground_truth(self) -> None:
        pred = [[Detection(10, 10, 50, 50, 0.9, 0, "person")]]
        gt = [[Detection(10, 10, 50, 50, 1.0, 0, "person"), Detection(200, 200, 260, 260, 1.0, 4, "car")]]
        agr = compute_detection_agreement(pred, pred, ground_truth_frames=gt)
        self.assertTrue(agr["ground_truth_metrics"]["available"])
        self.assertAlmostEqual(agr["ground_truth_metrics"]["cpu"]["precision"], 1.0)
        self.assertAlmostEqual(agr["ground_truth_metrics"]["cpu"]["recall"], 0.5)
        self.assertEqual(agr["ground_truth_metrics"]["cpu"]["false_negatives"], 1)

    def test_04_zone_store_and_safety_pulse_custom_names(self) -> None:
        """Zones with custom names aggregate properly via zone['type'] in Safety Pulse."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dir = Path(tmp) / "config"
            cfg_dir.mkdir(parents=True)
            shutil.copy(PKG_ROOT / "config" / "zones.json", cfg_dir / "zones.json")
            previews_dir = Path(tmp) / "previews"

            store = ZoneStore(cfg_dir, previews_dir)
            scene_a = "scene_junction_alpha"
            scene_b = "scene_highway_beta"

            store.save_zones(
                [
                    {
                        "zone_id": "z_rd",
                        "name": "North Arterial",
                        "type": "road",
                        "points": [[0.1, 0.5], [0.9, 0.5], [0.9, 0.95], [0.1, 0.95]],
                    },
                    {
                        "zone_id": "z_fp",
                        "name": "East Sidewalk",
                        "type": "footpath",
                        "points": [[0.0, 0.0], [0.1, 0.0], [0.1, 1.0], [0.0, 1.0]],
                    },
                ],
                scene_id=scene_a,
            )
            store.save_zones(
                [
                    {
                        "zone_id": "z_cr",
                        "name": "Midblock Crosswalk",
                        "type": "crossing",
                        "points": [[0.2, 0.4], [0.8, 0.4], [0.8, 0.6], [0.2, 0.6]],
                    }
                ],
                scene_id=scene_b,
            )

            loaded_a = store.load_zones(scene_a)
            loaded_b = store.load_zones(scene_b)
            self.assertEqual(len(loaded_a["zones"]), 2)
            self.assertEqual(len(loaded_b["zones"]), 1)
            self.assertEqual(loaded_a["zones"][0]["name"], "North Arterial")
            self.assertEqual(loaded_b["zones"][0]["name"], "Midblock Crosswalk")

            # Test Safety Pulse aggregation by type='road' for custom-named zone
            h, w = 400, 600
            # Person standing near bottom-center inside North Arterial road polygon + car on North Arterial
            dets = [
                Detection(280, 200, 320, 320, 0.92, 0, "person"),
                Detection(120, 220, 240, 340, 0.89, 4, "car"),
            ]
            za = ZoneAnalyzer(loaded_a)
            sp = analyze_frame(dets, (h, w), zone_analyzer=za, cfg=self.safety_cfg)
            self.assertIn("North Arterial", sp["zone_counts"])
            self.assertEqual(sp["zone_counts"]["North Arterial"]["person"], 1)
            self.assertEqual(sp["zone_counts"]["North Arterial"]["car"], 1)
            self.assertEqual(sp["persons_road"], 1)
            self.assertGreaterEqual(sp["vehicles_road"], 1)

    def test_05_zone_editor_api_crud_video_preview_and_suggest(self) -> None:
        scene_id = "unit_test_scene_001"
        ctx = api_module.get_extended_context()
        zones_dir = ctx["zone_store"].scenes_dir
        previews_dir = ctx["zone_store"].previews_dir
        extracted_sid = None

        try:
            with TestClient(api_module.app) as client:
                # 1. Hardware & latest benchmark endpoints
                r_hw = client.get("/api/v1/hardware/status")
                self.assertEqual(r_hw.status_code, 200)
                hw = r_hw.json()
                self.assertIn("system_cpu_percent", hw)
                self.assertIn("throttling", hw)
                self.assertIn("npu", hw)

                # 2. Out-of-bounds polygon rejected with HTTP 400
                r_bad = client.post(
                    "/api/v1/zones",
                    json={
                        "scene_id": scene_id,
                        "name": "Bad Zone",
                        "type": "road",
                        "points": [[0.1, 0.2], [1.2, 0.3], [0.5, 0.9]],
                    },
                )
                self.assertEqual(r_bad.status_code, 400)

                # 3. Create 2 custom zones via POST /api/v1/zones
                r_create = client.post(
                    "/api/v1/zones",
                    json={
                        "scene_id": scene_id,
                        "zones": [
                            {
                                "zone_id": "zone_main_rd",
                                "name": "Main Junction Road",
                                "type": "road",
                                "points": [[0.15, 0.4], [0.85, 0.4], [0.90, 0.95], [0.10, 0.95]],
                            },
                            {
                                "zone_id": "zone_west_fp",
                                "name": "West Sidewalk",
                                "type": "footpath",
                                "points": [[0.0, 0.3], [0.14, 0.3], [0.14, 0.95], [0.0, 0.95]],
                            },
                        ],
                    },
                )
                self.assertEqual(r_create.status_code, 200)
                self.assertEqual(len(r_create.json()["zones"]), 2)

                # 4. Update zone name, type, and vertices via PUT /api/v1/zones/{zone_id}
                r_upd = client.put(
                    f"/api/v1/zones/zone_main_rd?scene_id={scene_id}",
                    json={
                        "name": "West Crosswalk",
                        "type": "crossing",
                        "points": [[0.2, 0.45], [0.8, 0.45], [0.8, 0.70], [0.2, 0.70]],
                    },
                )
                self.assertEqual(r_upd.status_code, 200)
                z_upd = r_upd.json()["config"]["zones"][0]
                self.assertEqual(z_upd["name"], "West Crosswalk")
                self.assertEqual(z_upd["type"], "crossing")

                # 5. Extract representative frame preview from synthetic MP4 video
                with tempfile.TemporaryDirectory() as tmp:
                    mp4_path = Path(tmp) / "test_clip.mp4"
                    vw = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
                    for f_i in range(6):
                        frame = np.full((120, 160, 3), fill_value=30 + f_i * 20, dtype=np.uint8)
                        cv2.rectangle(frame, (20, 40), (140, 100), (200, 180, 50), -1)
                        vw.write(frame)
                    vw.release()
                    mp4_bytes = mp4_path.read_bytes()

                r_prev = client.post(
                    "/api/v1/zones/preview",
                    files={"file": ("test_clip.mp4", mp4_bytes, "video/mp4")},
                )
                self.assertEqual(r_prev.status_code, 200)
                prev_data = r_prev.json()
                extracted_sid = prev_data["scene_id"]
                self.assertEqual(prev_data["width"], 160)
                self.assertEqual(prev_data["height"], 120)
                self.assertEqual(prev_data["media_type"], "video")

                r_img = client.get(prev_data["preview_url"])
                self.assertEqual(r_img.status_code, 200)
                self.assertIn("image/jpeg", r_img.headers["content-type"])

                # 6. Heuristic suggestion (non-destructive)
                r_sug = client.post(f"/api/v1/zones/suggest?scene_id={scene_id}")
                self.assertEqual(r_sug.status_code, 200)
                self.assertGreaterEqual(len(r_sug.json()["zones"]), 1)
                r_get = client.get(f"/api/v1/zones?scene_id={scene_id}")
                self.assertEqual(r_get.json()["zones"][0]["name"], "West Crosswalk")

                # 7. Delete one zone -> leaves 1; deleting last remaining returns 400
                r_del = client.delete(f"/api/v1/zones/zone_west_fp?scene_id={scene_id}")
                self.assertEqual(r_del.status_code, 200)
                self.assertEqual(len(r_del.json()["config"]["zones"]), 1)

                r_del_last = client.delete(f"/api/v1/zones/zone_main_rd?scene_id={scene_id}")
                self.assertEqual(r_del_last.status_code, 400)

                # 8. Verify bundled UI extensions are served offline
                self.assertEqual(client.get("/assets/sv_extensions.js").status_code, 200)
                self.assertEqual(client.get("/assets/sv_extensions.css").status_code, 200)
        finally:
            (zones_dir / f"{scene_id}.json").unlink(missing_ok=True)
            if extracted_sid:
                (previews_dir / f"{extracted_sid}.jpg").unlink(missing_ok=True)
                (previews_dir / f"{extracted_sid}.json").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
