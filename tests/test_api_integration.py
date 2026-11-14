"""FastAPI backend integration tests against real ExecuTorch .pte model and safety pipeline."""

from __future__ import annotations

import csv
import io
import json
import shutil
import sys
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

PKG_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PKG_ROOT / "src"
import tempfile
import cv2

FIXTURES_DIR = PKG_ROOT / "tests" / "fixtures"
SAMPLE_IMG_1 = FIXTURES_DIR / "sample_road_1.jpg"
SAMPLE_IMG_2 = FIXTURES_DIR / "sample_road_2.jpg"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import api  # noqa: E402


class TestFastApiIntegration(unittest.TestCase):
    created_ids: list[str] = []

    @classmethod
    def setUpClass(cls) -> None:
        cls.client_ctx = TestClient(api.app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_ctx.__exit__(None, None, None)
        for iid in cls.created_ids:
            shutil.rmtree(api.INSPECTIONS_DIR / iid, ignore_errors=True)
            api._JOBS.pop(iid, None)

    def _poll_until_done(self, iid: str, timeout_s: float = 30.0) -> dict:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            r = self.client.get(f"/api/v1/inspections/{iid}/status")
            self.assertEqual(r.status_code, 200)
            st = r.json()
            if st["status"] in {"completed", "failed"}:
                return st
            time.sleep(0.15)
        self.fail(f"Inspection {iid} did not finish within {timeout_s}s")

    def test_01_health_and_configs(self) -> None:
        r = self.client.get("/api/v1/system/health")
        self.assertEqual(r.status_code, 200)
        h = r.json()
        self.assertEqual(h["status"], "online")
        self.assertEqual(h["mode"], "offline")
        self.assertTrue(h["model"]["loaded"])
        self.assertEqual(len(h["model"]["classes"]), 7)
        self.assertIn("auto_rickshaw", h["model"]["classes"])
        self.assertEqual(h["runtime"], {"name": "ExecuTorch", "backend": "XNNPACK"})

        from zone_analysis import infer_zone_type_from_def
        rz = self.client.get("/api/v1/config/zones")
        self.assertEqual(rz.status_code, 200)
        orig_zones = rz.json()
        types = {infer_zone_type_from_def(z) for z in orig_zones["zones"]}
        self.assertEqual(types, {"crossing", "road", "footpath"})
        self.assertGreaterEqual(len(orig_zones["zones"]), 3)

        # Invalid zone edit rejected (out of bounds > 1.0)
        bad_payload = {"anchor_mode": "bottom_center", "zones": [{"name": "road", "polygon_norm": [[0, 0], [1.5, 0], [1, 1]]}]}
        self.assertEqual(self.client.put("/api/v1/config/zones", json=bad_payload).status_code, 400)

        # Valid round-trip PUT & restore
        put_ok = self.client.put("/api/v1/config/zones", json=orig_zones)
        self.assertEqual(put_ok.status_code, 200)
        self.assertEqual(len(put_ok.json()["zones"]), len(orig_zones["zones"]))

        self.assertEqual(self.client.get("/api/v1/config/safety").status_code, 200)
        self.assertEqual(self.client.get("/api/v1/system/benchmark").status_code, 200)

    def test_02_upload_validation(self) -> None:
        bad = self.client.post(
            "/api/v1/inspections",
            files=[("files", ("notes.txt", b"hello", "text/plain"))],
        )
        self.assertEqual(bad.status_code, 400)
        empty = self.client.post(
            "/api/v1/inspections",
            files=[("files", ("empty.jpg", b"", "image/jpeg"))],
        )
        self.assertEqual(empty.status_code, 400)

    def test_03_single_and_multi_image_real_pte_workflow(self) -> None:
        if not SAMPLE_IMG_1.is_file() or not SAMPLE_IMG_2.is_file():
            self.skipTest("Sample images missing")
        files = [
            ("files", (SAMPLE_IMG_1.name, SAMPLE_IMG_1.read_bytes(), "image/jpeg")),
            ("files", (SAMPLE_IMG_2.name, SAMPLE_IMG_2.read_bytes(), "image/jpeg")),
        ]
        up = self.client.post(
            "/api/v1/inspections",
            files=files,
            data={
                "sample_count": "15",
                "confidence_threshold": "0.25",
                "enable_zones": "true",
                "enable_heatmap": "true",
                "enable_person_blurring": "true",
                "generate_safety_brief": "true",
            },
        )
        self.assertEqual(up.status_code, 200)
        iid = up.json()["inspection_id"]
        self.created_ids.append(iid)
        self.assertEqual(up.json()["total_frames"], 2)

        p1 = self.client.post(f"/api/v1/inspections/{iid}/process", json={})
        self.assertEqual(p1.status_code, 200)
        # Duplicate click guard
        p2 = self.client.post(f"/api/v1/inspections/{iid}/process", json={})
        self.assertEqual(p2.status_code, 200)
        self.assertTrue(p2.json()["already_started"])

        st = self._poll_until_done(iid)
        self.assertEqual(st["status"], "completed", st)
        self.assertEqual(st["processed_frames"], 2)

        res = self.client.get(f"/api/v1/inspections/{iid}/results")
        self.assertEqual(res.status_code, 200)
        rj = res.json()
        self.assertEqual(rj["total_frames"], 2)
        self.assertTrue(rj["options"]["blur_applied"])
        self.assertEqual(len(rj["frames"]), 2)
        self.assertGreater(rj["class_counts"]["auto_rickshaw"], 0)
        self.assertIn("Density ", rj["frames"][0]["safety_pulse"]["explanation"])

        fr = self.client.get(f"/api/v1/inspections/{iid}/frames/1")
        self.assertEqual(fr.status_code, 200)
        self.assertEqual(fr.headers["content-type"], "image/jpeg")

        hm = self.client.get(f"/api/v1/inspections/{iid}/heatmap")
        self.assertEqual(hm.status_code, 200)
        self.assertTrue(hm.json()["available"])
        self.assertEqual(len(hm.json()["cells"]), 9)
        hm_img = self.client.get(f"/api/v1/inspections/{iid}/heatmap/image")
        self.assertEqual(hm_img.status_code, 200)
        self.assertEqual(hm_img.headers["content-type"], "image/png")

        rep = self.client.get(f"/api/v1/inspections/{iid}/report")
        self.assertEqual(rep.status_code, 200)
        self.assertTrue(rep.json()["available"])
        self.assertIn("NOT a validated accident-prediction", rep.json()["indicator_disclaimer"])

        bm = self.client.get(f"/api/v1/inspections/{iid}/benchmark")
        self.assertEqual(bm.status_code, 200)
        self.assertGreater(bm.json()["mean_inference_ms"], 0.0)
        self.assertIsNone(bm.json()["cpu_temperature_c"])

        csv_r = self.client.get(f"/api/v1/inspections/{iid}/export/csv")
        self.assertEqual(csv_r.status_code, 200)
        rows = list(csv.DictReader(io.StringIO(csv_r.text)))
        self.assertEqual(len(rows), 2)
        self.assertIn("auto_rickshaw", rows[0])
        self.assertIn("safety_pulse", rows[0])

    def test_04_video_real_pte_workflow(self) -> None:
        if not SAMPLE_IMG_1.is_file() or not SAMPLE_IMG_2.is_file():
            self.skipTest("Sample road fixtures missing")
        im1 = cv2.imread(str(SAMPLE_IMG_1))
        im2 = cv2.imread(str(SAMPLE_IMG_2))
        h, w = im1.shape[:2]
        im2_r = cv2.resize(im2, (w, h)) if im2.shape[:2] != (h, w) else im2
        with tempfile.TemporaryDirectory() as tmp:
            vid_path = Path(tmp) / "test_junction.mp4"
            writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
            for i in range(15):
                writer.write(im1 if i % 2 == 0 else im2_r)
            writer.release()
            vid_bytes = vid_path.read_bytes()
        up = self.client.post(
            "/api/v1/inspections",
            files=[("files", ("test_junction.mp4", vid_bytes, "video/mp4"))],
            data={"sample_count": "10", "confidence_threshold": "0.25"},
        )
        self.assertEqual(up.status_code, 200)
        iid = up.json()["inspection_id"]
        self.created_ids.append(iid)
        self.assertEqual(up.json()["input_type"], "video")
        self.assertEqual(up.json()["total_frames"], 10)

        self.assertEqual(self.client.post(f"/api/v1/inspections/{iid}/process").status_code, 200)
        st = self._poll_until_done(iid, timeout_s=30.0)
        self.assertEqual(st["status"], "completed", st)

        res = self.client.get(f"/api/v1/inspections/{iid}/results").json()
        self.assertEqual(res["total_frames"], 10)
        self.assertEqual(len(res["frames"]), 10)
        self.assertGreater(res["frames"][-1]["timestamp_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
