"""Standalone deployment package verification suite for SafeVision_PI_Deployment."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

PKG_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PKG_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import api  # noqa: E402

EXPECTED_SHA256 = "E9C932AA70EDBE3D8065612B576AAA25B4CD86840A3017864461A436D5959B56"


class TestPiDeploymentPackage(unittest.TestCase):
    def test_01_model_file_and_sha256_integrity(self) -> None:
        model_path = PKG_ROOT / "models" / "custom_model.pte"
        self.assertTrue(model_path.is_file(), f"Missing model: {model_path}")
        self.assertFalse(
            (PKG_ROOT / "models" / "custom_model_fp32.pte").exists(),
            "Redundant FP32 sibling model should be excluded from Pi deployment package",
        )
        h = hashlib.sha256()
        with model_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest().upper()
        self.assertEqual(digest, EXPECTED_SHA256)

        prov = json.loads((PKG_ROOT / "models" / "MODEL_PROVENANCE.json").read_text(encoding="utf-8"))
        self.assertEqual(prov["sha256"], EXPECTED_SHA256)
        self.assertEqual(prov["deployment_model_file"], "custom_model.pte")

    def test_02_no_external_paths_windows_paths_or_cloud_urls(self) -> None:
        forbidden_paths = re.compile(
            r"(REPO_ROOT|parents\[2\]|SafeVision-Edge|C:\\\\|E:\\\\|C:/|E:/|fonts\.googleapis|fonts\.gstatic)"
        )
        forbidden_cloud = re.compile(r"\b(ultralytics|torch\.hub|huggingface_hub)\b")
        runtime_files = list((PKG_ROOT / "src").glob("*.py")) + list((PKG_ROOT / "scripts").glob("*.py"))
        all_py_files = runtime_files + [p for p in (PKG_ROOT / "tests").glob("*.py") if p.name != Path(__file__).name]
        for py_file in all_py_files:
            text = py_file.read_text(encoding="utf-8")
            m = forbidden_paths.search(text)
            self.assertIsNone(m, f"Forbidden path/URL {m.group(0) if m else ''} found in {py_file}")
        for py_file in runtime_files:
            text = py_file.read_text(encoding="utf-8")
            m = forbidden_cloud.search(text)
            self.assertIsNone(m, f"Forbidden dependency {m.group(0) if m else ''} found in {py_file}")

        index_html = (PKG_ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("http://", index_html)
        self.assertNotIn("https://", index_html)
        self.assertNotIn("fonts.googleapis", index_html)

    def test_03_shell_scripts_use_unix_lf_line_endings(self) -> None:
        sh_files = list((PKG_ROOT / "scripts").glob("*.sh"))
        self.assertGreaterEqual(len(sh_files), 4)
        for sh in sh_files:
            raw = sh.read_bytes()
            self.assertNotIn(b"\r\n", raw, f"Windows CRLF found in {sh.name}")
            self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"), f"Invalid shebang in {sh.name}")

    def test_04_offline_spa_and_bundled_fonts_served_by_fastapi(self) -> None:
        assets_dir = PKG_ROOT / "static" / "assets"
        self.assertTrue(assets_dir.is_dir())
        woff2_files = list(assets_dir.glob("inter-*.woff2"))
        self.assertGreater(len(woff2_files), 0, "Bundled offline Inter fonts missing from static/assets/")

        with TestClient(api.app) as client:
            home = client.get("/")
            self.assertEqual(home.status_code, 200)
            self.assertIn("SafeVision Edge", home.text)

            asset_refs = re.findall(r'(?:src|href)="(/assets/[^"]+)"', home.text)
            self.assertGreaterEqual(len(asset_refs), 2, f"Expected JS/CSS assets in index.html, got {asset_refs}")
            for ref in asset_refs:
                res = client.get(ref)
                self.assertEqual(res.status_code, 200, f"Asset {ref} returned {res.status_code}")

    def test_05_minimal_clean_package_footprint(self) -> None:
        self.assertFalse((PKG_ROOT / ".venv").exists())
        self.assertFalse((PKG_ROOT / "frontend").exists())
        self.assertFalse((PKG_ROOT / "outputs" / "annotated").exists())

    def test_06_requirements_pinned_and_xnnpack_active(self) -> None:
        req_lines = [
            ln.strip()
            for ln in (PKG_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#") and not ln.strip().startswith("--")
        ]
        self.assertGreaterEqual(len(req_lines), 9)
        for line in req_lines:
            self.assertTrue(
                any(op in line for op in ("==", ">=")),
                f"Dependency does not specify version requirement: {line}",
            )
        req_text = "\n".join(req_lines)
        for expected in (
            "executorch",
            "numpy",
            "opencv-python-headless",
            "pydantic",
            "torch",
            "fastapi",
        ):
            self.assertIn(expected, req_text)

        # Ensure no CUDA or GPU packages in requirements
        self.assertNotIn("cuda", req_text.lower())
        self.assertNotIn("nvidia", req_text.lower())

        for asset in list((PKG_ROOT / "static" / "assets").glob("*.js")) + list(
            (PKG_ROOT / "static" / "assets").glob("*.css")
        ):
            txt = asset.read_text(encoding="utf-8", errors="ignore")
            self.assertNotIn("fonts.googleapis", txt)
            self.assertNotIn("fonts.gstatic", txt)
            self.assertNotIn("cdn.jsdelivr", txt)
            self.assertNotIn("unpkg.com", txt)

        from executorch.runtime import Runtime

        backends = list(Runtime.get().backend_registry.registered_backend_names)
        self.assertIn("XnnpackBackend", backends)


if __name__ == "__main__":
    unittest.main(verbosity=2)
