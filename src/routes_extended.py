"""Extended FastAPI routes for Engine Selection, Hailo-8 NPU Comparison, Hardware Telemetry, and Interactive Zone Editor."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from benchmarks.comparison_runner import ComparisonRunner
from benchmarks.hardware_monitor import HardwareMonitor
from benchmarks.performance_report import load_latest_comparison, save_comparison_reports
from engines.base_engine import ALLOWED_ENGINE_MODES, EngineMode
from engines.engine_manager import EngineManager, InvalidEngineModeError
from infer import IMAGE_EXTS, VIDEO_EXTS
from zone_analysis import compute_homography_calibration
from zone_store import ZoneStore
from zone_suggest import estimate_scene_zones

router = APIRouter(prefix="/api/v1", tags=["extended"])

_BM_LOCK = threading.Lock()
_BM_JOBS: dict[str, dict[str, Any]] = {}


class EngineSelectRequest(BaseModel):
    mode: str


class BenchmarkCompareRequest(BaseModel):
    mode: str = EngineMode.COMPARE
    warmup_runs: int = Field(default=3, ge=0, le=20)
    measured_runs: int = Field(default=10, ge=1, le=200)
    conf_thres: float = Field(default=0.25, ge=0.05, le=0.95)
    iou_thres: float = Field(default=0.45, ge=0.10, le=0.95)
    inspection_id: str | None = None


class ZoneSavePayload(BaseModel):
    scene_id: str | None = None
    anchor_mode: str = "bottom_center"
    zones: list[dict[str, Any]] | None = None
    # Single-zone creation fields when `zones` is omitted
    zone_id: str | None = None
    name: str | None = None
    type: str | None = None
    color_bgr: list[int] | None = None
    points: list[list[float]] | None = None
    polygon_norm: list[list[float]] | None = None


class ZoneUpdatePayload(BaseModel):
    name: str | None = None
    type: str | None = None
    color_bgr: list[int] | None = None
    points: list[list[float]] | None = None
    polygon_norm: list[list[float]] | None = None


class ZoneCalibrationPayload(BaseModel):
    scene_id: str | None = None
    image_points_norm: list[list[float]]
    world_points_m: list[list[float]]
    frame_width: int = 1280
    frame_height: int = 720


def _get_ctx() -> dict[str, Any]:
    import api as api_mod

    return api_mod.get_extended_context()


@router.get("/engines")
def list_engines() -> dict[str, Any]:
    ctx = _get_ctx()
    manager: EngineManager = ctx["engine_manager"]
    return manager.get_engines()


@router.get("/engines/status")
def engines_status() -> dict[str, Any]:
    ctx = _get_ctx()
    manager: EngineManager = ctx["engine_manager"]
    return manager.get_status()


@router.post("/engines/select")
def select_engine(payload: EngineSelectRequest) -> dict[str, Any]:
    ctx = _get_ctx()
    manager: EngineManager = ctx["engine_manager"]
    try:
        status = manager.select_mode(payload.mode)
    except InvalidEngineModeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return status


@router.get("/hardware/status")
def hardware_status() -> dict[str, Any]:
    ctx = _get_ctx()
    monitor = HardwareMonitor(hef_path=ctx["hef_path"])
    return monitor.snapshot(include_npu=True)


def _gather_benchmark_frames(ctx: dict[str, Any], inspection_id: str | None) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    if inspection_id:
        jobs = ctx["jobs"]
        with ctx["jobs_lock"]:
            job = jobs.get(inspection_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Inspection {inspection_id} not found")
        for spath in job.get("input_paths", []):
            p = Path(spath)
            if p.suffix.lower() in VIDEO_EXTS:
                cap = cv2.VideoCapture(str(p))
                if cap.isOpened():
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                    indices = [int(i) for i in np.linspace(0, max(0, total - 1), min(5, max(1, total)), dtype=int)]
                    for idx in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ok, fr = cap.read()
                        if ok and fr is not None:
                            frames.append(fr)
                    cap.release()
            else:
                im = cv2.imread(str(p))
                if im is not None:
                    frames.append(im)

    if not frames:
        fixtures_dir = ctx["pkg_root"] / "tests" / "fixtures"
        for fname in ("sample_road_1.jpg", "sample_road_2.jpg"):
            fpath = fixtures_dir / fname
            if fpath.is_file():
                im = cv2.imread(str(fpath))
                if im is not None:
                    frames.append(im)

    if not frames:
        frames.append(np.full((720, 1280, 3), 120, dtype=np.uint8))
    return frames


@router.post("/benchmark/compare")
def start_benchmark_compare(body: BenchmarkCompareRequest | None = None) -> dict[str, Any]:
    req = body or BenchmarkCompareRequest()
    mode = str(req.mode).strip().lower()
    if mode not in ALLOWED_ENGINE_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid benchmark mode '{req.mode}'. Allowed: {list(ALLOWED_ENGINE_MODES)}",
        )

    ctx = _get_ctx()
    frames = _gather_benchmark_frames(ctx, req.inspection_id)
    job_id = f"BM-{uuid.uuid4().hex[:8].upper()}"

    job_record: dict[str, Any] = {
        "job_id": job_id,
        "status": "running",
        "progress": 0,
        "completed_runs": 0,
        "total_runs": req.measured_runs,
        "mode": mode,
        "error": None,
        "report": None,
        "saved_files": None,
    }
    with _BM_LOCK:
        _BM_JOBS[job_id] = job_record

    def _worker() -> None:
        try:
            manager: EngineManager = ctx["engine_manager"]
            runner = ComparisonRunner(
                cpu_engine=manager.cpu_engine,
                npu_engine=manager.npu_engine,
                hef_path=ctx["hef_path"],
            )

            def _on_prog(done: int, total: int, _stage: str) -> None:
                pct = int(round((done / max(1, total)) * 100))
                with _BM_LOCK:
                    _BM_JOBS[job_id].update(
                        {"completed_runs": done, "total_runs": total, "progress": pct}
                    )

            report = runner.run(
                frames_bgr=frames,
                mode=mode,
                warmup_runs=req.warmup_runs,
                measured_runs=req.measured_runs,
                conf_thres=req.conf_thres,
                iou_thres=req.iou_thres,
                progress_cb=_on_prog,
            )
            saved = save_comparison_reports(report, ctx["outputs_dir"])
            with _BM_LOCK:
                _BM_JOBS[job_id].update(
                    {
                        "status": "completed",
                        "progress": 100,
                        "completed_runs": req.measured_runs,
                        "report": report,
                        "saved_files": saved,
                    }
                )
        except Exception as exc:
            with _BM_LOCK:
                _BM_JOBS[job_id].update({"status": "failed", "error": str(exc)})

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return {
        "job_id": job_id,
        "status": "running",
        "mode": mode,
        "warmup_runs": req.warmup_runs,
        "measured_runs": req.measured_runs,
    }


@router.get("/benchmark/compare/{job_id}")
def get_benchmark_job(job_id: str) -> dict[str, Any]:
    with _BM_LOCK:
        job = _BM_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Benchmark job {job_id} not found")
        return dict(job)


@router.get("/benchmark/results")
def get_benchmark_results() -> dict[str, Any]:
    ctx = _get_ctx()
    outputs_dir: Path = ctx["outputs_dir"]
    cmp_data = load_latest_comparison(outputs_dir)
    cpu_p = outputs_dir / "benchmark" / "cpu" / "latest.json"
    npu_p = outputs_dir / "benchmark" / "npu" / "latest.json"
    csv_p = outputs_dir / "benchmark" / "comparison" / "latest.csv"
    return {
        **cmp_data,
        "cpu_report_available": cpu_p.is_file(),
        "npu_report_available": npu_p.is_file(),
        "csv_available": csv_p.is_file(),
        "csv_url": "/api/v1/benchmark/results/csv" if csv_p.is_file() else None,
    }


@router.get("/benchmark/results/csv")
def download_benchmark_csv() -> FileResponse:
    ctx = _get_ctx()
    csv_p = ctx["outputs_dir"] / "benchmark" / "comparison" / "latest.csv"
    if not csv_p.is_file():
        raise HTTPException(status_code=404, detail="Comparison benchmark CSV not generated yet")
    return FileResponse(str(csv_p), media_type="text/csv", filename="cpu_npu_comparison_latest.csv")


@router.get("/zones")
def get_zones(scene_id: str | None = Query(default=None)) -> dict[str, Any]:
    ctx = _get_ctx()
    store: ZoneStore = ctx["zone_store"]
    try:
        return store.load_zones(scene_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/zones")
def create_or_save_zones(payload: ZoneSavePayload) -> dict[str, Any]:
    ctx = _get_ctx()
    store: ZoneStore = ctx["zone_store"]
    try:
        if payload.zones is not None:
            return store.save_zones(
                payload.zones,
                scene_id=payload.scene_id,
                anchor_mode=payload.anchor_mode,
            )
        if payload.name is None and payload.points is None and payload.polygon_norm is None:
            raise ValueError("Provide either 'zones' list or single zone 'name' and 'points'")
        return store.add_zone(
            {
                "zone_id": payload.zone_id,
                "name": payload.name or "New Zone",
                "type": payload.type,
                "color_bgr": payload.color_bgr,
                "points": payload.points,
                "polygon_norm": payload.polygon_norm,
            },
            scene_id=payload.scene_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/zones/calibration")
def save_zone_calibration(payload: ZoneCalibrationPayload) -> dict[str, Any]:
    ctx = _get_ctx()
    store: ZoneStore = ctx["zone_store"]
    sid = payload.scene_id or "default"
    try:
        cal = compute_homography_calibration(
            payload.image_points_norm,
            payload.world_points_m,
            frame_width=payload.frame_width,
            frame_height=payload.frame_height,
        )
        return store.save_calibration(sid, cal)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/zones/calibration")
def delete_zone_calibration(scene_id: str | None = Query(default=None)) -> dict[str, Any]:
    ctx = _get_ctx()
    store: ZoneStore = ctx["zone_store"]
    sid = scene_id or "default"
    try:
        return store.save_calibration(sid, None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/zones/{zone_id}")
def update_zone(
    zone_id: str,
    payload: ZoneUpdatePayload,
    scene_id: str | None = Query(default=None),
) -> dict[str, Any]:
    ctx = _get_ctx()
    store: ZoneStore = ctx["zone_store"]
    try:
        return store.update_zone(zone_id, payload.model_dump(exclude_none=True), scene_id=scene_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/zones/{zone_id}")
def delete_zone(
    zone_id: str,
    scene_id: str | None = Query(default=None),
) -> dict[str, Any]:
    ctx = _get_ctx()
    store: ZoneStore = ctx["zone_store"]
    try:
        return store.delete_zone(zone_id, scene_id=scene_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/zones/preview")
async def upload_zone_preview(file: UploadFile = File(...)) -> dict[str, Any]:
    ctx = _get_ctx()
    store: ZoneStore = ctx["zone_store"]
    filename = file.filename or "scene.jpg"
    ext = Path(filename).suffix.lower()
    if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported preview format '{ext}'. Allowed: {sorted(IMAGE_EXTS | VIDEO_EXTS)}",
        )
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded preview file is empty")
    try:
        meta = store.extract_preview_from_bytes(
            content,
            filename,
            is_video=(ext in VIDEO_EXTS),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    zones_cfg = store.load_zones(meta["scene_id"])
    return {**meta, "zones_config": zones_cfg}


@router.get("/zones/preview/{scene_id}")
def get_zone_preview_image(scene_id: str) -> FileResponse:
    ctx = _get_ctx()
    store: ZoneStore = ctx["zone_store"]
    try:
        p = store.get_preview_path(scene_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"Preview frame for scene '{scene_id}' not found")
    return FileResponse(str(p), media_type="image/jpeg")


@router.get("/inspections/{inspection_id}/preview")
def get_inspection_preview(inspection_id: str) -> dict[str, Any]:
    ctx = _get_ctx()
    store: ZoneStore = ctx["zone_store"]
    with ctx["jobs_lock"]:
        job = ctx["jobs"].get(inspection_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Inspection {inspection_id} not found")
        paths = list(job.get("input_paths", []))
        input_type = job.get("input_type", "images")
        opts = dict(job.get("options", {}))
    if not paths:
        raise HTTPException(status_code=404, detail="No input media found for this inspection")
    scene_id = opts.get("scene_id") or inspection_id
    meta = store.extract_preview_from_file(Path(paths[0]), scene_id=scene_id, is_video=(input_type == "video"))
    zones_cfg = store.load_zones(scene_id)
    return {**meta, "zones_config": zones_cfg}


@router.post("/zones/suggest")
def suggest_zones(scene_id: str | None = Query(default=None)) -> dict[str, Any]:
    ctx = _get_ctx()
    store: ZoneStore = ctx["zone_store"]
    img_bgr = None
    dets = None
    if scene_id:
        try:
            p = store.get_preview_path(scene_id)
            if p.is_file():
                img_bgr = cv2.imread(str(p))
        except ValueError:
            img_bgr = None

    # Use CPU detector predictions strictly as auxiliary sanity cue when preview exists and model is ready
    if img_bgr is not None:
        manager: EngineManager = ctx["engine_manager"]
        if manager.cpu_engine.is_available():
            try:
                dets, _ = manager.cpu_engine.predict_frame(img_bgr, conf_thres=0.35)
            except Exception:
                dets = None

    suggestion = estimate_scene_zones(img_bgr=img_bgr, detections=dets)
    return {
        "scene_id": scene_id or "default",
        "heuristic_only": suggestion.get("method") != "cv_geometry_perspective",
        **suggestion,
    }


@router.get("/zones/scenes")
def list_zone_scenes() -> dict[str, Any]:
    ctx = _get_ctx()
    store: ZoneStore = ctx["zone_store"]
    return {"scenes": store.list_scenes()}


@router.post("/zones/reset")
def reset_scene_zones(scene_id: str | None = Query(default=None)) -> dict[str, Any]:
    ctx = _get_ctx()
    store: ZoneStore = ctx["zone_store"]
    try:
        return store.reset_scene_zones(scene_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
