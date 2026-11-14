"""FastAPI offline backend for SafeVision Edge — Offline Junction Safety Intelligence."""

from __future__ import annotations

import json
import os
import platform
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import sys

SRC_DIR = Path(__file__).resolve().parent
PKG_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from counter import (  # noqa: E402
    blur_persons,
    build_csv_row,
    render_annotations,
    write_counts_csv,
)
from benchmarks.comparison_runner import compute_detection_agreement  # noqa: E402
from engines.base_engine import EngineMode, detection_to_dict  # noqa: E402
from engines.cpu_engine import CPUInferenceEngine  # noqa: E402
from engines.engine_manager import EngineManager, InvalidEngineModeError  # noqa: E402
from engines.hailo_engine import HailoInferenceEngine  # noqa: E402
from infer import IMAGE_EXTS, VIDEO_EXTS, EdgeDetector, load_configs  # noqa: E402
from routes_extended import router as extended_router  # noqa: E402
from safety_pulse import (  # noqa: E402
    ALL_CLASSES,
    PedestrianHeatmapAccumulator,
    analyze_frame,
    compute_temporal_occupancy,
    generate_safety_brief,
    load_safety_config,
)
from zone_analysis import ZoneAnalyzer  # noqa: E402
from zone_store import ZoneStore  # noqa: E402

CONFIG_DIR = PKG_ROOT / "config"
MODEL_PATH = Path(os.environ.get("SAFEVISION_MODEL_PATH", PKG_ROOT / "models" / "custom_model.pte"))
def _resolve_hef_path() -> Path:
    env_p = os.environ.get("SAFEVISION_HEF_PATH")
    if env_p:
        return Path(env_p)
    cfg_file = CONFIG_DIR / "model_config.json"
    if cfg_file.is_file():
        try:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            rel = cfg.get("models", {}).get("npu", {}).get("file")
            if rel:
                return PKG_ROOT / rel
        except Exception:
            pass
    return PKG_ROOT / "models" / "npu" / "custom_model.hef"


HEF_MODEL_PATH = _resolve_hef_path()
INSPECTIONS_DIR = PKG_ROOT / "outputs" / "inspections"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MiB
MAX_IMAGES = 20

_DETECTOR_LOCK = threading.Lock()
_DETECTOR: EdgeDetector | None = None
_MODEL_ERROR: str | None = None
_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}
_EXT_LOCK = threading.Lock()
_ENGINE_MANAGER: EngineManager | None = None
_ZONE_STORE: ZoneStore | None = None


def _load_detector_once() -> EdgeDetector | None:
    global _DETECTOR, _MODEL_ERROR
    with _DETECTOR_LOCK:
        if _DETECTOR is None and _MODEL_ERROR is None:
            try:
                class_names, prep_cfg, _ = load_configs(CONFIG_DIR)
                _DETECTOR = EdgeDetector(MODEL_PATH, class_names, prep_cfg)
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                _MODEL_ERROR = str(exc)
        return _DETECTOR


def _get_engine_manager() -> EngineManager:
    global _ENGINE_MANAGER
    with _EXT_LOCK:
        if _ENGINE_MANAGER is None:
            class_names, prep_cfg, _ = load_configs(CONFIG_DIR)
            det = _load_detector_once()
            cpu_eng = CPUInferenceEngine(MODEL_PATH, class_names, prep_cfg, detector=det)
            npu_eng = HailoInferenceEngine(HEF_MODEL_PATH, class_names, prep_cfg)
            _ENGINE_MANAGER = EngineManager(cpu_engine=cpu_eng, npu_engine=npu_eng, initial_mode=EngineMode.CPU)
        return _ENGINE_MANAGER


def _get_zone_store() -> ZoneStore:
    global _ZONE_STORE
    with _EXT_LOCK:
        if _ZONE_STORE is None:
            _ZONE_STORE = ZoneStore(CONFIG_DIR, PKG_ROOT / "outputs" / "zone_previews")
        return _ZONE_STORE


def get_extended_context() -> dict[str, Any]:
    return {
        "pkg_root": PKG_ROOT,
        "config_dir": CONFIG_DIR,
        "outputs_dir": PKG_ROOT / "outputs",
        "hef_path": HEF_MODEL_PATH,
        "engine_manager": _get_engine_manager(),
        "zone_store": _get_zone_store(),
        "jobs": _JOBS,
        "jobs_lock": _JOBS_LOCK,
    }


def _read_pi_thermals(is_pi: bool) -> tuple[float | None, bool | None]:
    """Read real ARM Linux thermal / throttle status if available; never fabricate values."""
    if not is_pi:
        return None, None
    temp_c: float | None = None
    throttled: bool | None = None
    thermal_file = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal_file.is_file():
        try:
            temp_c = round(int(thermal_file.read_text(encoding="utf-8").strip()) / 1000.0, 1)
        except Exception:
            temp_c = None
    if shutil.which("vcgencmd"):
        try:
            import subprocess

            out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True, timeout=1.5).strip()
            if "=" in out:
                hex_val = int(out.split("=", 1)[1], 16)
                throttled = (hex_val & 0xF) != 0
        except Exception:
            throttled = None
    return temp_c, throttled


def _build_explanation(safety: dict[str, Any], zones_enabled: bool) -> str:
    """Create a transparent rule breakdown matching exact backend Safety Pulse math."""
    veh = safety["total_vehicles"]
    d_cls = safety["density_class"]
    d_sc = safety["density_score"]
    p_cross = safety["persons_crossing"]
    v_cross = safety["vehicles_crossing"]
    cop = safety["co_presence"]
    cop_sc = safety["co_presence_score"]
    vru = safety["vru_share"]
    vru_sc = safety["vru_score"]
    score = safety["safety_pulse"]
    label = safety["safety_pulse_label"]
    zone_part = (
        f"Crossing Co-Presence {cop} (+{cop_sc}: {p_cross} person & {v_cross} vehicle in crossing)"
        if zones_enabled
        else "Zones disabled (Crossing Co-Presence NO +0)"
    )
    return (
        f"Density {d_cls} (+{d_sc} for {veh} vehicles) + "
        f"{zone_part} + "
        f"VRU Share {vru:.2f} (+{vru_sc} {'>=' if vru >= 0.5 and safety['total_detections'] > 0 else '<'} 0.50) "
        f"= {score} ({label})."
    )


def _save_job_meta(job_dir: Path, payload: dict[str, Any]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_saved_jobs() -> None:
    if not INSPECTIONS_DIR.exists():
        return
    for d in sorted(INSPECTIONS_DIR.iterdir()):
        jf = d / "job.json"
        if d.is_dir() and jf.is_file():
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                if data.get("status") == "processing":
                    data["status"] = "failed"
                    data["current_stage"] = "failed"
                    data["error"] = "Interrupted by server restart"
                _JOBS[data["inspection_id"]] = data
            except Exception:
                continue


class ProcessRequest(BaseModel):
    confidence_threshold: float | None = Field(default=None, ge=0.10, le=0.90)
    sample_count: int | None = Field(default=None, ge=10, le=20)
    enable_zones: bool | None = None
    enable_heatmap: bool | None = None
    enable_person_blurring: bool | None = None
    generate_safety_brief: bool | None = None
    engine_mode: str | None = None
    scene_id: str | None = None


class ZoneItem(BaseModel):
    name: str
    color_bgr: list[int] | None = None
    polygon_norm: list[list[float]]
    extra_polygons_norm: list[list[list[float]]] | None = None


class ZoneConfigPayload(BaseModel):
    anchor_mode: str = "bottom_center"
    zones: list[ZoneItem]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    INSPECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    _load_saved_jobs()
    _load_detector_once()
    yield


app = FastAPI(
    title="SafeVision Edge API",
    version="1.0.0",
    description="Offline ExecuTorch + XNNPACK Junction Safety Intelligence API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _no_cache_static_middleware(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if (
        path == "/"
        or path.startswith("/assets")
        or path.endswith(".html")
        or path.endswith(".js")
        or path.endswith(".css")
    ):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/api/v1/system/health")
def get_health() -> dict[str, Any]:
    det = _load_detector_once()
    mach = platform.machine()
    is_pi = "aarch64" in mach.lower() or "arm" in mach.lower()
    size_mb = round(MODEL_PATH.stat().st_size / (1024 ** 2), 2) if MODEL_PATH.exists() else 0.0
    quant = "FP32" if "fp32" in MODEL_PATH.name.lower() else "INT8 PT2E"
    return {
        "status": "online",
        "mode": "offline",
        "device": platform.platform(),
        "cpu_architecture": mach,
        "processor": platform.processor(),
        "is_raspberry_pi": is_pi,
        "python_version": platform.python_version(),
        "model": {
            "name": MODEL_PATH.name,
            "format": "ExecuTorch PTE",
            "quantization": quant,
            "size_mb": size_mb,
            "loaded": det is not None,
            "error": _MODEL_ERROR,
            "classes": list(ALL_CLASSES),
        },
        "runtime": {
            "name": "ExecuTorch",
            "backend": "XNNPACK",
        },
        "engines": _get_engine_manager().get_status(),
        "limits": {
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "max_images": MAX_IMAGES,
            "sample_count_min": 10,
            "sample_count_max": 20,
            "allowed_image_extensions": sorted(IMAGE_EXTS),
            "allowed_video_extensions": sorted(VIDEO_EXTS),
        },
    }


@app.get("/api/v1/config/zones")
def get_zones_config() -> dict[str, Any]:
    zpath = CONFIG_DIR / "zones.json"
    if not zpath.is_file():
        raise HTTPException(status_code=404, detail="zones.json not found")
    return json.loads(zpath.read_text(encoding="utf-8"))


@app.put("/api/v1/config/zones")
def update_zones_config(payload: ZoneConfigPayload) -> dict[str, Any]:
    if not payload.zones:
        raise HTTPException(status_code=400, detail="At least one zone polygon is required")
    for z in payload.zones:
        if not z.name.strip():
            raise HTTPException(status_code=400, detail="Zone name cannot be empty")
        if len(z.polygon_norm) < 3:
            raise HTTPException(status_code=400, detail=f"Zone '{z.name}' must have >= 3 vertices")
        for pt in z.polygon_norm:
            if len(pt) != 2 or not (0.0 <= pt[0] <= 1.0 and 0.0 <= pt[1] <= 1.0):
                raise HTTPException(status_code=400, detail=f"Zone '{z.name}' has coordinates outside [0, 1]")

    data = {
        "anchor_mode": payload.anchor_mode,
        "zones": [z.model_dump(exclude_none=True) for z in payload.zones],
    }
    (CONFIG_DIR / "zones.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


@app.get("/api/v1/config/safety")
def get_safety_config() -> dict[str, Any]:
    return load_safety_config(CONFIG_DIR / "safety_config.json")


@app.get("/api/v1/inspections")
def list_inspections() -> list[dict[str, Any]]:
    with _JOBS_LOCK:
        items = list(_JOBS.values())
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return [
        {
            "inspection_id": it["inspection_id"],
            "created_at": it.get("created_at"),
            "status": it.get("status"),
            "input_type": it.get("input_type"),
            "filenames": it.get("filenames", []),
            "total_frames": it.get("total_frames", 0),
        }
        for it in items
    ]


@app.post("/api/v1/inspections")
async def create_inspection(
    files: list[UploadFile] = File(...),
    sample_count: int = Form(15),
    confidence_threshold: float = Form(0.25),
    enable_zones: bool = Form(True),
    enable_heatmap: bool = Form(True),
    enable_person_blurring: bool = Form(True),
    generate_safety_brief: bool = Form(True),
    engine_mode: str | None = Form(None),
    scene_id: str | None = Form(None),
) -> dict[str, Any]:
    manager = _get_engine_manager()
    try:
        resolved_mode = EngineManager.validate_mode(engine_mode) if engine_mode is not None else manager.selected_mode
    except InvalidEngineModeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not files:
        raise HTTPException(status_code=400, detail="At least one image or video file must be uploaded")
    if sample_count < 10 or sample_count > 20:
        raise HTTPException(status_code=400, detail="sample_count must be between 10 and 20")
    if confidence_threshold < 0.10 or confidence_threshold > 0.90:
        raise HTTPException(status_code=400, detail="confidence_threshold must be between 0.10 and 0.90")

    exts = [Path(f.filename or "").suffix.lower() for f in files]
    is_imgs = [e in IMAGE_EXTS for e in exts]
    is_vids = [e in VIDEO_EXTS for e in exts]
    if any(not (im or vd) for im, vd in zip(is_imgs, is_vids)):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed images: {sorted(IMAGE_EXTS)}, videos: {sorted(VIDEO_EXTS)}",
        )
    if any(is_imgs) and any(is_vids):
        raise HTTPException(status_code=400, detail="Do not mix image and video files in one inspection")
    if all(is_vids) and len(files) > 1:
        raise HTTPException(status_code=400, detail="Upload one video file per inspection")
    if all(is_imgs) and len(files) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_IMAGES} images allowed per inspection")

    inspection_id = f"INS-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"
    job_dir = INSPECTIONS_DIR / inspection_id
    input_dir = job_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    filenames: list[str] = []
    total_bytes = 0
    try:
        for idx, up in enumerate(files):
            safe_name = f"{idx:02d}_{Path(up.filename or f'file_{idx}').name}"
            content = await up.read()
            if not content:
                raise HTTPException(status_code=400, detail=f"Uploaded file '{up.filename}' is empty")
            total_bytes += len(content)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit",
                )
            dest = input_dir / safe_name
            dest.write_bytes(content)
            saved_paths.append(str(dest))
            filenames.append(Path(up.filename or safe_name).name)
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    input_type = "video" if all(is_vids) else "images"
    if input_type == "video":
        cap = cv2.VideoCapture(saved_paths[0])
        if not cap.isOpened():
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="Uploaded video cannot be decoded cleanly by OpenCV")
        raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()
        if raw_frames <= 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="Uploaded video contains 0 readable frames")
        planned_total = min(sample_count, raw_frames)
    else:
        for spath in saved_paths:
            if cv2.imread(spath) is None:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(status_code=400, detail=f"Invalid or corrupted image: {Path(spath).name}")
        planned_total = len(saved_paths)

    job_meta = {
        "inspection_id": inspection_id,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": "queued",
        "current_stage": "queued",
        "progress": 0,
        "processed_frames": 0,
        "total_frames": planned_total,
        "error": None,
        "input_type": input_type,
        "filenames": filenames,
        "input_paths": saved_paths,
        "options": {
            "sample_count": int(sample_count),
            "confidence_threshold": float(confidence_threshold),
            "enable_zones": bool(enable_zones),
            "enable_heatmap": bool(enable_heatmap),
            "enable_person_blurring": bool(enable_person_blurring),
            "generate_safety_brief": bool(generate_safety_brief),
            "engine_mode": resolved_mode,
            "scene_id": scene_id.strip() if isinstance(scene_id, str) and scene_id.strip() else None,
        },
    }
    _save_job_meta(job_dir, job_meta)
    with _JOBS_LOCK:
        _JOBS[inspection_id] = job_meta

    return {
        "inspection_id": inspection_id,
        "status": "queued",
        "input_type": input_type,
        "filenames": filenames,
        "total_frames": planned_total,
        "sample_count": sample_count,
        "engine_mode": resolved_mode,
        "scene_id": job_meta["options"]["scene_id"],
    }


def _run_inspection_worker(inspection_id: str) -> None:
    job_dir = INSPECTIONS_DIR / inspection_id
    with _JOBS_LOCK:
        job = _JOBS[inspection_id]
        opts = dict(job["options"])
        input_type = job["input_type"]
        input_paths = list(job["input_paths"])
        filenames = list(job["filenames"])

    def _update(**fields: Any) -> None:
        with _JOBS_LOCK:
            _JOBS[inspection_id].update(fields)
            snapshot = dict(_JOBS[inspection_id])
        _save_job_meta(job_dir, snapshot)

    try:
        _update(status="processing", current_stage="loading_model", progress=0, processed_frames=0, error=None)
        manager = _get_engine_manager()
        engine_mode = EngineManager.validate_mode(opts.get("engine_mode") or manager.selected_mode)
        manager.require_ready_for_mode(engine_mode)

        if engine_mode in (EngineMode.CPU, EngineMode.COMPARE):
            manager.cpu_engine.load()
        if engine_mode in (EngineMode.NPU, EngineMode.COMPARE):
            manager.npu_engine.load()

        safety_cfg = load_safety_config(CONFIG_DIR / "safety_config.json")
        zone_analyzer: ZoneAnalyzer | None = None
        used_scene_id: str | None = opts.get("scene_id")
        used_zone_cfg: dict[str, Any] | None = None
        if opts["enable_zones"]:
            zone_store = _get_zone_store()
            req_scene = opts.get("scene_id")
            if req_scene and zone_store.has_scene_config(req_scene):
                used_zone_cfg = zone_store.load_zones(req_scene)
                used_scene_id = req_scene
            elif zone_store.has_scene_config(inspection_id):
                used_zone_cfg = zone_store.load_zones(inspection_id)
                used_scene_id = inspection_id
            else:
                used_zone_cfg = zone_store.load_zones("default")
                used_scene_id = used_scene_id or "default"
            zone_analyzer = ZoneAnalyzer(used_zone_cfg)

        samples: list[tuple[int, int, str, float, np.ndarray]] = []
        if input_type == "images":
            for idx, (spath, orig_name) in enumerate(zip(input_paths, filenames), start=1):
                img = cv2.imread(spath)
                if img is None:
                    raise RuntimeError(f"Failed to read uploaded image {orig_name}")
                samples.append((idx, idx - 1, orig_name, float(idx - 1), img))
        else:
            cap = cv2.VideoCapture(input_paths[0])
            if not cap.isOpened():
                raise RuntimeError("Failed to reopen uploaded video")
            fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
            total_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            n_target = min(int(opts["sample_count"]), total_vid)
            indices = [int(i) for i in np.linspace(0, total_vid - 1, n_target, dtype=int)]
            seen_idx = set()
            unique_indices = []
            for x in indices:
                if x not in seen_idx:
                    seen_idx.add(x)
                    unique_indices.append(x)
            for out_idx, src_idx in enumerate(unique_indices, start=1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                samples.append((out_idx, src_idx, f"frame_{src_idx:04d}", round(src_idx / fps, 2), frame))
            cap.release()

        if not samples:
            raise RuntimeError("No readable frames extracted from uploaded input")

        total_frames = len(samples)
        _update(current_stage="detecting", total_frames=total_frames, processed_frames=0, progress=0)

        frames_out_dir = job_dir / "frames"
        frames_out_dir.mkdir(parents=True, exist_ok=True)
        heatmap = PedestrianHeatmapAccumulator()
        csv_rows: list[dict[str, object]] = []
        zone_logs: list[dict[str, Any]] = []
        api_frames: list[dict[str, Any]] = []
        class_totals: dict[str, int] = {cls_name: 0 for cls_name in ALL_CLASSES}

        proc = psutil.Process()
        proc.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

        blur_on = bool(opts["enable_person_blurring"])
        zones_on = bool(opts["enable_zones"])
        conf_thres = float(opts["confidence_threshold"])
        cpu_dets_all: list[list[Any]] = []
        npu_dets_all: list[list[Any]] = []
        npu_inf_list: list[float] = []

        for frame_id, src_idx, src_label, ts_sec, frame_bgr in samples:
            t_frame0 = time.perf_counter()
            npu_frame_meta: dict[str, Any] | None = None
            if engine_mode == EngineMode.CPU:
                dets, inf_ms = manager.cpu_engine.predict_frame(frame_bgr, conf_thres=conf_thres)
            elif engine_mode == EngineMode.NPU:
                dets, inf_ms = manager.npu_engine.predict_frame(frame_bgr, conf_thres=conf_thres)
            else:
                # Compare Both: run both CPU and NPU on the exact same frame
                dets, inf_ms = manager.cpu_engine.predict_frame(frame_bgr, conf_thres=conf_thres)
                npu_dets, npu_inf_ms = manager.npu_engine.predict_frame(frame_bgr, conf_thres=conf_thres)
                npu_safety = analyze_frame(npu_dets, frame_bgr.shape[:2], zone_analyzer, cfg=safety_cfg)
                cpu_dets_all.append(dets)
                npu_dets_all.append(npu_dets)
                npu_inf_list.append(npu_inf_ms)
                npu_frame_meta = {
                    "inference_ms": round(npu_inf_ms, 2),
                    "total_detections": len(npu_dets),
                    "counts": npu_safety["counts"],
                    "safety_pulse": npu_safety["safety_pulse"],
                    "detections": [detection_to_dict(d) for d in npu_dets],
                }
            safety = analyze_frame(dets, frame_bgr.shape[:2], zone_analyzer, cfg=safety_cfg)
            if opts["enable_heatmap"]:
                heatmap.update(dets, frame_bgr.shape[:2])

            vis = blur_persons(frame_bgr, dets) if blur_on else frame_bgr.copy()
            if zone_analyzer is not None:
                vis = zone_analyzer.draw_zones(vis, safety["zone_counts"])

            pre_draw_tot = max(inf_ms, (time.perf_counter() - t_frame0) * 1000.0)
            vis = render_annotations(
                vis,
                dets,
                safety,
                inf_ms,
                pre_draw_tot,
                frame_label=f"#{frame_id} ({src_label})",
                blur_privacy=False,
            )
            cv2.imwrite(str(frames_out_dir / f"{frame_id}.jpg"), vis)
            total_ms = max(inf_ms, (time.perf_counter() - t_frame0) * 1000.0)

            csv_rows.append(build_csv_row(frame_id, safety, inf_ms, total_ms))
            zone_logs.append({"frame_id": frame_id, "zones": safety["zone_counts"]})

            for cname in ALL_CLASSES:
                class_totals[cname] += int(safety["counts"].get(cname, 0))

            api_frames.append(
                {
                    "frame_id": frame_id,
                    "source_frame_index": src_idx,
                    "source_label": src_label,
                    "timestamp_seconds": ts_sec,
                    "counts": safety["counts"],
                    "zones": {
                        "road": {
                            "person": safety["persons_road"],
                            "vehicles": safety["vehicles_road"],
                            "total": safety["persons_road"] + safety["vehicles_road"],
                        },
                        "crossing": {
                            "person": safety["persons_crossing"],
                            "vehicles": safety["vehicles_crossing"],
                            "total": safety["persons_crossing"] + safety["vehicles_crossing"],
                        },
                        "footpath": {
                            "person": safety["persons_footpath"],
                            "vehicles": safety["vehicles_footpath"],
                            "total": safety["persons_footpath"] + safety["vehicles_footpath"],
                        },
                        "unassigned": {
                            "total": safety["unassigned_count"],
                            "person": safety["unassigned_counts"].get("person", 0),
                            "vehicles": safety["unassigned_counts"].get("vehicles", 0),
                            "by_class": {c: safety["unassigned_counts"].get(c, 0) for c in ALL_CLASSES},
                        },
                        "by_category_classes": safety.get("category_class_counts", {}),
                        "by_zone": safety.get("zone_counts", {}),
                        "boundary_assignments": safety.get("boundary_assignments", 0),
                    },
                    "calibrated_densities": safety.get("calibrated_densities", {"calibrated": False, "physical_density_available": False, "per_zone": {}}),
                    "persons_road": safety["persons_road"],
                    "persons_crossing": safety["persons_crossing"],
                    "persons_footpath": safety["persons_footpath"],
                    "vehicles_road": safety["vehicles_road"],
                    "vehicles_crossing": safety["vehicles_crossing"],
                    "vehicles_footpath": safety["vehicles_footpath"],
                    "vehicles": safety["total_vehicles"],
                    "density": safety["density_class"],
                    "density_score": safety["density_score"],
                    "co_presence": safety["co_presence"] == "YES",
                    "co_presence_score": safety["co_presence_score"],
                    "vru_count": safety["vru_count"],
                    "total": safety["total_detections"],
                    "vru_share": safety["vru_share"],
                    "vru_score": safety["vru_score"],
                    "safety_pulse": {
                        "score": safety["safety_pulse"],
                        "label": safety["safety_pulse_label"],
                        "explanation": _build_explanation(safety, zones_on),
                    },
                    "inference_ms": round(inf_ms, 2),
                    "total_ms": round(total_ms, 2),
                    "annotated_frame_url": f"/api/v1/inspections/{inspection_id}/frames/{frame_id}",
                    **({"npu_comparison": npu_frame_meta} if npu_frame_meta is not None else {}),
                }
            )
            pct = int(round((frame_id / total_frames) * 95))
            _update(processed_frames=frame_id, progress=pct, current_stage="detecting")

        _update(current_stage="generating_reports", progress=97)
        csv_path = job_dir / f"{inspection_id}_counts.csv"
        write_counts_csv(csv_rows, csv_path)
        (job_dir / f"{inspection_id}_counts.zones.json").write_text(json.dumps(zone_logs, indent=2), encoding="utf-8")

        heatmap_path = job_dir / "pedestrian_heatmap.png"
        if opts["enable_heatmap"]:
            heatmap.save_heatmap(heatmap_path)
        heatmap_payload = {
            "available": bool(opts["enable_heatmap"]),
            "cells": [int(x) for x in heatmap.grid.flatten().tolist()] if opts["enable_heatmap"] else [],
            "grid_2d": heatmap.grid.tolist() if opts["enable_heatmap"] else [],
            "total_pedestrian_positions": int(heatmap.total_points) if opts["enable_heatmap"] else 0,
            "image_url": f"/api/v1/inspections/{inspection_id}/heatmap/image" if opts["enable_heatmap"] else None,
        }
        (job_dir / "heatmap.json").write_text(json.dumps(heatmap_payload, indent=2), encoding="utf-8")

        rss_mb = proc.memory_info().rss / (1024 ** 2)
        cpu_pct = proc.cpu_percent(interval=None)
        sys_cpu = psutil.cpu_percent(interval=None)

        active_model_path = HEF_MODEL_PATH if engine_mode == EngineMode.NPU else MODEL_PATH
        active_runtime = (
            "HailoRT (Hailo-8)"
            if engine_mode == EngineMode.NPU
            else ("ExecuTorch + XNNPACK vs HailoRT (Compare)" if engine_mode == EngineMode.COMPARE else "ExecuTorch + XNNPACK")
        )
        active_model_label = (
            f"{HEF_MODEL_PATH.name} (Hailo-8 HEF)"
            if engine_mode == EngineMode.NPU
            else f"{MODEL_PATH.name} ({'FP32' if 'fp32' in MODEL_PATH.name.lower() else 'INT8 PT2E'})"
        )

        brief_md = job_dir / "safety_brief.md"
        brief_json = job_dir / "safety_brief.json"
        if opts["generate_safety_brief"]:
            brief = generate_safety_brief(
                rows=csv_rows,
                zone_logs=zone_logs,
                model_path=active_model_path,
                brief_md_path=brief_md,
                brief_json_path=brief_json,
                rss_mb=rss_mb,
                cpu_percent=cpu_pct,
                heatmap_path=heatmap_path if opts["enable_heatmap"] else None,
                disclaimer=str(safety_cfg.get("disclaimer", "")),
            )
        else:
            brief = None

        peak_f = max(api_frames, key=lambda f: (f["safety_pulse"]["score"], f["persons_crossing"] + f["vehicles_crossing"]))
        alert_frames = sum(1 for f in api_frames if f["safety_pulse"]["label"] == "High")
        cp_frames = sum(1 for f in api_frames if f["co_presence"])
        inf_list = [f["inference_ms"] for f in api_frames]
        tot_list = [f["total_ms"] for f in api_frames]
        mean_inf = float(np.mean(inf_list))
        mean_tot = float(np.mean(tot_list))

        mach = platform.machine()
        is_pi = "aarch64" in mach.lower() or "arm" in mach.lower()
        temp_c, throttled = _read_pi_thermals(is_pi)

        comparison_summary: dict[str, Any] | None = None
        if engine_mode == EngineMode.COMPARE and npu_inf_list:
            npu_mean = float(np.mean(npu_inf_list))
            comparison_summary = {
                "cpu_mean_inference_ms": round(mean_inf, 2),
                "cpu_average_fps": round(1000.0 / mean_inf, 2) if mean_inf > 0 else 0.0,
                "npu_mean_inference_ms": round(npu_mean, 2),
                "npu_average_fps": round(1000.0 / npu_mean, 2) if npu_mean > 0 else 0.0,
                "inference_speedup_x": round(mean_inf / npu_mean, 2) if npu_mean > 0 else None,
                "detection_agreement": compute_detection_agreement(cpu_dets_all, npu_dets_all),
            }

        benchmark_payload = {
            "engine_mode": engine_mode,
            "model_name": active_model_path.name,
            "runtime": active_runtime,
            "hardware": f"{platform.platform()} ({mach})",
            "is_raspberry_pi": is_pi,
            "measured_frames": total_frames,
            "mean_inference_ms": round(mean_inf, 2),
            "median_inference_ms": round(float(np.median(inf_list)), 2),
            "p95_inference_ms": round(float(np.percentile(inf_list, 95)), 2),
            "average_fps": round(1000.0 / mean_inf, 2) if mean_inf > 0 else 0.0,
            "mean_total_ms": round(mean_tot, 2),
            "median_total_ms": round(float(np.median(tot_list)), 2),
            "p95_total_ms": round(float(np.percentile(tot_list, 95)), 2),
            "end_to_end_fps": round(1000.0 / mean_tot, 2) if mean_tot > 0 else 0.0,
            "ram_usage_mb": round(rss_mb, 1),
            "process_cpu_percent": round(cpu_pct, 1),
            "system_cpu_percent": round(sys_cpu, 1),
            "cpu_temperature_c": temp_c,
            "throttling_detected": throttled,
            **({"comparison": comparison_summary} if comparison_summary is not None else {}),
        }
        (job_dir / "benchmark.json").write_text(json.dumps(benchmark_payload, indent=2), encoding="utf-8")

        temporal_occ = compute_temporal_occupancy(api_frames)
        cal_info = (used_zone_cfg or {}).get("calibration") if used_zone_cfg else None
        results_payload = {
            "inspection_id": inspection_id,
            "created_at": job["created_at"],
            "input_type": input_type,
            "engine_mode": engine_mode,
            "scene_id": used_scene_id,
            "model": active_model_label,
            "runtime": active_runtime,
            "options": {**opts, "blur_applied": blur_on},
            "total_frames": total_frames,
            "total_detections": int(sum(class_totals.values())),
            "class_counts": class_totals,
            "summary": {
                "peak_score": peak_f["safety_pulse"]["score"],
                "peak_label": peak_f["safety_pulse"]["label"],
                "peak_frame": peak_f["frame_id"],
                "alert_frames": alert_frames,
                "alert_pct": round(100.0 * alert_frames / total_frames, 1),
                "co_presence_frames": cp_frames,
                "busiest_zone": (brief or {}).get("busiest_zone", {}).get("name", "N/A"),
            },
            "temporal_occupancy": temporal_occ,
            "calibration": cal_info or {"calibrated": False, "physical_density_available": False},
            "zone_schema": used_zone_cfg,
            "indicator_disclaimer": str(safety_cfg.get("disclaimer", "")),
            **({"comparison": comparison_summary} if comparison_summary is not None else {}),
            "frames": api_frames,
        }
        (job_dir / "results.json").write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
        _update(status="completed", current_stage="completed", progress=100, processed_frames=total_frames)
    except Exception as exc:
        _update(status="failed", current_stage="failed", error=str(exc))


@app.post("/api/v1/inspections/{inspection_id}/process")
def start_processing(inspection_id: str, body: ProcessRequest | None = None) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(inspection_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Inspection {inspection_id} not found")
        if job["status"] in {"processing", "completed"}:
            return {"inspection_id": inspection_id, "status": job["status"], "already_started": True}

        if body is not None:
            opts = job["options"]
            if body.engine_mode is not None:
                try:
                    opts["engine_mode"] = EngineManager.validate_mode(body.engine_mode)
                except InvalidEngineModeError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
            if body.scene_id is not None:
                opts["scene_id"] = body.scene_id.strip() or None
            for field_name in (
                "confidence_threshold",
                "sample_count",
                "enable_zones",
                "enable_heatmap",
                "enable_person_blurring",
                "generate_safety_brief",
            ):
                val = getattr(body, field_name)
                if val is not None:
                    opts[field_name] = val
        job["status"] = "processing"
        job["current_stage"] = "loading_model"
        _save_job_meta(INSPECTIONS_DIR / inspection_id, dict(job))

    worker = threading.Thread(target=_run_inspection_worker, args=(inspection_id,), daemon=True)
    worker.start()
    return {"inspection_id": inspection_id, "status": "processing", "already_started": False}


@app.get("/api/v1/inspections/{inspection_id}/status")
def get_status(inspection_id: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(inspection_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Inspection {inspection_id} not found")
        return {
            "inspection_id": inspection_id,
            "status": job["status"],
            "progress": job["progress"],
            "processed_frames": job["processed_frames"],
            "total_frames": job["total_frames"],
            "current_stage": job["current_stage"],
            "error": job.get("error"),
        }


@app.get("/api/v1/inspections/{inspection_id}/results")
def get_results(inspection_id: str) -> dict[str, Any]:
    res_path = INSPECTIONS_DIR / inspection_id / "results.json"
    if not res_path.is_file():
        raise HTTPException(status_code=404, detail="Inspection results not ready or not found")
    return json.loads(res_path.read_text(encoding="utf-8"))


@app.get("/api/v1/inspections/{inspection_id}/frames/{frame_id}")
def get_frame_image(inspection_id: str, frame_id: int) -> FileResponse:
    img_path = INSPECTIONS_DIR / inspection_id / "frames" / f"{frame_id}.jpg"
    if not img_path.is_file():
        raise HTTPException(status_code=404, detail=f"Annotated frame {frame_id} not found")
    return FileResponse(str(img_path), media_type="image/jpeg")


@app.get("/api/v1/inspections/{inspection_id}/heatmap")
def get_heatmap_meta(inspection_id: str) -> dict[str, Any]:
    hm_path = INSPECTIONS_DIR / inspection_id / "heatmap.json"
    if not hm_path.is_file():
        return {"available": False, "cells": [], "grid_2d": [], "total_pedestrian_positions": 0, "image_url": None}
    return json.loads(hm_path.read_text(encoding="utf-8"))


@app.get("/api/v1/inspections/{inspection_id}/heatmap/image")
def get_heatmap_image(inspection_id: str) -> FileResponse:
    png_path = INSPECTIONS_DIR / inspection_id / "pedestrian_heatmap.png"
    if not png_path.is_file():
        raise HTTPException(status_code=404, detail="Heatmap image not generated for this inspection")
    return FileResponse(str(png_path), media_type="image/png")


@app.get("/api/v1/inspections/{inspection_id}/report")
def get_safety_report(inspection_id: str) -> dict[str, Any]:
    brief_path = INSPECTIONS_DIR / inspection_id / "safety_brief.json"
    if not brief_path.is_file():
        return {"available": False}
    data = json.loads(brief_path.read_text(encoding="utf-8"))
    peak = data.get("peak_safety_pulse_frame") or {}
    peak_id = peak.get("frame_id", "N/A")
    peak_score = peak.get("safety_pulse", 0)
    peak_label = peak.get("safety_pulse_label", "Low")
    cp_frames = data.get("co_presence_alert_frames", 0)
    total_f = data.get("total_frames_processed", 0)
    cp_pct = data.get("co_presence_frame_percent", data.get("co_presence_alert_percent", 0.0))
    busiest = (data.get("busiest_zone") or {}).get("name", "N/A")
    avg_inf = data.get("avg_inference_ms", 0.0)
    avg_tot = data.get("avg_total_ms", 0.0)
    approx_fps = data.get("approx_inference_fps", 0.0)
    e2e_fps = data.get("approx_end_to_end_fps", 0.0)
    hw = data.get("hardware") or {}
    model_info = data.get("model") or {}
    brief_text = (
        f"Processed {total_f} frames using {model_info.get('name', 'custom_model.pte')} ({model_info.get('size_mb', 0)} MiB). "
        f"Peak Safety Pulse was {peak_score} ({peak_label}) in frame {peak_id}. "
        f"Crossing pedestrian-vehicle co-presence occurred in {cp_frames} of {total_f} frames ({cp_pct}%). "
        f"Busiest zone was '{busiest}'. Mean inference time: {avg_inf} ms (~{approx_fps} FPS model-only, {avg_tot} ms / {e2e_fps} FPS end-to-end)."
    )
    return {
        "available": True,
        "frames_processed": total_f,
        "peak_risk_frame": peak_id,
        "peak_safety_pulse_score": peak_score,
        "peak_safety_pulse_label": peak_label,
        "co_presence_frames": cp_frames,
        "co_presence_frame_percentage": cp_pct,
        "busiest_zone": busiest,
        "busiest_zone_totals": (data.get("busiest_zone") or {}).get("all_zones", {}),
        "average_inference_ms": avg_inf,
        "average_total_ms": avg_tot,
        "approximate_fps": approx_fps,
        "end_to_end_fps": e2e_fps,
        "model": model_info,
        "hardware": hw,
        "safety_brief": brief_text,
        "indicator_disclaimer": data.get("indicator_disclaimer", ""),
    }


@app.get("/api/v1/inspections/{inspection_id}/export/csv")
def export_csv(inspection_id: str) -> FileResponse:
    csv_path = INSPECTIONS_DIR / inspection_id / f"{inspection_id}_counts.csv"
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail="CSV report not found for this inspection")
    return FileResponse(str(csv_path), media_type="text/csv", filename=f"{inspection_id}_counts.csv")


@app.get("/api/v1/inspections/{inspection_id}/benchmark")
def get_benchmark(inspection_id: str) -> dict[str, Any]:
    bm_path = INSPECTIONS_DIR / inspection_id / "benchmark.json"
    if not bm_path.is_file():
        raise HTTPException(status_code=404, detail="Benchmark not recorded for this inspection")
    return json.loads(bm_path.read_text(encoding="utf-8"))


@app.get("/api/v1/system/benchmark")
def get_system_benchmark() -> dict[str, Any]:
    """Return offline outputs/benchmark.json if generated previously; never fabricate."""
    bm_path = PKG_ROOT / "outputs" / "benchmark.json"
    if not bm_path.is_file():
        return {"available": False}
    raw = json.loads(bm_path.read_text(encoding="utf-8"))
    fwd = raw.get("forward_only_ms", {})
    e2e = raw.get("end_to_end_ms", {})
    mach = str(raw.get("machine", platform.machine()))
    is_pi = bool(raw.get("is_raspberry_pi", False))
    temp_c, throttled = _read_pi_thermals(is_pi)
    return {
        "available": True,
        "model_name": Path(str(raw.get("model_path", MODEL_PATH.name))).name,
        "runtime": "ExecuTorch + XNNPACK",
        "hardware": f"{raw.get('platform', platform.platform())} ({mach})",
        "is_raspberry_pi": is_pi,
        "measured_frames": raw.get("measured_runs"),
        "mean_inference_ms": fwd.get("mean"),
        "median_inference_ms": fwd.get("median"),
        "p95_inference_ms": fwd.get("p95"),
        "average_fps": fwd.get("fps"),
        "mean_total_ms": e2e.get("mean"),
        "median_total_ms": e2e.get("median"),
        "p95_total_ms": e2e.get("p95"),
        "end_to_end_fps": e2e.get("fps"),
        "ram_usage_mb": raw.get("process_rss_mb"),
        "process_cpu_percent": raw.get("process_cpu_percent"),
        "system_cpu_percent": raw.get("system_cpu_percent"),
        "cpu_temperature_c": temp_c,
        "throttling_detected": throttled,
    }


app.include_router(extended_router)

# Serve bundled offline SPA directly from static/ (self-contained package)
_STATIC_DIR = PKG_ROOT / "static" if (PKG_ROOT / "static" / "index.html").is_file() else None
if _STATIC_DIR is not None:
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="spa")


def main() -> int:  # pragma: no cover
    import uvicorn

    host = os.environ.get("SAFEVISION_HOST", "0.0.0.0")
    port = int(os.environ.get("SAFEVISION_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
