"""Export fine-tuned IDD YOLOv11-S PyTorch checkpoint (best.pt) to validated static ONNX."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_CLASSES = [
    "person",
    "bicycle",
    "motorcycle",
    "auto_rickshaw",
    "car",
    "bus",
    "truck",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def validate_onnx_model(onnx_path: Path, imgsz: int = 640) -> dict[str, Any]:
    """Run onnx.checker and optional onnxruntime shape verification."""
    import onnx  # type: ignore[import-not-found]

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    inputs = [
        {
            "name": inp.name,
            "shape": [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim],
        }
        for inp in model.graph.input
    ]
    outputs = [
        {
            "name": out.name,
            "shape": [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim],
        }
        for out in model.graph.output
    ]
    op_types = sorted({n.op_type for n in model.graph.node})

    ort_shapes: list[list[int]] = []
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        outs = sess.run(None, {in_name: dummy})
        ort_shapes = [list(o.shape) for o in outs]
    except ImportError:
        pass

    return {
        "onnx_path": str(onnx_path),
        "sha256": _sha256(onnx_path),
        "opset_imports": [{"domain": o.domain or "ai.onnx", "version": int(o.version)} for o in model.opset_import],
        "inputs": inputs,
        "outputs": outputs,
        "ort_verified_output_shapes": ort_shapes,
        "op_types_count": len(op_types),
        "op_types": op_types,
    }


def export_pt_to_onnx(
    weights_path: Path,
    out_onnx: Path,
    imgsz: int = 640,
    opset: int = 12,
) -> dict[str, Any]:
    """Export YOLOv11-S .pt checkpoint to static [1,3,640,640] ONNX and validate."""
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Source checkpoint not found: {weights_path}. "
            "Expected fine-tuned IDD YOLOv11-S checkpoint (outputs/training/IDD_YOLOv11S/weights/best.pt)."
        )

    try:
        ultra = importlib.import_module("ultralytics")
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required ONLY on the offline export workstation to convert best.pt -> ONNX. "
            "Install it in a separate conversion environment (not inside the Raspberry Pi runtime venv)."
        ) from exc

    yolo_model = ultra.YOLO(str(weights_path))
    names = getattr(yolo_model, "names", {})
    ordered_names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    if ordered_names and ordered_names != EXPECTED_CLASSES:
        raise ValueError(f"Class mapping mismatch: {ordered_names} != {EXPECTED_CLASSES}")

    exported = yolo_model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=True,
        dynamic=False,
        batch=1,
    )
    exported_path = Path(str(exported))
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != out_onnx.resolve():
        out_onnx.write_bytes(exported_path.read_bytes())

    report = validate_onnx_model(out_onnx, imgsz=imgsz)
    report["source_weights"] = str(weights_path)
    report["source_weights_sha256"] = _sha256(weights_path)
    report["classes"] = EXPECTED_CLASSES
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLOv11-S best.pt to validated static ONNX for Hailo-8.")
    parser.add_argument("--weights", type=Path, default=Path("outputs/training/IDD_YOLOv11S/weights/best.pt"))
    parser.add_argument("--out", type=Path, default=Path("models/custom_model.onnx"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--validate-only", action="store_true", help="Validate an existing ONNX file without exporting.")
    args = parser.parse_args()

    if args.validate_only:
        report = validate_onnx_model(args.out, imgsz=args.imgsz)
    else:
        report = export_pt_to_onnx(args.weights, args.out, imgsz=args.imgsz, opset=args.opset)

    prov_path = args.out.with_suffix(".onnx_provenance.json")
    prov_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
