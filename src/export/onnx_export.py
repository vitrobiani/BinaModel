"""
src/export/onnx_export.py
─────────────────────────
Export Ultralytics .pt → ONNX (Generic_Traning_Plan §4.4).

Usage:
  # Export every specialist's best.pt next to itself as best.onnx
  python src/export/onnx_export.py --target specialists

  # Export the student
  python src/export/onnx_export.py --target student

  # Export a specific checkpoint
  python src/export/onnx_export.py --ckpt runs/student/bina_v1/weights/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONDITIONS = ["caries", "gingivitis", "plaque", "discoloration", "ulcer", "recession"]


def export_ckpt(ckpt: Path, *, imgsz: int = 640, opset: int = 12,
                dynamic: bool = False, simplify: bool = True) -> Path:
    """Export a single .pt checkpoint to ONNX. Returns the .onnx path."""
    from ultralytics import YOLO  # imported lazily so --help works without torch

    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    print(f"  exporting {ckpt}")
    model = YOLO(str(ckpt))
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
    )
    onnx_path = Path(onnx_path)
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  → {onnx_path}  ({size_mb:.2f} MB)")
    return onnx_path


def export_all_specialists(imgsz: int = 640) -> list[Path]:
    results = []
    base = ROOT / "runs" / "specialists"
    for cond in CONDITIONS:
        ckpt = base / f"specialist_{cond}" / "weights" / "best.pt"
        if not ckpt.exists():
            print(f"  skip {cond}: no checkpoint at {ckpt}")
            continue
        results.append(export_ckpt(ckpt, imgsz=imgsz))
    return results


def export_student(imgsz: int = 640) -> Path | None:
    ckpt = ROOT / "runs" / "student" / "bina_v1" / "weights" / "best.pt"
    if not ckpt.exists():
        print(f"  no student checkpoint at {ckpt}")
        return None
    return export_ckpt(ckpt, imgsz=imgsz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["specialists", "student", "all"],
                        default="all")
    parser.add_argument("--ckpt", default=None,
                        help="explicit path to a single .pt to export")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if args.ckpt:
        export_ckpt(Path(args.ckpt), imgsz=args.imgsz)
    else:
        if args.target in ("specialists", "all"):
            export_all_specialists(imgsz=args.imgsz)
        if args.target in ("student", "all"):
            export_student(imgsz=args.imgsz)
