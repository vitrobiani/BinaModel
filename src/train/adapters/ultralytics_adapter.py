"""
UltralyticsAdapter — wraps the Ultralytics YOLO/RT-DETR flow behind the
common SpecialistAdapter interface. This is the path Phase-1A already used;
having it as an adapter lets the sweep dispatch uniformly across families.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .base import Prediction, SpecialistAdapter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


_TRAIN_KEYS = (
    "epochs", "imgsz", "batch", "optimizer", "lr0", "lrf", "momentum",
    "weight_decay", "warmup_epochs",
    "hsv_h", "hsv_s", "hsv_v", "fliplr", "flipud", "degrees",
    "translate", "scale", "mosaic", "copy_paste",
)


class UltralyticsAdapter(SpecialistAdapter):
    """Wrap Ultralytics YOLO()/RTDETR() for both YOLO11/26 and RT-DETR slugs.

    The `YOLO` class in modern Ultralytics dispatches internally between YOLO
    and RT-DETR based on the weight filename, so we can use a single import
    for all Ultralytics-family architectures.
    """

    def __init__(self, arch: str):
        self.arch = arch

    # ── Train ────────────────────────────────────────────────────────────────

    def train(
        self,
        condition: str,
        train_args: dict,
        output_dir: Path,
        weight: str,
        *,
        resume: bool = False,
        device: str = "0",
    ) -> Path:
        from ultralytics import YOLO  # lazy

        run_name = f"specialist_{condition}"
        project_dir = output_dir.parent  # adapter caller passes the run dir
        project_dir.mkdir(parents=True, exist_ok=True)

        data_yaml = ROOT / "data" / "processed" / condition / "dataset.yaml"

        if resume:
            last_ckpt = output_dir / "weights" / "last.pt"
            if not last_ckpt.exists():
                resume = False
        model = YOLO(str(output_dir / "weights" / "last.pt") if resume else weight)

        kwargs = {k: train_args[k] for k in _TRAIN_KEYS if k in train_args}
        model.train(
            data=str(data_yaml),
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            resume=resume,
            device=device,
            save=True,
            save_period=10,
            plots=True,
            val=True,
            verbose=False,
            **kwargs,
        )
        return output_dir / "weights" / "best.pt"

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict_batch(
        self,
        ckpt: Path,
        img_paths: list[Path],
        *,
        conf_min: float = 0.001,
        imgsz: int = 640,
        batch: int = 16,
        device: str = "0",
    ) -> list[Prediction]:
        from ultralytics import YOLO  # lazy

        model = YOLO(str(ckpt))
        results = model.predict(
            source=[str(p) for p in img_paths],
            conf=conf_min,
            iou=0.50,
            imgsz=imgsz,
            device=device,
            batch=batch,
            verbose=False,
            stream=False,
        )
        out: list[Prediction] = []
        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                out.append(Prediction(
                    boxes_xyxyn=np.zeros((0, 4), dtype=np.float32),
                    scores=np.zeros((0,), dtype=np.float32),
                    labels=np.zeros((0,), dtype=np.int64),
                ))
                continue
            # Ultralytics offers xyxyn (normalized xyxy).
            xyxyn = res.boxes.xyxyn.cpu().numpy().astype(np.float32)
            conf = res.boxes.conf.cpu().numpy().astype(np.float32)
            cls = res.boxes.cls.cpu().numpy().astype(np.int64)
            out.append(Prediction(boxes_xyxyn=xyxyn, scores=conf, labels=cls))
        return out

    # ── mAP@0.5 ──────────────────────────────────────────────────────────────

    def compute_map50(
        self,
        ckpt: Path,
        data_yaml: Path,
        split: str,
        *,
        imgsz: int = 640,
        batch: int = 16,
        device: str = "0",
    ) -> float:
        from ultralytics import YOLO  # lazy
        model = YOLO(str(ckpt))
        results = model.val(
            data=str(data_yaml),
            split=split,
            device=device,
            imgsz=imgsz,
            batch=batch,
            verbose=False,
        )
        return float(getattr(results.box, "map50", 0.0))

    # ── ONNX export ──────────────────────────────────────────────────────────

    def export_onnx(self, ckpt: Path, *, imgsz: int = 640) -> Path:
        from ultralytics import YOLO  # lazy
        model = YOLO(str(ckpt))
        onnx_path = model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True)
        return Path(onnx_path)
