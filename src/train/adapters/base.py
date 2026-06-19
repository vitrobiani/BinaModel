"""SpecialistAdapter ABC + Prediction value type."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Prediction:
    """One image's predictions for a single-class specialist.

    Fields are returned in a uniform shape regardless of model family, so
    threshold_finder and kpi_gate can run the same math on YOLO, FRCNN, DETR.

    boxes_xyxyn: (N, 4) — normalized [x1, y1, x2, y2] in [0, 1].
    scores:     (N,)    — confidence in [0, 1], sorted descending is fine.
    labels:     (N,)    — class id, always 0 for single-class specialists.
    """
    boxes_xyxyn: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


class SpecialistAdapter(ABC):
    """Common interface across Ultralytics, torchvision, HuggingFace.

    Subclasses cover:
      - UltralyticsAdapter   (YOLO11/26, RT-DETR)
      - FasterRCNNAdapter    (torchvision Faster R-CNN ResNet-50-FPN)
      - DETRAdapter          (HuggingFace DETR-R50)

    All adapters consume the same on-disk YOLO format
    (data/processed/<cond>/{images,labels}/{train,val,test}/) and write into
    the canonical sweep layout (runs/sweep/<arch>/specialist_<cond>/).
    """

    arch: str  # canonical slug; set by subclass

    @abstractmethod
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
        """Train one specialist.

        Args:
          condition: dental-condition slug (caries|gingivitis|...).
          train_args: merged hyperparameters (pipeline → per-model → HPO → CLI).
          output_dir: target dir; weights/best.pt goes here.
          weight: starting weight identifier (e.g. "yolo26s.pt", "frcnn-r50",
            "facebook/detr-resnet-50"). Adapter interprets it.
          resume: resume from last.pt if present.
          device: device string ("0", "cpu", "cuda:0").

        Returns the path to weights/best.pt.
        """

    @abstractmethod
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
        """Run inference at very low conf and return per-image predictions
        with boxes in normalized xyxy. Same order as `img_paths`."""

    @abstractmethod
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
        """Compute mAP@0.5 on the given split of the dataset.yaml."""

    @abstractmethod
    def export_onnx(self, ckpt: Path, *, imgsz: int = 640) -> Path:
        """Export the checkpoint to ONNX; return the .onnx path."""
