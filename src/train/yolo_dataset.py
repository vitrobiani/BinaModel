"""
src/train/yolo_dataset.py
─────────────────────────
PyTorch Dataset for the YOLO-format directories produced by normalize.py.

Used by the non-Ultralytics adapters (torchvision Faster R-CNN, HF DETR).
Returns per-image data in the shape each model family expects:

  fmt="frcnn"  → (image_tensor [3,H,W] float in [0,1],
                  {"boxes": Tensor[N,4]=xyxy pixel,
                   "labels": Tensor[N]=int64 (always 1 for single-class)})

  fmt="detr"   → (PIL.Image,
                  [{"image_id": int, "category_id": 0,
                    "bbox": [x, y, w, h] pixels,         (COCO-style)
                    "area": float, "iscrowd": 0}, ...])
                  The HuggingFace DetrImageProcessor consumes this format.

Both flavors are single-class. Background is implicit:
  - torchvision Faster R-CNN: label `0` is background, target label `1` = object.
  - HF DETR: `num_labels` excludes background, so target `category_id` is 0.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]


# ── Label readers ────────────────────────────────────────────────────────────


def _read_yolo_label(lbl_path: Path) -> np.ndarray:
    """Return Nx4 array of normalized [cx, cy, w, h]. Skips invalid lines.
    Class id is implicit (single-class): always 0 after normalize.py filtering."""
    if not lbl_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            rows.append([float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4])])
        except ValueError:
            continue
    return np.asarray(rows, dtype=np.float32) if rows else \
        np.zeros((0, 4), dtype=np.float32)


def _cxcywh_n_to_xyxy_pixel(boxes_n: np.ndarray, w: int, h: int) -> np.ndarray:
    """[cx,cy,bw,bh] normalized → [x1,y1,x2,y2] pixel."""
    if len(boxes_n) == 0:
        return boxes_n
    cx = boxes_n[:, 0] * w
    cy = boxes_n[:, 1] * h
    bw = boxes_n[:, 2] * w
    bh = boxes_n[:, 3] * h
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def _cxcywh_n_to_xywh_pixel(boxes_n: np.ndarray, w: int, h: int) -> np.ndarray:
    """[cx,cy,bw,bh] normalized → [x, y, w, h] pixel (COCO box format)."""
    if len(boxes_n) == 0:
        return boxes_n
    cx = boxes_n[:, 0] * w
    cy = boxes_n[:, 1] * h
    bw = boxes_n[:, 2] * w
    bh = boxes_n[:, 3] * h
    x = cx - bw / 2
    y = cy - bh / 2
    return np.stack([x, y, bw, bh], axis=1).astype(np.float32)


# ── Dataset ──────────────────────────────────────────────────────────────────


class YoloDirDataset(Dataset):
    """One YOLO-format split (train/val/test) for a single condition.

    Args:
      condition: e.g. "caries" (must match a dir under data/processed/).
      split: "train" | "val" | "test".
      fmt: "frcnn" or "detr" — controls __getitem__ output shape.
      root: project root (default: auto-detected).
    """

    def __init__(
        self,
        condition: str,
        split: str,
        fmt: str,
        *,
        root: Path | None = None,
    ):
        if fmt not in ("frcnn", "detr"):
            raise ValueError(f"fmt must be 'frcnn' or 'detr', got {fmt!r}")
        self.fmt = fmt
        base = (root or ROOT) / "data" / "processed" / condition
        self.img_dir = base / "images" / split
        self.lbl_dir = base / "labels" / split
        if not self.img_dir.exists():
            raise FileNotFoundError(f"No images dir at {self.img_dir}")
        self.imgs = sorted(
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img_path = self.imgs[idx]
        # Read image with cv2 → RGB
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to read {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        boxes_n = _read_yolo_label(self.lbl_dir / f"{img_path.stem}.txt")

        if self.fmt == "frcnn":
            return self._format_frcnn(rgb, boxes_n, h, w, idx)
        return self._format_detr(rgb, boxes_n, h, w, idx)

    # ── format-specific ──

    def _format_frcnn(self, rgb: np.ndarray, boxes_n: np.ndarray,
                      h: int, w: int, idx: int):
        # torchvision detection models expect image as FloatTensor [3,H,W] in [0,1].
        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        boxes_xyxy = _cxcywh_n_to_xyxy_pixel(boxes_n, w, h)
        # Ensure non-degenerate boxes (FRCNN rejects zero-area)
        if len(boxes_xyxy):
            keep = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & \
                   (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
            boxes_xyxy = boxes_xyxy[keep]
        n = len(boxes_xyxy)
        target = {
            "boxes": torch.as_tensor(boxes_xyxy, dtype=torch.float32),
            # label 0 is background in torchvision; foreground = 1.
            "labels": torch.ones(n, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return img, target

    def _format_detr(self, rgb: np.ndarray, boxes_n: np.ndarray,
                     h: int, w: int, idx: int):
        # HuggingFace DetrImageProcessor consumes PIL images + COCO-format
        # annotation dicts (bbox = [x, y, w, h] in pixels).
        from PIL import Image
        pil = Image.fromarray(rgb)
        xywh = _cxcywh_n_to_xywh_pixel(boxes_n, w, h)
        annotations = []
        for i, b in enumerate(xywh):
            annotations.append({
                "image_id": idx,
                "category_id": 0,        # single-class
                "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "area": float(b[2] * b[3]),
                "iscrowd": 0,
                "id": idx * 1000 + i,
            })
        return pil, {"image_id": idx, "annotations": annotations}


# ── Collate functions ───────────────────────────────────────────────────────


def collate_frcnn(batch):
    """torchvision detection models take (list[Tensor], list[dict])."""
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


def collate_detr_pre_processor(batch):
    """Pre-processor stage: just zip. The adapter applies DetrImageProcessor
    to (images, annotations) before passing to the model."""
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)
