"""
src/inference/ensemble.py
──────────────────────────
Runs all 6 specialist models over a directory of unlabeled images and
returns raw detections per model. Called by merge.py.

Each detection is a dict:
  {
    "image":      Path,
    "condition":  str,          # "caries", "gingivitis", ...
    "class_id":   int,          # unified class id (0-5)
    "conf":       float,
    "bbox_yolo":  [cx, cy, w, h],   # normalised YOLO format
    "bbox_xyxy":  [x1, y1, x2, y2]  # absolute pixels
  }
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
PIPELINE_CFG = ROOT / "configs" / "pipeline.yaml"

UNIFIED_CLASS_IDS = {
    "caries":        0,
    "gingivitis":    1,
    "plaque":        2,
    "discoloration": 3,
    "ulcer":         4,
    "recession":     5,
}


def _load_cfg() -> dict:
    with open(PIPELINE_CFG) as f:
        return yaml.safe_load(f)


def _find_best_ckpt(condition: str) -> Path:
    p = ROOT / "runs" / "specialists" / f"specialist_{condition}" / "weights" / "best.pt"
    if not p.exists():
        raise FileNotFoundError(
            f"No trained checkpoint for [{condition}] at {p}\n"
            f"Run: python src/train/train_specialist.py --condition {condition}"
        )
    return p


def load_specialists(conditions: list[str] | None = None) -> dict[str, YOLO]:
    """Load all (or specified) specialist models into memory."""
    cfg = _load_cfg()
    targets = conditions or list(cfg["specialists"].keys())

    models = {}
    for cond in targets:
        ckpt = _find_best_ckpt(cond)
        print(f"  Loading [{cond}] from {ckpt.name}")
        models[cond] = YOLO(str(ckpt))

    return models


def run_ensemble(
    image_dir: Path,
    models: dict[str, YOLO],
    conf_thresholds: dict[str, float] | None = None,
    iou_thresholds:  dict[str, float] | None = None,
    batch_size: int = 16,
    img_exts: tuple = (".jpg", ".jpeg", ".png", ".JPG", ".PNG"),
) -> dict[str, list[dict]]:
    """
    Run all specialist models over every image in image_dir.

    Returns:
        detections_by_image: {image_stem: [detection_dict, ...]}
    """
    cfg = _load_cfg()
    specs = cfg["specialists"]

    conf_thresh = conf_thresholds or {c: specs[c]["conf_thresh"] for c in models}
    iou_thresh  = iou_thresholds  or {c: specs[c]["iou_thresh"]  for c in models}

    images = sorted([p for p in image_dir.iterdir()
                     if p.suffix in img_exts])
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    print(f"\n[Ensemble] {len(images)} images × {len(models)} models "
          f"= {len(images) * len(models)} forward passes")

    detections_by_image: dict[str, list[dict]] = {img.stem: [] for img in images}

    for condition, model in models.items():
        unified_id = UNIFIED_CLASS_IDS[condition]
        conf = conf_thresh[condition]
        iou  = iou_thresh[condition]

        t0 = time.time()
        print(f"\n  ▶ {condition} (conf≥{conf}, iou={iou})")

        # Process in batches
        for i in tqdm(range(0, len(images), batch_size),
                      desc=f"  {condition}", unit="batch"):
            batch = images[i: i + batch_size]
            batch_paths = [str(p) for p in batch]

            results = model.predict(
                source=batch_paths,
                conf=conf,
                iou=iou,
                verbose=False,
                device=cfg["project"]["device"],
            )

            for img_path, result in zip(batch, results):
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                img_h, img_w = result.orig_shape

                for j in range(len(boxes)):
                    conf_j   = float(boxes.conf[j])
                    xyxy     = boxes.xyxy[j].tolist()     # [x1,y1,x2,y2] abs
                    xywhn    = boxes.xywhn[j].tolist()    # [cx,cy,w,h] normalised

                    # Plaque has 3 sub-classes — collapse to single "plaque" for unified id
                    # but keep original sub-class for the specialist output
                    local_cls = int(boxes.cls[j])

                    detections_by_image[img_path.stem].append({
                        "image":      img_path,
                        "condition":  condition,
                        "class_id":   unified_id,
                        "local_cls":  local_cls,
                        "conf":       conf_j,
                        "bbox_yolo":  xywhn,     # [cx, cy, w, h] normalised
                        "bbox_xyxy":  xyxy,      # [x1, y1, x2, y2] absolute
                    })

        elapsed = time.time() - t0
        n_dets = sum(1 for v in detections_by_image.values()
                     for d in v if d["condition"] == condition)
        print(f"    {n_dets} detections in {elapsed:.1f}s")

    return detections_by_image


def ensemble_generator(
    image_dir: Path,
    models: dict[str, YOLO],
    batch_size: int = 8,
) -> Iterator[tuple[Path, list[dict]]]:
    """
    Memory-efficient generator version: yields (image_path, [detections])
    one image at a time. Useful when image_dir is very large.
    """
    cfg = _load_cfg()
    specs = cfg["specialists"]

    images = sorted([p for p in image_dir.iterdir()
                     if p.suffix in (".jpg", ".jpeg", ".png")])

    # Pre-run each model across all images, collect into a dict first
    all_dets = run_ensemble(image_dir, models, batch_size=batch_size)
    for img_path in images:
        yield img_path, all_dets.get(img_path.stem, [])
