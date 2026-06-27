"""
src/inference/merge.py
───────────────────────
Takes the raw ensemble detections and produces clean, merged YOLO annotations
suitable for training the final student model.

Key responsibilities:
  1. Per-model confidence filtering (already done at inference time)
  2. Cross-model deduplication: if two DIFFERENT conditions produce
     overlapping boxes on the same region, keep the higher-confidence one
  3. Intra-class NMS: deduplicate within the same condition
  4. Second-pass global confidence filter (student.pseudo_label_min_conf)
  5. Write YOLO .txt label files

The result is a dataset at data/pseudo_labeled/ with the same structure
as data/processed/<condition>/ — ready to be combined with real data.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
PIPELINE_CFG = ROOT / "configs" / "pipeline.yaml"


# ── IoU utilities ────────────────────────────────────────────────────────────

def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    """
    Compute IoU between two boxes in XYXY format [x1, y1, x2, y2].
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    """
    Standard NMS within a single class.
    detections: list of detection dicts with "conf" and "bbox_xyxy" keys.
    Returns the filtered list.
    """
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d["conf"], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets
                if bbox_iou(best["bbox_xyxy"], d["bbox_xyxy"]) < iou_threshold]
    return keep


def cross_model_dedup(detections: list[dict],
                      cross_iou_threshold: float) -> list[dict]:
    """
    Among detections from DIFFERENT conditions, if two boxes overlap
    above cross_iou_threshold, keep only the higher-confidence one.

    This prevents the same tooth region being annotated as both
    "discoloration" and "plaque", for example.

    Within the same condition, this is NOT applied (intra-class NMS handles that).
    """
    if len(detections) <= 1:
        return detections

    # Sort by confidence descending
    dets = sorted(detections, key=lambda d: d["conf"], reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        survivors = []
        for d in dets:
            # Only suppress if different condition
            if d["condition"] != best["condition"]:
                iou = bbox_iou(best["bbox_xyxy"], d["bbox_xyxy"])
                if iou >= cross_iou_threshold:
                    continue  # suppress this one
            survivors.append(d)
        dets = survivors

    return keep


# ── Main merging pipeline ────────────────────────────────────────────────────

class MergeStats(NamedTuple):
    total_raw:         int
    after_intra_nms:   int
    after_cross_dedup: int
    after_conf_filter: int
    images_with_labels: int
    images_empty:       int


def merge_detections(
    detections_by_image: dict[str, list[dict]],
    output_dir: Path,
    image_src_dir: Path,
    pseudo_label_min_conf: float = 0.45,
    cross_iou_threshold: float   = 0.60,
    intra_nms_iou: float         = 0.45,
    copy_images: bool            = True,
) -> MergeStats:
    """
    Full merge pipeline for pseudo-labeled data.

    Args:
        detections_by_image: {img_stem: [detection_dict,...]}
        output_dir:          where to write merged labels + images
        image_src_dir:       source of original images (for copying)
        pseudo_label_min_conf: second-pass confidence floor
        cross_iou_threshold: IoU above which cross-condition boxes are deduped
        intra_nms_iou:       IoU threshold for within-class NMS
        copy_images:         also copy image files to output_dir

    Output structure:
        output_dir/
          images/   ← image files
          labels/   ← YOLO .txt files (one line per detection)
    """
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    stats = dict(total_raw=0, after_intra=0, after_cross=0,
                 after_conf=0, with_labels=0, empty=0)

    for img_stem, dets in tqdm(detections_by_image.items(),
                                desc="Merging annotations"):
        if not dets:
            stats["empty"] += 1
            continue

        stats["total_raw"] += len(dets)

        # ── Step 1: Intra-class NMS per condition ────────────────────────
        by_condition: dict[str, list[dict]] = defaultdict(list)
        for d in dets:
            by_condition[d["condition"]].append(d)

        after_intra: list[dict] = []
        for cond_dets in by_condition.values():
            after_intra.extend(nms(cond_dets, iou_threshold=intra_nms_iou))

        stats["after_intra"] += len(after_intra)

        # ── Step 2: Cross-model deduplication ───────────────────────────
        after_cross = cross_model_dedup(after_intra, cross_iou_threshold)
        stats["after_cross"] += len(after_cross)

        # ── Step 3: Global confidence filter ────────────────────────────
        final = [d for d in after_cross if d["conf"] >= pseudo_label_min_conf]
        stats["after_conf"] += len(final)

        if not final:
            stats["empty"] += 1
            continue

        # ── Step 4: Write YOLO label file ────────────────────────────────
        label_path = output_dir / "labels" / (img_stem + ".txt")
        lines = []
        for d in final:
            cx, cy, w, h = d["bbox_yolo"]
            cls = d["class_id"]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        label_path.write_text("\n".join(lines))
        stats["with_labels"] += 1

        # ── Step 5: Copy image ───────────────────────────────────────────
        if copy_images:
            img_path: Path = dets[0]["image"]
            dst = output_dir / "images" / img_path.name
            if not dst.exists():
                shutil.copy(img_path, dst)

    return MergeStats(
        total_raw=stats["total_raw"],
        after_intra_nms=stats["after_intra"],
        after_cross_dedup=stats["after_cross"],
        after_conf_filter=stats["after_conf"],
        images_with_labels=stats["with_labels"],
        images_empty=stats["empty"],
    )


def write_student_dataset_yaml(pseudo_dir: Path, real_dirs: list[Path]):
    """
    Writes a dataset.yaml for the student model that combines:
      - all pseudo-labeled images
      - all original labeled images (real data, oversampled)

    Uses Ultralytics' multi-dataset format where you list multiple paths.
    """
    cfg_path = pseudo_dir.parent / "student_dataset.yaml"

    # Collect all real data train paths
    real_train = [str(d / "images" / "train") for d in real_dirs if d.exists()]
    real_val   = [str(d / "images" / "val")   for d in real_dirs if d.exists()]

    content = {
        "path": str(pseudo_dir.parent),
        # Ultralytics supports a list of train dirs
        "train": real_train + [str(pseudo_dir / "images")],
        "val":   real_val,
        "nc": 6,
        "names": ["caries", "gingivitis", "plaque",
                  "discoloration", "ulcer", "recession"],
    }

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False)

    print(f"✓ Student dataset YAML written to {cfg_path}")
    return cfg_path


def print_merge_report(stats: MergeStats):
    print(f"""
╔══════════════════════════════════════════╗
║         Merge Pipeline Report            ║
╠══════════════════════════════════════════╣
║  Raw detections (pre-merge) : {stats.total_raw:>8} ║
║  After intra-class NMS      : {stats.after_intra_nms:>8} ║
║  After cross-model dedup    : {stats.after_cross_dedup:>8} ║
║  After global conf filter   : {stats.after_conf_filter:>8} ║
╠══════════════════════════════════════════╣
║  Images with labels         : {stats.images_with_labels:>8} ║
║  Images (no detections)     : {stats.images_empty:>8} ║
╚══════════════════════════════════════════╝""")

    retention = (stats.after_conf_filter / max(stats.total_raw, 1)) * 100
    print(f"  Label retention rate: {retention:.1f}%")
    if retention < 40:
        print("  ⚠  Low retention — consider lowering conf thresholds in pipeline.yaml")
    elif retention > 85:
        print("  ⚠  Very high retention — consider raising conf thresholds to improve quality")
    else:
        print("  ✓  Retention looks healthy")
