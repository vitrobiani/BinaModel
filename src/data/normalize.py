"""
src/data/normalize.py
─────────────────────
Converts datasets from BinaDatasets/ into unified YOLO detection format
under data/processed/<condition>/.

Dataset → Condition Mapping:
  - caries:        Carries_Dataset + Carries_abrasion (classes 3-8) + extensive (class 0)
  - gingivitis:    Gingivites_Dataset (classes 3,4) + extensive (class 3)
  - plaque:        mendeley-dataset + CALCULUS_Dataset
  - discoloration: extensive (class 2)
  - ulcer:         extensive (class 1)
  - recession:     gum_recession_dataset (class 0) + Spot (class 5, polygons) + augmentation

Usage:
  python src/data/normalize.py --condition caries
  python src/data/normalize.py --condition all
"""

import argparse
import math
import random
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
BINA_DATASETS_DIR = ROOT / "BinaDatasets"
OUT_DIR = ROOT / "data" / "processed"

SPLITS = {"train": 0.75, "val": 0.15, "test": 0.10}

# Dataset paths
CARRIES_DATASET = BINA_DATASETS_DIR / "Carries_Dataset"
CARRIES_ABRASION = BINA_DATASETS_DIR / "Carries_abrasion_restoration_dataset"
GINGIVITES_DATASET = BINA_DATASETS_DIR / "Gingivites_Dataset" / "Dataset"
MENDELEY_DATASET = BINA_DATASETS_DIR / "mendeley-dataset-materials_Part_2" / "data"
CALCULUS_DATASET = BINA_DATASETS_DIR / "CALCULUS_Dataset"
GUM_RECESSION = BINA_DATASETS_DIR / "gum_recession_dataset"
SPOT_DATASET = BINA_DATASETS_DIR / "Spot"
BIG_GUM_DATASET = BINA_DATASETS_DIR / "big_gum_dataset"
EXTENSIVE_DATASET = (
    BINA_DATASETS_DIR
    / "extensive_dataset"
    / "Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset"
    / "Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset"
    / "Data"
)

# Extensive dataset class mapping (from data.yaml):
# 0: Caries, 1: Ulcer, 2: Tooth Discoloration, 3: Gingivitis
EXTENSIVE_CLASSES = {
    "caries": 0,
    "ulcer": 1,
    "discoloration": 2,
    "gingivitis": 3,
}

# Gingivites dataset uses classes 3, 4, 5 for gum conditions
# Classes 3,4 are gingivitis-related
GINGIVITES_CLASSES = {3, 4}


# ── Base Converters ─────────────────────────────────────────────────────────


def polygon_to_bbox_yolo(polygon_values: list[float]) -> tuple[float, float, float, float]:
    """
    Convert YOLO polygon format to YOLO bbox.

    YOLO polygon format: class_id x1 y1 x2 y2 x3 y3 ... (normalized)
    Returns: (cx, cy, w, h) normalized
    """
    # polygon_values contains pairs of x,y coordinates (normalized)
    xs = polygon_values[0::2]
    ys = polygon_values[1::2]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min

    return cx, cy, w, h


def convert_mendeley_line(line: str) -> Optional[str]:
    """
    Convert mendeley custom format to standard YOLO.

    Mendeley format: plaque_flag cx cy w h tooth_id
    Standard YOLO:   class_id cx cy w h

    Only keep lines where plaque_flag == 1 (has plaque)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    plaque_flag = int(parts[0])
    if plaque_flag != 1:
        return None

    # cx, cy, w, h are parts[1:5], ignore tooth_id (parts[5] if exists)
    cx, cy, w, h = parts[1], parts[2], parts[3], parts[4]
    return f"0 {cx} {cy} {w} {h}"


def copy_yolo_with_remap(
    src_label_dir: Path,
    src_img_dir: Path,
    out_label_dir: Path,
    out_img_dir: Path,
    remap: dict[int, int],
    label_glob: str = "*.txt",
    name_prefix: str = "",
) -> int:
    """
    Copy YOLO-format labels, remapping class IDs.

    Args:
        src_label_dir: Source labels directory
        src_img_dir: Source images directory
        out_label_dir: Output labels directory
        out_img_dir: Output images directory
        remap: {source_class_id: target_class_id}
        label_glob: Glob pattern for label files
        name_prefix: Prefix prepended to output filenames to avoid collisions
            when merging multiple source datasets into one pool.

    Returns:
        Number of images copied
    """
    labels = list(src_label_dir.glob(label_glob))
    copied = 0

    for lp in tqdm(labels, desc=f"  YOLO remap {src_label_dir.name}", leave=False):
        lines_in = lp.read_text().splitlines()
        lines_out = []

        for line in lines_in:
            parts = line.strip().split()
            if len(parts) < 5:  # Need class + 4 bbox values
                continue

            try:
                src_id = int(parts[0])
            except ValueError:
                continue  # Skip non-numeric class IDs (e.g., header files)

            if src_id not in remap:
                continue

            new_id = remap[src_id]
            lines_out.append(f"{new_id} " + " ".join(parts[1:5]))  # Only keep bbox (5 values total)

        if not lines_out:
            continue

        # Find and copy image
        img_stem = lp.stem
        img_copied = False
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"]:
            img_src = src_img_dir / (img_stem + ext)
            if img_src.exists():
                out_name = f"{name_prefix}{img_src.name}"
                shutil.copy(img_src, out_img_dir / out_name)
                img_copied = True
                break

        if img_copied:
            (out_label_dir / f"{name_prefix}{lp.name}").write_text("\n".join(lines_out))
            copied += 1

    return copied


def copy_polygon_to_bbox(
    src_label_dir: Path,
    src_img_dir: Path,
    out_label_dir: Path,
    out_img_dir: Path,
    remap: dict[int, int],
    name_prefix: str = "",
) -> int:
    """
    Convert YOLO polygon (segmentation) format to YOLO bbox, with class remapping.

    Polygon format: class_id x1 y1 x2 y2 x3 y3 ...
    Bbox format:    class_id cx cy w h

    name_prefix is prepended to output filenames so multiple source datasets
    can be merged into the same temp pool without collisions.
    """
    labels = list(src_label_dir.glob("*.txt"))
    copied = 0

    for lp in tqdm(labels, desc=f"  Polygon→bbox {src_label_dir.name}", leave=False):
        lines_in = lp.read_text().splitlines()
        lines_out = []

        for line in lines_in:
            parts = line.strip().split()
            if len(parts) < 5:  # Need at least class + 2 points (4 coords)
                continue

            try:
                src_id = int(parts[0])
            except ValueError:
                continue  # Skip non-numeric class IDs

            if src_id not in remap:
                continue

            new_id = remap[src_id]
            try:
                polygon_values = [float(p) for p in parts[1:]]
            except ValueError:
                continue
            cx, cy, w, h = polygon_to_bbox_yolo(polygon_values)
            lines_out.append(f"{new_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not lines_out:
            continue

        # Find and copy image
        img_stem = lp.stem
        img_copied = False
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"]:
            img_src = src_img_dir / (img_stem + ext)
            if img_src.exists():
                out_name = f"{name_prefix}{img_src.name}"
                shutil.copy(img_src, out_img_dir / out_name)
                img_copied = True
                break

        if img_copied:
            (out_label_dir / f"{name_prefix}{lp.name}").write_text("\n".join(lines_out))
            copied += 1

    return copied


def process_mendeley_dataset(
    src_dir: Path,
    out_label_dir: Path,
    out_img_dir: Path,
) -> int:
    """
    Process mendeley plaque dataset with custom format.

    Directory structure: data/labels/patientXXXX/*.txt
                        data/images/patientXXXX/*.jpg
    Format: plaque_flag cx cy w h tooth_id
    """
    labels_dir = src_dir / "labels"
    images_dir = src_dir / "images"
    copied = 0

    # Iterate through patient folders
    for patient_dir in tqdm(list(labels_dir.iterdir()), desc="  Mendeley patients", leave=False):
        if not patient_dir.is_dir():
            continue

        patient_img_dir = images_dir / patient_dir.name

        for lp in patient_dir.glob("*.txt"):
            lines_in = lp.read_text().splitlines()
            lines_out = []

            for line in lines_in:
                converted = convert_mendeley_line(line)
                if converted:
                    lines_out.append(converted)

            if not lines_out:
                continue

            # Find and copy image
            img_stem = lp.stem
            img_copied = False
            for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
                img_src = patient_img_dir / (img_stem + ext)
                if img_src.exists():
                    # Use patient_filename to avoid collisions
                    out_name = f"{patient_dir.name}_{img_stem}"
                    shutil.copy(img_src, out_img_dir / f"{out_name}{ext}")
                    (out_label_dir / f"{out_name}.txt").write_text("\n".join(lines_out))
                    copied += 1
                    img_copied = True
                    break

    return copied


# ── Split and Dataset YAML Utilities ─────────────────────────────────────────


def create_splits(
    all_img_dir: Path,
    all_lbl_dir: Path,
    out_dir: Path,
    seed: int = 42,
) -> dict[str, int]:
    """
    Split images into train/val/test.

    Args:
        all_img_dir: Directory containing all images
        all_lbl_dir: Directory containing all labels
        out_dir: Base output directory (will create images/train etc.)
        seed: Random seed for reproducibility

    Returns:
        Dict with counts per split
    """
    random.seed(seed)

    all_imgs = list(all_img_dir.glob("*.*"))
    # Filter to only image files
    all_imgs = [f for f in all_imgs if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    random.shuffle(all_imgs)

    n = len(all_imgs)
    n_train = math.floor(n * SPLITS["train"])
    n_val = math.floor(n * SPLITS["val"])

    splits_data = {
        "train": all_imgs[:n_train],
        "val": all_imgs[n_train : n_train + n_val],
        "test": all_imgs[n_train + n_val :],
    }

    counts = {}
    for split, imgs in splits_data.items():
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img in imgs:
            shutil.copy(img, img_out / img.name)
            lbl = all_lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, lbl_out / lbl.name)

        counts[split] = len(imgs)

    return counts


def preserve_splits_copy(
    src_base: Path,
    out_dir: Path,
    splits_map: dict[str, tuple[Path, Path]],
    remap: dict[int, int],
    converter: str = "yolo",
) -> dict[str, int]:
    """
    Copy data preserving existing train/val/test splits.

    Args:
        src_base: Source base directory
        out_dir: Output base directory
        splits_map: {split_name: (img_dir, lbl_dir)} relative to src_base
        remap: Class ID remapping
        converter: "yolo" for standard, "polygon" for polygon→bbox

    Returns:
        Dict with counts per split
    """
    counts = {}

    for split, (img_rel, lbl_rel) in splits_map.items():
        src_img = src_base / img_rel
        src_lbl = src_base / lbl_rel

        if not src_img.exists():
            counts[split] = 0
            continue

        out_img = out_dir / "images" / split
        out_lbl = out_dir / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        if converter == "polygon":
            counts[split] = copy_polygon_to_bbox(src_lbl, src_img, out_lbl, out_img, remap)
        else:
            counts[split] = copy_yolo_with_remap(src_lbl, src_img, out_lbl, out_img, remap)

    return counts


def write_dataset_yaml(out_dir: Path, condition: str) -> None:
    """Generate YOLO dataset.yaml config file."""
    yaml_content = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": [condition],
    }

    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)


def augment_small_dataset(
    img_dir: Path,
    lbl_dir: Path,
    target_count: int = 300,
    seed: int = 42,
) -> int:
    """
    Augment a small dataset with geometric and color transforms.

    Creates augmented copies until target_count is reached.
    Returns number of augmented images created.
    """
    random.seed(seed)
    np.random.seed(seed)

    imgs = list(img_dir.glob("*.*"))
    imgs = [f for f in imgs if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    original_count = len(imgs)

    if original_count >= target_count:
        return 0

    augments_needed = target_count - original_count
    augmented = 0

    # Augmentation functions
    def flip_horizontal(img, boxes):
        flipped = cv2.flip(img, 1)
        new_boxes = []
        for box in boxes:
            cls, cx, cy, w, h = box
            new_boxes.append((cls, 1.0 - cx, cy, w, h))
        return flipped, new_boxes

    def adjust_brightness(img, factor):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def adjust_saturation(img, factor):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def rotate_small(img, boxes, angle):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        # For small angles, bbox adjustment is minimal - keep original
        return rotated, boxes

    augment_ops = [
        ("flip", lambda img, boxes: flip_horizontal(img, boxes)),
        ("bright_up", lambda img, boxes: (adjust_brightness(img, 1.2), boxes)),
        ("bright_down", lambda img, boxes: (adjust_brightness(img, 0.8), boxes)),
        ("sat_up", lambda img, boxes: (adjust_saturation(img, 1.3), boxes)),
        ("sat_down", lambda img, boxes: (adjust_saturation(img, 0.7), boxes)),
        ("rotate_5", lambda img, boxes: rotate_small(img, boxes, 5)),
        ("rotate_-5", lambda img, boxes: rotate_small(img, boxes, -5)),
    ]

    for img_path in tqdm(imgs, desc="  Augmenting recession", leave=False):
        if augmented >= augments_needed:
            break

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Load boxes
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        boxes = []
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append(tuple(map(float, parts[:5])))

        # Apply each augmentation
        for aug_name, aug_fn in augment_ops:
            if augmented >= augments_needed:
                break

            aug_img, aug_boxes = aug_fn(img, boxes)

            # Save augmented image and label
            out_stem = f"{img_path.stem}_aug_{aug_name}"
            out_img_path = img_dir / f"{out_stem}{img_path.suffix}"
            out_lbl_path = lbl_dir / f"{out_stem}.txt"

            cv2.imwrite(str(out_img_path), aug_img)

            lines = [f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in aug_boxes]
            out_lbl_path.write_text("\n".join(lines))

            augmented += 1

    return augmented


# ── Per-Condition Normalizers ────────────────────────────────────────────────


def normalize_caries(domain_shift: bool = False) -> None:  # noqa: ARG001
    """
    Normalize caries datasets:
    - Carries_Dataset (classes 0,1 → 0)
    - Carries_abrasion (polygon, classes 3-8 → 0)
    - extensive (class 0 → 0)
    """
    condition = "caries"
    out = OUT_DIR / condition
    shutil.rmtree(out, ignore_errors=True)

    # Temporary "all" directory for combining sources before split
    img_all = out / "images" / "all"
    lbl_all = out / "labels" / "all"
    img_all.mkdir(parents=True, exist_ok=True)
    lbl_all.mkdir(parents=True, exist_ok=True)

    total = 0

    # Source 1: Carries_Dataset (has existing train/val/test splits)
    print("  Processing Carries_Dataset...")
    if CARRIES_DATASET.exists():
        for split in ["train", "valid", "test"]:
            src_img = CARRIES_DATASET / split / "images"
            src_lbl = CARRIES_DATASET / split / "yolo"
            if src_img.exists():
                n = copy_yolo_with_remap(src_lbl, src_img, lbl_all, img_all, remap={0: 0, 1: 0})
                total += n
                print(f"    {split}: {n} images")

    # Source 2: Carries_abrasion (polygon format, classes 3-8 = caries)
    print("  Processing Carries_abrasion (polygon→bbox)...")
    if CARRIES_ABRASION.exists():
        caries_classes = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        for split in ["train", "valid"]:
            src_img = CARRIES_ABRASION / "images" / split
            src_lbl = CARRIES_ABRASION / "labels" / split
            if src_img.exists():
                n = copy_polygon_to_bbox(src_lbl, src_img, lbl_all, img_all, remap=caries_classes)
                total += n
                print(f"    {split}: {n} images")

    # Source 3: extensive dataset (class 0 = caries)
    print("  Processing extensive dataset (caries)...")
    if EXTENSIVE_DATASET.exists():
        for split in ["train", "val"]:
            src_img = EXTENSIVE_DATASET / "images" / split
            src_lbl = EXTENSIVE_DATASET / "labels" / split
            if src_img.exists():
                n = copy_yolo_with_remap(src_lbl, src_img, lbl_all, img_all, remap={0: 0})
                total += n
                print(f"    {split}: {n} images")

    # Create splits from combined data
    print("  Creating train/val/test splits...")
    counts = create_splits(img_all, lbl_all, out)
    print(f"    train={counts['train']} val={counts['val']} test={counts['test']}")

    # Clean up temp directories
    shutil.rmtree(img_all, ignore_errors=True)
    shutil.rmtree(lbl_all, ignore_errors=True)

    write_dataset_yaml(out, condition)
    print(f"  Total: {total} images → {sum(counts.values())} after dedup")


def normalize_gingivitis(domain_shift: bool = False) -> None:  # noqa: ARG001
    """
    Normalize gingivitis datasets:
    - Gingivites_Dataset (classes 3,4 → 0)
    - extensive (class 3 → 0)
    """
    condition = "gingivitis"
    out = OUT_DIR / condition
    shutil.rmtree(out, ignore_errors=True)

    img_all = out / "images" / "all"
    lbl_all = out / "labels" / "all"
    img_all.mkdir(parents=True, exist_ok=True)
    lbl_all.mkdir(parents=True, exist_ok=True)

    total = 0

    # Source 1: Gingivites_Dataset (classes 3,4 → 0)
    print("  Processing Gingivites_Dataset...")
    if GINGIVITES_DATASET.exists():
        gingivitis_remap = {3: 0, 4: 0}
        for split_name, folder in [("train", "Training"), ("val", "Validation"), ("test", "Test")]:
            src_img = GINGIVITES_DATASET / folder / "Images"
            src_lbl = GINGIVITES_DATASET / folder / "Labels"
            if src_img.exists():
                n = copy_yolo_with_remap(src_lbl, src_img, lbl_all, img_all, remap=gingivitis_remap)
                total += n
                print(f"    {split_name}: {n} images")

    # Source 2: extensive dataset (class 3 = gingivitis)
    print("  Processing extensive dataset (gingivitis)...")
    if EXTENSIVE_DATASET.exists():
        for split in ["train", "val"]:
            src_img = EXTENSIVE_DATASET / "images" / split
            src_lbl = EXTENSIVE_DATASET / "labels" / split
            if src_img.exists():
                n = copy_yolo_with_remap(src_lbl, src_img, lbl_all, img_all, remap={3: 0})
                total += n
                print(f"    {split}: {n} images")

    # Create splits
    print("  Creating train/val/test splits...")
    counts = create_splits(img_all, lbl_all, out)
    print(f"    train={counts['train']} val={counts['val']} test={counts['test']}")

    shutil.rmtree(img_all, ignore_errors=True)
    shutil.rmtree(lbl_all, ignore_errors=True)

    write_dataset_yaml(out, condition)
    print(f"  Total: {total} images → {sum(counts.values())} after dedup")


def normalize_plaque(domain_shift: bool = False) -> None:  # noqa: ARG001
    """
    Normalize plaque datasets:
    - mendeley-dataset (custom format: plaque_flag=1 → class 0)
    - CALCULUS_Dataset (class 0 → 0)
    """
    condition = "plaque"
    out = OUT_DIR / condition
    shutil.rmtree(out, ignore_errors=True)

    img_all = out / "images" / "all"
    lbl_all = out / "labels" / "all"
    img_all.mkdir(parents=True, exist_ok=True)
    lbl_all.mkdir(parents=True, exist_ok=True)

    total = 0

    # Source 1: mendeley-dataset (custom plaque format)
    print("  Processing mendeley-dataset...")
    if MENDELEY_DATASET.exists():
        n = process_mendeley_dataset(MENDELEY_DATASET, lbl_all, img_all)
        total += n
        print(f"    Processed: {n} images with plaque")

    # Source 2: CALCULUS_Dataset
    print("  Processing CALCULUS_Dataset...")
    if CALCULUS_DATASET.exists():
        src_img = CALCULUS_DATASET / "train" / "images"
        src_lbl = CALCULUS_DATASET / "train" / "labels"
        if src_img.exists():
            n = copy_yolo_with_remap(src_lbl, src_img, lbl_all, img_all, remap={0: 0})
            total += n
            print(f"    train: {n} images")

    # Create splits
    print("  Creating train/val/test splits...")
    counts = create_splits(img_all, lbl_all, out)
    print(f"    train={counts['train']} val={counts['val']} test={counts['test']}")

    shutil.rmtree(img_all, ignore_errors=True)
    shutil.rmtree(lbl_all, ignore_errors=True)

    write_dataset_yaml(out, condition)
    print(f"  Total: {total} images → {sum(counts.values())} after dedup")


def normalize_discoloration(domain_shift: bool = False) -> None:  # noqa: ARG001
    """
    Normalize discoloration datasets:
    - extensive (class 2 → 0)
    """
    condition = "discoloration"
    out = OUT_DIR / condition
    shutil.rmtree(out, ignore_errors=True)

    img_all = out / "images" / "all"
    lbl_all = out / "labels" / "all"
    img_all.mkdir(parents=True, exist_ok=True)
    lbl_all.mkdir(parents=True, exist_ok=True)

    total = 0

    # Source: extensive dataset (class 2 = tooth discoloration)
    print("  Processing extensive dataset (discoloration)...")
    if EXTENSIVE_DATASET.exists():
        for split in ["train", "val"]:
            src_img = EXTENSIVE_DATASET / "images" / split
            src_lbl = EXTENSIVE_DATASET / "labels" / split
            if src_img.exists():
                n = copy_yolo_with_remap(src_lbl, src_img, lbl_all, img_all, remap={2: 0})
                total += n
                print(f"    {split}: {n} images")

    # Create splits
    print("  Creating train/val/test splits...")
    counts = create_splits(img_all, lbl_all, out)
    print(f"    train={counts['train']} val={counts['val']} test={counts['test']}")

    shutil.rmtree(img_all, ignore_errors=True)
    shutil.rmtree(lbl_all, ignore_errors=True)

    write_dataset_yaml(out, condition)
    print(f"  Total: {total} images → {sum(counts.values())} after dedup")


def normalize_ulcer(domain_shift: bool = False) -> None:  # noqa: ARG001
    """
    Normalize ulcer datasets:
    - extensive (class 1 → 0)
    """
    condition = "ulcer"
    out = OUT_DIR / condition
    shutil.rmtree(out, ignore_errors=True)

    img_all = out / "images" / "all"
    lbl_all = out / "labels" / "all"
    img_all.mkdir(parents=True, exist_ok=True)
    lbl_all.mkdir(parents=True, exist_ok=True)

    total = 0

    # Source: extensive dataset (class 1 = ulcer)
    print("  Processing extensive dataset (ulcer)...")
    if EXTENSIVE_DATASET.exists():
        for split in ["train", "val"]:
            src_img = EXTENSIVE_DATASET / "images" / split
            src_lbl = EXTENSIVE_DATASET / "labels" / split
            if src_img.exists():
                n = copy_yolo_with_remap(src_lbl, src_img, lbl_all, img_all, remap={1: 0})
                total += n
                print(f"    {split}: {n} images")

    # Create splits
    print("  Creating train/val/test splits...")
    counts = create_splits(img_all, lbl_all, out)
    print(f"    train={counts['train']} val={counts['val']} test={counts['test']}")

    shutil.rmtree(img_all, ignore_errors=True)
    shutil.rmtree(lbl_all, ignore_errors=True)

    write_dataset_yaml(out, condition)
    print(f"  Total: {total} images → {sum(counts.values())} after dedup")


def normalize_recession(domain_shift: bool = False) -> None:  # noqa: ARG001
    """
    Normalize recession datasets:
    - gum_recession_dataset (class 0 → 0, ignore classes 1,2)
    - Spot dataset (polygon, class 5 "Caries 5 class" → 0; other classes dropped)
    - big_gum_dataset (bbox, class 1 "receding_gum" → 0; class 0 "diseased_gum" dropped)
    - Apply heavy augmentation if train set is still small
    """
    condition = "recession"
    out = OUT_DIR / condition
    shutil.rmtree(out, ignore_errors=True)

    # For recession, we'll create splits first, then augment train set
    img_all = out / "images" / "all"
    lbl_all = out / "labels" / "all"
    img_all.mkdir(parents=True, exist_ok=True)
    lbl_all.mkdir(parents=True, exist_ok=True)

    total = 0

    # Source 1: gum_recession_dataset (class 0 = gum recession)
    print("  Processing gum_recession_dataset...")
    if GUM_RECESSION.exists():
        src_img = GUM_RECESSION / "train" / "images"
        src_lbl = GUM_RECESSION / "train" / "labels"
        if src_img.exists():
            n = copy_yolo_with_remap(src_lbl, src_img, lbl_all, img_all, remap={0: 0})
            total += n
            print(f"    train: {n} images")

    # Source 2: Spot dataset (polygon segmentation, class 5 = "Caries 5 class")
    # Spot annotates recession as "Caries 5 class". We keep only class 5 lines
    # (images with no class 5 are dropped automatically by copy_polygon_to_bbox
    # when lines_out is empty) and remap to 0 to match the recession specialist.
    print("  Processing Spot dataset (class 5 → recession)...")
    if SPOT_DATASET.exists():
        for split in ["train", "valid", "test"]:
            src_img = SPOT_DATASET / split / "images"
            src_lbl = SPOT_DATASET / split / "labels"
            if src_img.exists():
                n = copy_polygon_to_bbox(
                    src_lbl, src_img, lbl_all, img_all,
                    remap={5: 0}, name_prefix="spot_",
                )
                total += n
                print(f"    {split}: {n} images with class 5")

    # Source 3: big_gum_dataset (bbox, class 1 = "receding_gum")
    # data.yaml: names: ['diseased_gum', 'receding_gum'] → only class 1 is recession.
    # Class 0 (diseased_gum) is dropped per-line; images with no class 1 are
    # skipped automatically when lines_out is empty.
    print("  Processing big_gum_dataset (class 1 receding_gum → recession)...")
    if BIG_GUM_DATASET.exists():
        for split in ["train", "valid", "test"]:
            src_img = BIG_GUM_DATASET / split / "images"
            src_lbl = BIG_GUM_DATASET / split / "labels"
            if src_img.exists():
                n = copy_yolo_with_remap(
                    src_lbl, src_img, lbl_all, img_all,
                    remap={1: 0}, name_prefix="big_gum_",
                )
                total += n
                print(f"    {split}: {n} images with receding_gum")

    # Create splits
    print("  Creating train/val/test splits...")
    counts = create_splits(img_all, lbl_all, out)
    print(f"    train={counts['train']} val={counts['val']} test={counts['test']}")

    shutil.rmtree(img_all, ignore_errors=True)
    shutil.rmtree(lbl_all, ignore_errors=True)

    # Augment train set
    print("  Augmenting training set (target: 300 images)...")
    train_img = out / "images" / "train"
    train_lbl = out / "labels" / "train"
    augmented = augment_small_dataset(train_img, train_lbl, target_count=300)
    print(f"    Created {augmented} augmented images")

    write_dataset_yaml(out, condition)
    final_train = len(list(train_img.glob("*.*")))
    print(f"  Total: {total} images → train={final_train} after augmentation")


# ── Entry Point ──────────────────────────────────────────────────────────────

NORMALIZERS = {
    "caries": normalize_caries,
    "gingivitis": normalize_gingivitis,
    "plaque": normalize_plaque,
    "discoloration": normalize_discoloration,
    "ulcer": normalize_ulcer,
    "recession": normalize_recession,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize BinaDatasets to unified YOLO format"
    )
    parser.add_argument(
        "--condition",
        default="all",
        choices=list(NORMALIZERS.keys()) + ["all"],
        help="Condition to normalize (default: all)",
    )
    parser.add_argument(
        "--domain-shift",
        action="store_true",
        help="Apply domain shift augmentation (not implemented)",
    )
    args = parser.parse_args()

    conditions = list(NORMALIZERS.keys()) if args.condition == "all" else [args.condition]

    for cond in conditions:
        print(f"\n{'=' * 60}")
        print(f"  Normalizing: {cond}")
        print(f"{'=' * 60}")
        NORMALIZERS[cond](domain_shift=args.domain_shift)

    print(f"\n{'=' * 60}")
    print("  Normalization complete!")
    print(f"  Output: {OUT_DIR}")
    print(f"{'=' * 60}")
