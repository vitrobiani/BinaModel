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
from collections import defaultdict
from dataclasses import dataclass, field
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

# 70/15/15 stratified, per Generic_Traning_Plan §1.4.
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}

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
    source_name: str = "",
    sources_map: Optional[dict[str, str]] = None,
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
        source_name: Logical name of the source dataset, recorded into
            sources_map for stratified splitting downstream.
        sources_map: If provided, this dict is populated as
            {output_filename: source_name} for every image copied.

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
        copied_name = ""
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"]:
            img_src = src_img_dir / (img_stem + ext)
            if img_src.exists():
                copied_name = f"{name_prefix}{img_src.name}"
                shutil.copy(img_src, out_img_dir / copied_name)
                img_copied = True
                break

        if img_copied:
            (out_label_dir / f"{name_prefix}{lp.name}").write_text("\n".join(lines_out))
            if sources_map is not None and source_name:
                sources_map[copied_name] = source_name
            copied += 1

    return copied


def copy_polygon_to_bbox(
    src_label_dir: Path,
    src_img_dir: Path,
    out_label_dir: Path,
    out_img_dir: Path,
    remap: dict[int, int],
    name_prefix: str = "",
    source_name: str = "",
    sources_map: Optional[dict[str, str]] = None,
) -> int:
    """
    Convert YOLO polygon (segmentation) format to YOLO bbox, with class remapping.

    Polygon format: class_id x1 y1 x2 y2 x3 y3 ...
    Bbox format:    class_id cx cy w h

    name_prefix is prepended to output filenames so multiple source datasets
    can be merged into the same temp pool without collisions.

    sources_map (if provided) is populated with {output_filename: source_name}
    so the downstream splitter can stratify by source dataset.
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
        copied_name = ""
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"]:
            img_src = src_img_dir / (img_stem + ext)
            if img_src.exists():
                copied_name = f"{name_prefix}{img_src.name}"
                shutil.copy(img_src, out_img_dir / copied_name)
                img_copied = True
                break

        if img_copied:
            (out_label_dir / f"{name_prefix}{lp.name}").write_text("\n".join(lines_out))
            if sources_map is not None and source_name:
                sources_map[copied_name] = source_name
            copied += 1

    return copied


def process_mendeley_dataset(
    src_dir: Path,
    out_label_dir: Path,
    out_img_dir: Path,
    source_name: str = "",
    sources_map: Optional[dict[str, str]] = None,
) -> int:
    """
    Process mendeley plaque dataset with custom format.

    Directory structure: data/labels/patientXXXX/*.txt
                        data/images/patientXXXX/*.jpg
    Format: plaque_flag cx cy w h tooth_id

    sources_map (if provided) is populated with {output_filename: source_name}.
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
            for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
                img_src = patient_img_dir / (img_stem + ext)
                if img_src.exists():
                    # Use patient_filename to avoid collisions
                    out_name = f"{patient_dir.name}_{img_stem}"
                    copied_name = f"{out_name}{ext}"
                    shutil.copy(img_src, out_img_dir / copied_name)
                    (out_label_dir / f"{out_name}.txt").write_text("\n".join(lines_out))
                    if sources_map is not None and source_name:
                        sources_map[copied_name] = source_name
                    copied += 1
                    break

    return copied


# ── Split and Dataset YAML Utilities ─────────────────────────────────────────


# ── Source-aware stratified splitting (Generic_Traning_Plan §1.4) ────────────
#
# Rules:
#   - A source with full native train/val/test splits is routed *directly* into
#     our train/val/test (strict isolation: a source's test images can never
#     appear in our train/val).
#   - A source with partial or no native splits is pooled, then stratified-split
#     70/15/15 *per source*. Stratification by source preserves each source's
#     proportional representation in every output split.


@dataclass
class SourceSpec:
    """One source-dataset → (one or more) split contribution."""
    name: str                              # logical source name, used for stratification
    src_img: Path
    src_lbl: Path
    remap: dict[int, int]
    converter: str = "yolo"                # "yolo" | "polygon"
    name_prefix: str = ""                  # output filename prefix (collision-safety)
    native_split: Optional[str] = None     # if "train"/"val"/"test", route there directly


@dataclass
class MendeleySourceSpec:
    """Mendeley plaque dataset (custom per-patient layout)."""
    name: str
    src_base: Path
    native_split: Optional[str] = None     # Mendeley has no native splits → always pool


def _run_source_copy(
    spec: SourceSpec,
    out_lbl_dir: Path,
    out_img_dir: Path,
    sources_map: Optional[dict[str, str]],
) -> int:
    if spec.converter == "polygon":
        return copy_polygon_to_bbox(
            spec.src_lbl, spec.src_img, out_lbl_dir, out_img_dir,
            spec.remap, name_prefix=spec.name_prefix,
            source_name=spec.name, sources_map=sources_map,
        )
    return copy_yolo_with_remap(
        spec.src_lbl, spec.src_img, out_lbl_dir, out_img_dir,
        spec.remap, name_prefix=spec.name_prefix,
        source_name=spec.name, sources_map=sources_map,
    )


def _stratified_split_by_source(
    pool_img: Path,
    pool_lbl: Path,
    out_dir: Path,
    sources_map: dict[str, str],
    seed: int = 42,
) -> dict[str, int]:
    """
    Group pool images by their source name and assign each group 70/15/15.
    Guarantees each source contributes proportionally to train/val/test.
    """
    rng = random.Random(seed)
    by_source: dict[str, list[Path]] = defaultdict(list)
    for img in pool_img.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        src = sources_map.get(img.name, "_unknown")
        by_source[src].append(img)

    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    per_source_report: dict[str, dict[str, int]] = {}
    for src, imgs in by_source.items():
        rng.shuffle(imgs)
        n = len(imgs)
        n_train = math.floor(n * SPLITS["train"])
        n_val = math.floor(n * SPLITS["val"])
        splits["train"].extend(imgs[:n_train])
        splits["val"].extend(imgs[n_train:n_train + n_val])
        splits["test"].extend(imgs[n_train + n_val:])
        per_source_report[src] = {
            "train": n_train,
            "val": n_val,
            "test": n - n_train - n_val,
        }

    counts: dict[str, int] = {}
    for split, imgs in splits.items():
        out_img = out_dir / "images" / split
        out_lbl = out_dir / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy(img, out_img / img.name)
            lbl = pool_lbl / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy(lbl, out_lbl / lbl.name)
        counts[split] = len(imgs)

    for src, r in per_source_report.items():
        print(f"      pool source {src}: train={r['train']} val={r['val']} test={r['test']}")
    return counts


def process_sources(
    out_dir: Path,
    sources: list[SourceSpec],
    mendeley_sources: Optional[list[MendeleySourceSpec]] = None,
    seed: int = 42,
) -> dict[str, int]:
    """
    Materialize all sources under out_dir/{images,labels}/{train,val,test}.

    Sources with `native_split` set are copied directly into that split.
    Sources without are pooled then stratified-split (70/15/15) by source name.
    """
    pool_img = out_dir / "_pool" / "images"
    pool_lbl = out_dir / "_pool" / "labels"
    pool_img.mkdir(parents=True, exist_ok=True)
    pool_lbl.mkdir(parents=True, exist_ok=True)

    pool_sources_map: dict[str, str] = {}
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for spec in sources:
        if not spec.src_img.exists() or not spec.src_lbl.exists():
            print(f"    skip (missing on disk): {spec.name}")
            continue
        if spec.native_split:
            assert spec.native_split in ("train", "val", "test")
            out_img = out_dir / "images" / spec.native_split
            out_lbl = out_dir / "labels" / spec.native_split
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)
            n = _run_source_copy(spec, out_lbl, out_img, sources_map=None)
            counts[spec.native_split] += n
            print(f"    {spec.name} → {spec.native_split} (native): {n}")
        else:
            n = _run_source_copy(spec, pool_lbl, pool_img, sources_map=pool_sources_map)
            print(f"    {spec.name} → pool: {n}")

    if mendeley_sources:
        for ms in mendeley_sources:
            if not ms.src_base.exists():
                print(f"    skip (missing): {ms.name}")
                continue
            if ms.native_split:
                out_img = out_dir / "images" / ms.native_split
                out_lbl = out_dir / "labels" / ms.native_split
                out_img.mkdir(parents=True, exist_ok=True)
                out_lbl.mkdir(parents=True, exist_ok=True)
                n = process_mendeley_dataset(
                    ms.src_base, out_lbl, out_img,
                    source_name=ms.name, sources_map=None,
                )
                counts[ms.native_split] += n
                print(f"    {ms.name} → {ms.native_split} (native): {n}")
            else:
                n = process_mendeley_dataset(
                    ms.src_base, pool_lbl, pool_img,
                    source_name=ms.name, sources_map=pool_sources_map,
                )
                print(f"    {ms.name} → pool: {n}")

    pool_files = [f for f in pool_img.iterdir()
                  if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if pool_files:
        print(f"  Stratified-split of pool ({len(pool_files)} imgs, by source)...")
        pool_counts = _stratified_split_by_source(
            pool_img, pool_lbl, out_dir, pool_sources_map, seed=seed,
        )
        for k, v in pool_counts.items():
            counts[k] += v

    shutil.rmtree(out_dir / "_pool", ignore_errors=True)
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
    with open(yaml_path, "w", encoding="utf-8") as f:
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
    - Carries_Dataset (full native splits, classes 0,1 → 0)
    - Carries_abrasion (partial native splits, polygon, classes 3-8 → 0) → pooled
    - extensive (partial native splits, class 0 → 0) → pooled
    """
    condition = "caries"
    out = OUT_DIR / condition
    shutil.rmtree(out, ignore_errors=True)

    sources: list[SourceSpec] = []

    # Carries_Dataset has full train/valid/test → route directly (strict isolation).
    if CARRIES_DATASET.exists():
        for native, our in (("train", "train"), ("valid", "val"), ("test", "test")):
            sources.append(SourceSpec(
                name="Carries_Dataset",
                src_img=CARRIES_DATASET / native / "images",
                src_lbl=CARRIES_DATASET / native / "yolo",
                remap={0: 0, 1: 0},
                native_split=our,
            ))

    # Carries_abrasion has only train+valid (no test) → pool for stratified split.
    if CARRIES_ABRASION.exists():
        caries_remap = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        for native in ("train", "valid"):
            sources.append(SourceSpec(
                name="Carries_abrasion",
                src_img=CARRIES_ABRASION / "images" / native,
                src_lbl=CARRIES_ABRASION / "labels" / native,
                remap=caries_remap,
                converter="polygon",
                name_prefix="abrasion_",
                native_split=None,  # pooled
            ))

    # extensive has only train+val (no test) → pool for stratified split.
    if EXTENSIVE_DATASET.exists():
        for native in ("train", "val"):
            sources.append(SourceSpec(
                name="extensive",
                src_img=EXTENSIVE_DATASET / "images" / native,
                src_lbl=EXTENSIVE_DATASET / "labels" / native,
                remap={0: 0},
                name_prefix="ext_",
                native_split=None,  # pooled
            ))

    counts = process_sources(out, sources)
    print(f"  Final: train={counts['train']} val={counts['val']} test={counts['test']}")
    write_dataset_yaml(out, condition)


def normalize_gingivitis(domain_shift: bool = False) -> None:  # noqa: ARG001
    """
    Normalize gingivitis datasets:
    - Gingivites_Dataset (full native splits, classes 3,4 → 0)
    - extensive (partial native splits, class 3 → 0) → pooled
    """
    condition = "gingivitis"
    out = OUT_DIR / condition
    shutil.rmtree(out, ignore_errors=True)

    sources: list[SourceSpec] = []

    # Gingivites has full Training/Validation/Test → route directly.
    if GINGIVITES_DATASET.exists():
        for native, our in (("Training", "train"), ("Validation", "val"), ("Test", "test")):
            sources.append(SourceSpec(
                name="Gingivites_Dataset",
                src_img=GINGIVITES_DATASET / native / "Images",
                src_lbl=GINGIVITES_DATASET / native / "Labels",
                remap={3: 0, 4: 0},
                native_split=our,
            ))

    # extensive: train+val only, no test → pool for stratified split.
    if EXTENSIVE_DATASET.exists():
        for native in ("train", "val"):
            sources.append(SourceSpec(
                name="extensive",
                src_img=EXTENSIVE_DATASET / "images" / native,
                src_lbl=EXTENSIVE_DATASET / "labels" / native,
                remap={3: 0},
                name_prefix="ext_",
                native_split=None,
            ))

    counts = process_sources(out, sources)
    print(f"  Final: train={counts['train']} val={counts['val']} test={counts['test']}")
    write_dataset_yaml(out, condition)


def normalize_plaque(domain_shift: bool = False) -> None:  # noqa: ARG001
    """
    Normalize plaque datasets (none have native test splits → all pooled):
    - mendeley-dataset (custom per-patient layout, plaque_flag=1 → class 0)
    - CALCULUS_Dataset (only train, class 0 → 0)
    """
    condition = "plaque"
    out = OUT_DIR / condition
    shutil.rmtree(out, ignore_errors=True)

    sources: list[SourceSpec] = []
    mendeley_sources: list[MendeleySourceSpec] = []

    if MENDELEY_DATASET.exists():
        mendeley_sources.append(MendeleySourceSpec(
            name="mendeley", src_base=MENDELEY_DATASET, native_split=None,
        ))

    if CALCULUS_DATASET.exists():
        sources.append(SourceSpec(
            name="CALCULUS_Dataset",
            src_img=CALCULUS_DATASET / "train" / "images",
            src_lbl=CALCULUS_DATASET / "train" / "labels",
            remap={0: 0},
            name_prefix="calc_",
            native_split=None,
        ))

    counts = process_sources(out, sources, mendeley_sources=mendeley_sources)
    print(f"  Final: train={counts['train']} val={counts['val']} test={counts['test']}")
    write_dataset_yaml(out, condition)


def normalize_discoloration(domain_shift: bool = False) -> None:  # noqa: ARG001
    """
    Normalize discoloration datasets:
    - extensive (train+val, no native test, class 2 → 0) → pooled and stratified.
    """
    condition = "discoloration"
    out = OUT_DIR / condition
    shutil.rmtree(out, ignore_errors=True)

    sources: list[SourceSpec] = []
    if EXTENSIVE_DATASET.exists():
        for native in ("train", "val"):
            sources.append(SourceSpec(
                name="extensive",
                src_img=EXTENSIVE_DATASET / "images" / native,
                src_lbl=EXTENSIVE_DATASET / "labels" / native,
                remap={2: 0},
                name_prefix="ext_",
                native_split=None,
            ))

    counts = process_sources(out, sources)
    print(f"  Final: train={counts['train']} val={counts['val']} test={counts['test']}")
    write_dataset_yaml(out, condition)


def normalize_ulcer(domain_shift: bool = False) -> None:  # noqa: ARG001
    """
    Normalize ulcer datasets:
    - extensive (train+val, no native test, class 1 → 0) → pooled and stratified.
    """
    condition = "ulcer"
    out = OUT_DIR / condition
    shutil.rmtree(out, ignore_errors=True)

    sources: list[SourceSpec] = []
    if EXTENSIVE_DATASET.exists():
        for native in ("train", "val"):
            sources.append(SourceSpec(
                name="extensive",
                src_img=EXTENSIVE_DATASET / "images" / native,
                src_lbl=EXTENSIVE_DATASET / "labels" / native,
                remap={1: 0},
                name_prefix="ext_",
                native_split=None,
            ))

    counts = process_sources(out, sources)
    print(f"  Final: train={counts['train']} val={counts['val']} test={counts['test']}")
    write_dataset_yaml(out, condition)


def normalize_recession(domain_shift: bool = False) -> None:  # noqa: ARG001
    """
    Normalize recession datasets:
    - gum_recession_dataset (only train, class 0 → 0) → pooled
    - Spot (full native splits, polygon, class 5 → 0; other classes dropped) → routed
    - big_gum_dataset (full native splits, bbox, class 1 receding_gum → 0) → routed
    - Augment train set only if it ends up under target_count.
    """
    condition = "recession"
    out = OUT_DIR / condition
    shutil.rmtree(out, ignore_errors=True)

    sources: list[SourceSpec] = []

    # gum_recession_dataset: train only, no native val/test → pool.
    if GUM_RECESSION.exists():
        sources.append(SourceSpec(
            name="gum_recession_dataset",
            src_img=GUM_RECESSION / "train" / "images",
            src_lbl=GUM_RECESSION / "train" / "labels",
            remap={0: 0},
            name_prefix="gum_",
            native_split=None,
        ))

    # Spot: full native train/valid/test → route directly.
    if SPOT_DATASET.exists():
        for native, our in (("train", "train"), ("valid", "val"), ("test", "test")):
            sources.append(SourceSpec(
                name="Spot",
                src_img=SPOT_DATASET / native / "images",
                src_lbl=SPOT_DATASET / native / "labels",
                remap={5: 0},
                converter="polygon",
                name_prefix="spot_",
                native_split=our,
            ))

    # big_gum_dataset: full native train/valid/test → route directly.
    if BIG_GUM_DATASET.exists():
        for native, our in (("train", "train"), ("valid", "val"), ("test", "test")):
            sources.append(SourceSpec(
                name="big_gum_dataset",
                src_img=BIG_GUM_DATASET / native / "images",
                src_lbl=BIG_GUM_DATASET / native / "labels",
                remap={1: 0},
                name_prefix="big_gum_",
                native_split=our,
            ))

    counts = process_sources(out, sources)
    print(f"  Final (pre-aug): train={counts['train']} val={counts['val']} test={counts['test']}")

    # Augment train set only if still under target (with three sources merged
    # it usually isn't, so this becomes a no-op for current data).
    train_img = out / "images" / "train"
    train_lbl = out / "labels" / "train"
    if train_img.exists():
        augmented = augment_small_dataset(train_img, train_lbl, target_count=300)
        if augmented:
            print(f"  Augmented +{augmented} train images")

    write_dataset_yaml(out, condition)


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
