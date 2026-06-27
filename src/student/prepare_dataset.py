"""
src/student/prepare_dataset.py
──────────────────────────────
Phase 3 dataset assembly: merge the 6 per-condition real datasets + the
filtered pseudo-labels into a single unified multi-class YOLO dataset for
the student.

Critical step: per-condition real labels are single-class (`class_id=0` for
every box). The student needs UNIFIED class IDs (0=caries, 1=gingivitis,
2=plaque, 3=discoloration, 4=ulcer, 5=recession), so we rewrite every label
file during the copy. Without this remap the student would see every real
image as "caries" and learn nothing about the other 5 classes.

Pseudo-labels already carry unified IDs (set by ensemble.py's
UNIFIED_CLASS_IDS at generation time) — they get copied as-is.

Per the locked Phase 3 design (memory: project_phase3_student_design.md),
we use **file copies, not symlinks** because Windows symlinks need admin
mode and silently fall back to copies anyway.

Filename collisions across conditions are avoided by prefixing every file
with the source condition (or "pseudo" for the pseudo pool).

Output layout:
    data/student/
    ├── images/{train,val,test}/<cond>__<orig>.jpg
    ├── labels/{train,val,test}/<cond>__<orig>.txt
    └── dataset.yaml          # nc=6, train/val/test paths

Usage (called via pipeline.py):
    python src/pipeline.py --phase student-prepare
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PIPELINE_CFG = ROOT / "configs" / "pipeline.yaml"

CONDITIONS = ["caries", "gingivitis", "plaque", "discoloration",
              "ulcer", "recession"]
# Single source of truth for unified class IDs. Mirrors
# inference/ensemble.py:UNIFIED_CLASS_IDS — keep in sync.
UNIFIED_CLASS_IDS = {c: i for i, c in enumerate(CONDITIONS)}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}


def _rewrite_label_with_class_id(src_lbl: Path, dst_lbl: Path,
                                  unified_id: int) -> int:
    """Read a single-class YOLO label, rewrite every line's class column to
    `unified_id`. Returns the number of boxes written."""
    if not src_lbl.exists():
        dst_lbl.write_text("", encoding="utf-8")
        return 0
    n = 0
    out_lines = []
    for line in src_lbl.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        # Source class id is 0 (per-condition single-class). Overwrite.
        new_parts = [str(unified_id)] + parts[1:5]
        out_lines.append(" ".join(new_parts))
        n += 1
    dst_lbl.write_text("\n".join(out_lines) + ("\n" if out_lines else ""),
                       encoding="utf-8")
    return n


def _copy_split(cond: str, split: str, real_root: Path, out_root: Path) -> dict:
    """Copy all images+labels for one (condition, split). Returns a dict of
    counts for the report."""
    unified_id = UNIFIED_CLASS_IDS[cond]
    src_img = real_root / cond / "images" / split
    src_lbl = real_root / cond / "labels" / split
    dst_img = out_root / "images" / split
    dst_lbl = out_root / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    if not src_img.exists():
        return {"images": 0, "boxes": 0}

    n_img = 0
    n_box = 0
    for img in src_img.iterdir():
        if img.suffix not in IMG_EXTS:
            continue
        new_name = f"{cond}__{img.name}"
        shutil.copy2(img, dst_img / new_name)
        n_box += _rewrite_label_with_class_id(
            src_lbl / f"{img.stem}.txt",
            dst_lbl / f"{cond}__{img.stem}.txt",
            unified_id,
        )
        n_img += 1
    return {"images": n_img, "boxes": n_box}


def _append_pseudo_to_train(pseudo_dir: Path, out_root: Path) -> dict:
    """Copy pseudo-labeled images+labels into the TRAIN split only.
    Pseudo labels already carry unified class IDs — no remap needed.
    Val/test stay real-only for honest evaluation."""
    src_img = pseudo_dir / "images"
    src_lbl = pseudo_dir / "labels"
    dst_img = out_root / "images" / "train"
    dst_lbl = out_root / "labels" / "train"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    if not src_img.exists():
        return {"images": 0, "boxes": 0, "skipped": True}

    n_img = 0
    n_box = 0
    for img in src_img.iterdir():
        if img.suffix not in IMG_EXTS:
            continue
        new_name = f"pseudo__{img.name}"
        shutil.copy2(img, dst_img / new_name)
        lbl_src = src_lbl / f"{img.stem}.txt"
        lbl_dst = dst_lbl / f"pseudo__{img.stem}.txt"
        if lbl_src.exists():
            content = lbl_src.read_text(encoding="utf-8")
            lbl_dst.write_text(content, encoding="utf-8")
            n_box += sum(1 for ln in content.splitlines() if ln.strip())
        else:
            lbl_dst.write_text("", encoding="utf-8")
        n_img += 1
    return {"images": n_img, "boxes": n_box, "skipped": False}


def _resolve_pseudo_dir() -> Path | None:
    """Prefer the filtered/clean pool; fall back to raw if clean doesn't
    exist. Returns None if neither exists (real-only training)."""
    clean = ROOT / "data" / "pseudo_labeled_clean"
    raw = ROOT / "data" / "pseudo_labeled"
    if clean.exists() and any(clean.iterdir()):
        return clean
    if raw.exists() and any(raw.iterdir()):
        return raw
    return None


def prepare_student_dataset(
    *,
    out_dir: Path | None = None,
    real_root: Path | None = None,
    pseudo_dir: Path | None = None,
    overwrite: bool = True,
) -> Path:
    """Materialize the unified student dataset on disk and write dataset.yaml.

    Returns the path to dataset.yaml.
    """
    cfg = yaml.safe_load(PIPELINE_CFG.read_text(encoding="utf-8"))
    real_root = real_root or (ROOT / cfg["paths"]["data_processed"])
    out_root = out_dir or (ROOT / "data" / "student")
    if pseudo_dir is None:
        pseudo_dir = _resolve_pseudo_dir()

    if overwrite and out_root.exists():
        print(f"  wiping existing {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  Phase 3 — Student dataset preparation")
    print(f"{'=' * 60}")
    print(f"  real source:   {real_root}")
    print(f"  pseudo source: {pseudo_dir if pseudo_dir else '(none — real only)'}")
    print(f"  output:        {out_root}")
    print()

    # Real data: copy each (condition, split), remapping single-class 0 →
    # unified class id. val and test stay real-only (no pseudo) so the
    # student's evaluation isn't polluted by pseudo-label noise.
    splits = ("train", "val", "test")
    per_split = {s: {"images": 0, "boxes": 0} for s in splits}
    print("  Copying real data with class-id remap:")
    print("    cond            split    images    boxes")
    print("    " + "-" * 42)
    for cond in CONDITIONS:
        for split in splits:
            stats = _copy_split(cond, split, real_root, out_root)
            per_split[split]["images"] += stats["images"]
            per_split[split]["boxes"] += stats["boxes"]
            print(f"    {cond:<14}  {split:<6}  {stats['images']:>6}  "
                  f"{stats['boxes']:>7}")

    # Pseudo data: train split only.
    print()
    if pseudo_dir:
        print(f"  Appending pseudo-labels to train split (from {pseudo_dir.name}):")
        pseudo_stats = _append_pseudo_to_train(pseudo_dir, out_root)
        if pseudo_stats.get("skipped"):
            print("    skipped — pseudo dir empty")
        else:
            print(f"    pseudo          train   {pseudo_stats['images']:>6}  "
                  f"{pseudo_stats['boxes']:>7}")
            per_split["train"]["images"] += pseudo_stats["images"]
            per_split["train"]["boxes"] += pseudo_stats["boxes"]
    else:
        print("  No pseudo-labels available — training on real data only")
        print("  (run --phase pseudo then filter_pseudo.py first to add pseudo)")

    # dataset.yaml
    dataset_yaml = out_root / "dataset.yaml"
    content = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(CONDITIONS),
        "names": list(CONDITIONS),
    }
    dataset_yaml.write_text(yaml.dump(content, default_flow_style=False),
                            encoding="utf-8")

    print()
    print(f"  → wrote {dataset_yaml}")
    print()
    print("  TOTAL across all splits:")
    for split in splits:
        s = per_split[split]
        print(f"    {split:<6}  images={s['images']:>6}  boxes={s['boxes']:>7}")
    return dataset_yaml


if __name__ == "__main__":
    prepare_student_dataset()
