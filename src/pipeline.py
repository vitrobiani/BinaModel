"""
src/pipeline.py
────────────────
Top-level orchestrator for the full Bina training pipeline.

Phases:
  1. normalize  — convert raw datasets to YOLO format
  2. train      — train 6 specialist models
  3. pseudo     — run ensemble over unlabeled images, merge annotations
  4. student    — train final multi-class model on real + pseudo data

Usage:
  # Full end-to-end run
  python src/pipeline.py --phase all

  # Individual phases
  python src/pipeline.py --phase normalize
  python src/pipeline.py --phase train
  python src/pipeline.py --phase pseudo
  python src/pipeline.py --phase student

  # Skip to student using pre-existing pseudo labels
  python src/pipeline.py --phase student

  # Only pseudo-label with specific conditions' models
  python src/pipeline.py --phase pseudo --conditions caries gingivitis
"""

import argparse
import sys
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.normalize import NORMALIZERS
from train.train_specialist import train_specialist
from inference.ensemble import load_specialists, run_ensemble
from inference.merge import (
    merge_detections,
    write_student_dataset_yaml,
    print_merge_report,
)

PIPELINE_CFG = ROOT / "configs" / "pipeline.yaml"
CONDITIONS = ["caries", "gingivitis", "plaque", "discoloration", "ulcer", "recession"]


def load_cfg() -> dict:
    with open(PIPELINE_CFG) as f:
        return yaml.safe_load(f)


# ── Phase 1: Normalize ───────────────────────────────────────────────────────

def phase_normalize(conditions: list[str], domain_shift: bool):
    print("\n" + "█"*60)
    print("  PHASE 1 — DATA NORMALIZATION")
    print("█"*60)
    for cond in conditions:
        print(f"\n  [{cond}]")
        NORMALIZERS[cond](domain_shift=domain_shift)
    print("\n✓ Phase 1 complete.")


# ── Phase 2: Train specialists ───────────────────────────────────────────────

def phase_train(conditions: list[str], resume: bool):
    print("\n" + "█"*60)
    print("  PHASE 2 — SPECIALIST MODEL TRAINING")
    print("█"*60)
    checkpoints = {}
    for cond in conditions:
        ckpt = train_specialist(cond, resume=resume)
        checkpoints[cond] = ckpt
    print(f"\n✓ Phase 2 complete. {len(checkpoints)} specialists trained.")
    return checkpoints


# ── Phase 3: Pseudo-labeling ─────────────────────────────────────────────────

def phase_pseudo(conditions: list[str]):
    cfg = load_cfg()
    unlabeled_dir = ROOT / cfg["paths"]["data_unlabeled"]
    pseudo_dir    = ROOT / cfg["paths"]["data_pseudo"]
    cross_iou     = cfg["cross_model_iou_threshold"]
    min_conf      = cfg["student"]["pseudo_label_min_conf"]

    if not unlabeled_dir.exists() or not any(unlabeled_dir.iterdir()):
        print(f"\n⚠  No unlabeled images found at {unlabeled_dir}")
        print("   Drop images (no labels needed) into that directory and re-run.")
        return

    print("\n" + "█"*60)
    print("  PHASE 3 — PSEUDO-LABELING")
    print("█"*60)
    print(f"  Unlabeled images: {unlabeled_dir}")
    print(f"  Output:           {pseudo_dir}")

    t0 = time.time()

    # Load models
    print("\n  Loading specialist models...")
    models = load_specialists(conditions)

    # Run ensemble inference
    print("\n  Running ensemble inference...")
    detections = run_ensemble(
        image_dir=unlabeled_dir,
        models=models,
        batch_size=cfg.get("inference_batch_size", 16),
    )

    # Merge annotations
    print("\n  Merging and filtering annotations...")
    stats = merge_detections(
        detections_by_image=detections,
        output_dir=pseudo_dir,
        image_src_dir=unlabeled_dir,
        pseudo_label_min_conf=min_conf,
        cross_iou_threshold=cross_iou,
        copy_images=True,
    )

    print_merge_report(stats)
    print(f"\n  Time: {(time.time() - t0) / 60:.1f} min")
    print("✓ Phase 3 complete.")
    return pseudo_dir


# ── Phase 4: Student model ───────────────────────────────────────────────────

def phase_student(conditions: list[str]):
    cfg = load_cfg()
    s_cfg = cfg["student"]
    pseudo_dir = ROOT / cfg["paths"]["data_pseudo"]

    # Build list of real data dirs (all conditions' processed dirs)
    real_dirs = [ROOT / cfg["paths"]["data_processed"] / c for c in conditions]

    print("\n" + "█"*60)
    print("  PHASE 4 — STUDENT MODEL TRAINING")
    print("█"*60)

    if not pseudo_dir.exists():
        print("  ⚠  No pseudo-labeled data found. Running on real data only.")
        print("     Run --phase pseudo first to generate pseudo labels.")

    # Write combined dataset YAML
    dataset_yaml = write_student_dataset_yaml(pseudo_dir, real_dirs)

    print(f"\n  Base weights:   {s_cfg['weight']}")
    print(f"  Epochs:         {s_cfg['epochs']}")
    print(f"  Dataset:        {dataset_yaml}")

    model = YOLO(s_cfg["weight"])
    results = model.train(
        data=str(dataset_yaml),
        project=str(ROOT / "runs" / "student"),
        name="bina_v1",
        exist_ok=True,
        device=cfg["project"]["device"],
        epochs=s_cfg["epochs"],
        imgsz=s_cfg["imgsz"],
        batch=s_cfg["batch"],
        # Inherit augmentation from pipeline defaults
        **{k: v for k, v in cfg["train"].items()
           if k not in ("epochs", "imgsz", "batch", "optimizer",
                        "lr0", "lrf", "momentum", "weight_decay",
                        "warmup_epochs", "augment")},
        save=True,
        save_period=10,
        plots=True,
        val=True,
    )

    best = ROOT / "runs" / "student" / "bina_v1" / "weights" / "best.pt"
    print(f"\n✓ Phase 4 complete.")
    print(f"  Final model: {best}")
    print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    return best


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bina Dental Diagnosis — Full Training Pipeline"
    )
    parser.add_argument(
        "--phase",
        choices=["all", "normalize", "train", "pseudo", "student"],
        default="all",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=CONDITIONS,
        help="Restrict which conditions to process (default: all 6)",
    )
    parser.add_argument(
        "--domain-shift",
        action="store_true",
        help="Apply domain-shift augmentation during normalization (plaque)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted specialist training",
    )
    args = parser.parse_args()

    phases = (["normalize", "train", "pseudo", "student"]
              if args.phase == "all" else [args.phase])

    t_total = time.time()

    for phase in phases:
        if phase == "normalize":
            phase_normalize(args.conditions, args.domain_shift)
        elif phase == "train":
            phase_train(args.conditions, args.resume)
        elif phase == "pseudo":
            phase_pseudo(args.conditions)
        elif phase == "student":
            phase_student(args.conditions)

    elapsed = (time.time() - t_total) / 3600
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.2f} hours.")
    print(f"{'='*60}")
