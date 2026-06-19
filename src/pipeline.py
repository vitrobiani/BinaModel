"""
src/pipeline.py
────────────────
Top-level orchestrator for the full Bina training pipeline.

Phases (in canonical order):
  1. normalize  — convert raw datasets to YOLO format (stratified 70/15/15)
  2. hpo        — Optuna mini-runs to discover per-specialist hyperparams
                  (writes runs/hpo/<cond>_best.json; auto-picked up by train)
  3. train      — train 6 specialist models
  4. validate   — PR-curve threshold calibration + KPI gate against test set
                  (writes threshold.json + kpi_gate.json next to each ckpt)
  5. pseudo     — ensemble inference over unlabeled images, merge annotations
  6. student    — train final multi-class model on real + pseudo data
  7. export     — ONNX export + latency benchmark (hardware viability gate)

`--phase all` runs normalize → train → validate → pseudo → student → export.
HPO is intentionally NOT in `all` because it's exploratory and slow; run it
explicitly with `--phase hpo` when you want to refresh per-condition configs.

Usage:
  python src/pipeline.py --phase all
  python src/pipeline.py --phase hpo --conditions recession
  python src/pipeline.py --phase validate
  python src/pipeline.py --phase export --target specialists
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
from train.hpo import run_hpo
from inference.ensemble import load_specialists, run_ensemble
from inference.merge import (
    merge_detections,
    write_student_dataset_yaml,
    print_merge_report,
)
from validation.threshold_finder import find_specialist_threshold, _summarize as _th_summary
from validation.kpi_gate import evaluate_specialist, write_manifest as write_kpi_manifest, _summarize as _kpi_summary
from export.onnx_export import export_all_specialists, export_student
from export.latency_benchmark import benchmark_specialists, benchmark_student

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


# ── Phase 2 (optional): HPO mini-runs ────────────────────────────────────────

def phase_hpo(conditions: list[str], n_trials: int, epochs: int,
              fraction: float, seed: int, weight_override: str | None):
    print("\n" + "█"*60)
    print("  PHASE 2 — HPO MINI-RUNS  (Optuna, plan §5.1)")
    print("█"*60)
    print(f"  trials/cond={n_trials}  epochs/trial={epochs}  "
          f"subset={int(fraction*100)}%  weight_override={weight_override or 'cfg default'}")
    results = {}
    for cond in conditions:
        results[cond] = run_hpo(
            cond, n_trials=n_trials, epochs=epochs,
            fraction=fraction, seed=seed, weight_override=weight_override,
        )
    print("\n✓ Phase 2 (HPO) complete.")
    return results


# ── Phase 3: Train specialists ───────────────────────────────────────────────

def phase_train(conditions: list[str], resume: bool):
    print("\n" + "█"*60)
    print("  PHASE 3 — SPECIALIST MODEL TRAINING")
    print("█"*60)
    checkpoints = {}
    for cond in conditions:
        ckpt = train_specialist(cond, resume=resume)
        checkpoints[cond] = ckpt
    print(f"\n✓ Phase 3 complete. {len(checkpoints)} specialists trained.")
    return checkpoints


# ── Phase 4: Validate (threshold finder + KPI gate) ──────────────────────────

def phase_validate(conditions: list[str]):
    print("\n" + "█"*60)
    print("  PHASE 4 — VALIDATION (PR threshold + KPI gate, plan §2.4 / §3.3)")
    print("█"*60)

    print("\n  Step 1/2: per-specialist threshold calibration on val")
    for cond in conditions:
        try:
            print(f"\n[{cond}]")
            r = find_specialist_threshold(cond)
            print(_th_summary(cond, r))
        except FileNotFoundError as e:
            print(f"  skipped: {e}")

    print("\n  Step 2/2: KPI gate on test")
    verdicts: dict[str, dict] = {}
    for cond in conditions:
        print(f"\n[{cond}]")
        try:
            r = evaluate_specialist(cond)
            out = write_kpi_manifest(cond, r)
            print(_kpi_summary(r))
            print(f"  → {out}")
            verdicts[cond] = r
        except FileNotFoundError as e:
            print(f"  skipped: {e}")

    n_pass = sum(1 for r in verdicts.values() if r.get("passed"))
    print(f"\n✓ Phase 4 complete. {n_pass}/{len(verdicts)} specialists passed KPI gate.")
    return verdicts


# ── Phase 5: Pseudo-labeling ─────────────────────────────────────────────────

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
    print("✓ Phase 5 complete.")
    return pseudo_dir


# ── Phase 6: Student model ───────────────────────────────────────────────────

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
    print(f"\n✓ Phase 6 complete.")
    print(f"  Final model: {best}")
    print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    return best


# ── Phase 7: ONNX export + latency benchmark ─────────────────────────────────

def phase_export(target: str, device: str, iters: int, warmup: int, imgsz: int):
    print("\n" + "█"*60)
    print("  PHASE 7 — ONNX EXPORT + LATENCY BENCHMARK (plan §4.4 / §5.3)")
    print("█"*60)
    print(f"  target={target}  device={device}  iters={iters}  warmup={warmup}")

    if target in ("specialists", "all"):
        print("\n  Exporting specialists...")
        export_all_specialists(imgsz=imgsz)
        print("\n  Benchmarking specialists...")
        benchmark_specialists(device, iters, warmup, imgsz)

    if target in ("student", "all"):
        print("\n  Exporting student...")
        export_student(imgsz=imgsz)
        print("\n  Benchmarking student...")
        benchmark_student(device, iters, warmup, imgsz)

    print("\n✓ Phase 7 complete.")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bina Dental Diagnosis — Full Training Pipeline"
    )
    parser.add_argument(
        "--phase",
        choices=["all", "normalize", "hpo", "train", "validate",
                 "pseudo", "student", "export"],
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
    # HPO
    parser.add_argument("--hpo-trials", type=int, default=20,
                        help="Optuna trials per condition (--phase hpo)")
    parser.add_argument("--hpo-epochs", type=int, default=8,
                        help="epochs per HPO trial (plan §5.1: 5-10)")
    parser.add_argument("--hpo-fraction", type=float, default=0.10,
                        help="subset fraction for HPO (plan §5.1: 0.10)")
    parser.add_argument("--hpo-seed", type=int, default=42)
    parser.add_argument("--hpo-weight", default=None,
                        help="override base weight for HPO (sweep architectures)")
    # Export / benchmark
    parser.add_argument("--target", choices=["specialists", "student", "all"],
                        default="all", help="export/benchmark target")
    parser.add_argument("--bench-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--bench-iters", type=int, default=500)
    parser.add_argument("--bench-warmup", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    # `all` runs the canonical training pipeline but skips HPO (exploratory).
    phases = (["normalize", "train", "validate", "pseudo", "student", "export"]
              if args.phase == "all" else [args.phase])

    t_total = time.time()

    for phase in phases:
        if phase == "normalize":
            phase_normalize(args.conditions, args.domain_shift)
        elif phase == "hpo":
            phase_hpo(
                args.conditions,
                n_trials=args.hpo_trials,
                epochs=args.hpo_epochs,
                fraction=args.hpo_fraction,
                seed=args.hpo_seed,
                weight_override=args.hpo_weight,
            )
        elif phase == "train":
            phase_train(args.conditions, args.resume)
        elif phase == "validate":
            phase_validate(args.conditions)
        elif phase == "pseudo":
            phase_pseudo(args.conditions)
        elif phase == "student":
            phase_student(args.conditions)
        elif phase == "export":
            phase_export(
                args.target,
                device=args.bench_device,
                iters=args.bench_iters,
                warmup=args.bench_warmup,
                imgsz=args.imgsz,
            )

    elapsed = (time.time() - t_total) / 3600
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.2f} hours.")
    print(f"{'='*60}")
