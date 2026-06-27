"""
src/pipeline.py
────────────────
Top-level orchestrator for the full Bina training pipeline.

Phases (in canonical order):
  1. normalize  — convert raw datasets to YOLO format (stratified 70/15/15)
  2. hpo        — Optuna mini-runs for the current default architecture
                  (writes runs/hpo/<cond>_best.json; auto-picked up by train)
  3. sweep      — Multi-arch Phase 1: for each (arch, condition), run HPO →
                  train → threshold → KPI gate. Writes runs/sweep/results.json.
                  Each arch lives in its own runs/sweep/<arch>/specialist_<c>/.
  4. promote    — Pick the per-condition winner from the sweep and copy it
                  into runs/specialists/specialist_<cond>/ for downstream use.
  5. train      — Single-arch specialist training (legacy / re-training a
                  promoted arch with different overrides).
  6. validate   — PR-curve threshold calibration + KPI gate against test set
                  (writes threshold.json + kpi_gate.json next to each ckpt).
  7. pseudo     — Ensemble inference over unlabeled images, merge annotations.
  8. student    — Train final multi-class model on real + pseudo data.
  9. export     — ONNX export + latency benchmark (hardware viability gate).

`--phase all` runs normalize → train → validate → pseudo → student → export.
HPO, sweep, and promote are intentionally NOT in `all` because they're heavy
and exploratory; invoke them explicitly when you want to refresh configs or
rerun the Phase-1 arch comparison.

Usage:
  python src/pipeline.py --phase all
  python src/pipeline.py --phase sweep --conditions recession caries
  python src/pipeline.py --phase promote
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
from train.sweep import run_sweep, ALL_ARCHS, _summary_table as _sweep_summary
from train.promote import promote
from inference.ensemble import load_specialists, run_ensemble
from inference.merge import (
    merge_detections,
    write_student_dataset_yaml,   # legacy — kept for backwards compat
    print_merge_report,
)
from student.prepare_dataset import prepare_student_dataset
from validation.threshold_finder import find_specialist_threshold, _summarize as _th_summary
from validation.kpi_gate import evaluate_specialist, write_manifest as write_kpi_manifest, _summarize as _kpi_summary
from export.onnx_export import export_all_specialists, export_student
from export.latency_benchmark import benchmark_specialists, benchmark_student

PIPELINE_CFG = ROOT / "configs" / "pipeline.yaml"
CONDITIONS = ["caries", "gingivitis", "plaque", "discoloration", "ulcer", "recession"]


def load_cfg() -> dict:
    with open(PIPELINE_CFG, encoding="utf-8") as f:
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


# ── Phase 1B: Multi-architecture specialist sweep (Phase 1A of the plan) ─────

def phase_sweep(archs: list[str], conditions: list[str], *,
                skip_hpo: bool, hpo_trials: int, hpo_epochs: int,
                hpo_fraction: float, hpo_seed: int, resume: bool,
                train_overrides: dict | None = None):
    print("\n" + "█" * 60)
    print("  PHASE 1B — MULTI-ARCH SPECIALIST SWEEP")
    print("█" * 60)
    out = run_sweep(
        archs, conditions,
        skip_hpo=skip_hpo,
        hpo_trials=hpo_trials,
        hpo_epochs=hpo_epochs,
        hpo_fraction=hpo_fraction,
        hpo_seed=hpo_seed,
        resume=resume,
        train_overrides=train_overrides,
    )
    print(_sweep_summary(out["results"]))
    return out


def phase_promote(conditions: list[str], force_best: bool):
    print("\n" + "█" * 60)
    print("  PHASE 1C — PROMOTE PER-CONDITION SPECIALIST WINNERS")
    print("█" * 60)
    return promote(conditions, force_best=force_best)


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

def phase_student_prepare():
    """Phase 3a — assemble the unified multi-class student dataset.

    Materializes data/student/ by copying real data with single-class→unified
    class-id remapping AND appending the (filtered) pseudo-labels. Separated
    from training so you can re-prep without retraining (e.g. after refiltering
    pseudo-labels)."""
    print("\n" + "█" * 60)
    print("  PHASE 3a — STUDENT DATASET PREPARATION")
    print("█" * 60)
    dataset_yaml = prepare_student_dataset()
    print(f"\n✓ Phase 3a complete.  Dataset YAML: {dataset_yaml}")
    return dataset_yaml


def phase_student(conditions: list[str]):
    """Phase 3b — train the multi-class student on the unified dataset.

    Reads data/student/dataset.yaml (produced by phase_student_prepare). Falls
    back to running prepare if the unified dataset doesn't exist yet so the
    user can run --phase student in one shot."""
    cfg = load_cfg()
    s_cfg = cfg["student"]

    print("\n" + "█" * 60)
    print("  PHASE 3b — STUDENT MODEL TRAINING")
    print("█" * 60)

    dataset_yaml = ROOT / "data" / "student" / "dataset.yaml"
    if not dataset_yaml.exists():
        print("  No data/student/dataset.yaml — running prep first.")
        dataset_yaml = phase_student_prepare()

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
        choices=["all", "normalize", "hpo", "sweep", "promote", "train",
                 "validate", "pseudo", "student-prepare", "student", "export"],
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
    # Sweep / promote
    parser.add_argument("--archs", nargs="+", default=ALL_ARCHS,
                        help=f"architectures to sweep (default: {ALL_ARCHS})")
    parser.add_argument("--skip-hpo", action="store_true",
                        help="skip HPO mini-runs inside the sweep")
    parser.add_argument("--force-best", action="store_true",
                        help="promote highest-mAP even if no candidate "
                             "passes the KPI gate (smoke-test only)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override train.epochs in pipeline.yaml "
                             "(useful for smoke tests; e.g. --epochs 3)")
    parser.add_argument("--batch", type=int, default=None,
                        help="override train.batch in pipeline.yaml")
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
        elif phase == "sweep":
            train_overrides: dict = {}
            if args.epochs is not None:
                train_overrides["epochs"] = args.epochs
            if args.batch is not None:
                train_overrides["batch"] = args.batch
            phase_sweep(
                args.archs, args.conditions,
                skip_hpo=args.skip_hpo,
                hpo_trials=args.hpo_trials,
                hpo_epochs=args.hpo_epochs,
                hpo_fraction=args.hpo_fraction,
                hpo_seed=args.hpo_seed,
                resume=args.resume,
                train_overrides=train_overrides or None,
            )
        elif phase == "promote":
            phase_promote(args.conditions, force_best=args.force_best)
        elif phase == "train":
            phase_train(args.conditions, args.resume)
        elif phase == "validate":
            phase_validate(args.conditions)
        elif phase == "pseudo":
            phase_pseudo(args.conditions)
        elif phase == "student-prepare":
            phase_student_prepare()
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
