"""
src/train/hpo.py
─────────────────
Optuna "Mini-Run" HPO per Generic_Traning_Plan §5.1.

Workflow:
  1. Create a fractional subset (default 10%) of the processed train/val.
  2. For each Optuna trial, train the chosen base weight for a small number of
     epochs (default 8) and score by mAP@0.5 on the subset's val.
  3. Sample LR ∈ logU(1e-5, 1e-1), weight_decay ∈ logU(1e-5, 1e-3),
     batch ∈ {16, 32, 64} (RandomSampler — plan calls out random over grid).
  4. Save the top-3 configurations to runs/hpo/<condition>_best.json — these
     are what Phase-1 full training should be launched from.

This deliberately scores by mAP@0.5 at low-epoch counts. The plan's intent is
to rank LR/wd choices by initial-trajectory quality, not to converge.

Usage:
  python src/train/hpo.py --condition caries
  python src/train/hpo.py --condition all --n-trials 30 --epochs 10
  python src/train/hpo.py --condition recession --weight yolo11s.pt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PIPELINE_CFG = ROOT / "configs" / "pipeline.yaml"
CONDITIONS = ["caries", "gingivitis", "plaque", "discoloration", "ulcer", "recession"]


def _import_optuna():
    try:
        import optuna  # noqa: F401
        return __import__("optuna")
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Optuna is required for HPO. Install with `pip install optuna`."
        ) from e


def make_subset(condition: str, fraction: float = 0.10, seed: int = 42) -> Path:
    """
    Materialize a fractional copy of data/processed/<condition>/{train,val}.

    Single-class detection has no class-rebalance concern at the image level
    (every image already contains >=1 target box, by normalize.py's per-line
    filter), so random sampling preserves the distribution. For multi-class
    HPO (student stage), this needs replacement with iterative stratification.

    Returns the path to a freshly-written subset dataset.yaml.
    """
    rng = random.Random(seed)
    src = ROOT / "data" / "processed" / condition
    dst = ROOT / "data" / "hpo_subsets" / f"{condition}_p{int(fraction * 100)}"
    shutil.rmtree(dst, ignore_errors=True)

    for split in ("train", "val"):
        src_img = src / "images" / split
        src_lbl = src / "labels" / split
        if not src_img.exists():
            print(f"  WARN: missing {src_img}")
            continue
        imgs = [p for p in src_img.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        rng.shuffle(imgs)
        keep = max(1, math.floor(len(imgs) * fraction))
        chosen = imgs[:keep]

        out_img = dst / "images" / split
        out_lbl = dst / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        for img in chosen:
            shutil.copy(img, out_img / img.name)
            lbl = src_lbl / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy(lbl, out_lbl / lbl.name)
        print(f"  {split} subset: {len(chosen)} / {len(imgs)} images")

    subset_yaml = {
        "path": str(dst.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": [condition],
    }
    out = dst / "dataset.yaml"
    out.write_text(yaml.dump(subset_yaml, default_flow_style=False))
    return out


def _train_one_trial(
    trial,
    condition: str,
    weight: str,
    subset_yaml: Path,
    device: str,
    epochs: int,
    project: Path,
) -> float:
    from ultralytics import YOLO

    lr0 = trial.suggest_float("lr0", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch = trial.suggest_categorical("batch", [16, 32, 64])

    model = YOLO(weight)
    name = f"hpo_{condition}_t{trial.number}"
    try:
        results = model.train(
            data=str(subset_yaml),
            project=str(project),
            name=name,
            exist_ok=True,
            device=device,
            epochs=epochs,
            imgsz=640,
            batch=batch,
            optimizer="AdamW",
            lr0=lr0,
            weight_decay=weight_decay,
            warmup_epochs=min(3, max(1, epochs // 3)),
            save=False,
            plots=False,
            val=True,
            verbose=False,
        )
        return float(getattr(results.box, "map50", 0.0)) if results else 0.0
    except Exception as e:  # noqa: BLE001 — Optuna swallows; we log and return 0
        print(f"  trial {trial.number} crashed: {e!r}")
        return 0.0


def run_hpo(
    condition: str,
    *,
    n_trials: int = 20,
    epochs: int = 8,
    fraction: float = 0.10,
    seed: int = 42,
    weight_override: str | None = None,
) -> dict:
    optuna = _import_optuna()
    cfg = yaml.safe_load(PIPELINE_CFG.read_text())
    spec = cfg["specialists"][condition]
    weight = weight_override or spec["weight"]
    device = str(cfg["project"].get("device", "0"))

    print(f"\n  HPO[{condition}] base={weight} trials={n_trials} epochs={epochs}")
    subset_yaml = make_subset(condition, fraction=fraction, seed=seed)
    project = ROOT / "runs" / "hpo" / condition
    project.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.RandomSampler(seed=seed)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, study_name=f"hpo_{condition}",
    )

    def objective(trial):
        return _train_one_trial(
            trial, condition, weight, subset_yaml, device, epochs, project,
        )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    ranked = sorted(
        (t for t in study.trials if t.value is not None),
        key=lambda t: t.value,
        reverse=True,
    )
    top3 = [
        {"trial": t.number, "map50": float(t.value), "params": t.params}
        for t in ranked[:3]
    ]

    summary = {
        "condition": condition,
        "base_weight": weight,
        "n_trials": n_trials,
        "epochs_per_trial": epochs,
        "subset_fraction": fraction,
        "top3": top3,
        "best": top3[0] if top3 else None,
    }
    out_path = ROOT / "runs" / "hpo" / f"{condition}_best.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"  → {out_path}")
    if top3:
        print(f"  best: trial {top3[0]['trial']}  "
              f"mAP50={top3[0]['map50']:.3f}  params={top3[0]['params']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="all",
                        choices=CONDITIONS + ["all"])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=8,
                        help="epochs per HPO trial (plan §5.1: 5-10)")
    parser.add_argument("--fraction", type=float, default=0.10,
                        help="subset fraction (plan §5.1: 0.10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight", default=None,
                        help="override base weight (e.g. yolo11x.pt, "
                             "rtdetr-l.pt) for sweeping multiple specialist "
                             "architectures")
    args = parser.parse_args()

    targets = CONDITIONS if args.condition == "all" else [args.condition]
    for cond in targets:
        run_hpo(
            cond,
            n_trials=args.n_trials,
            epochs=args.epochs,
            fraction=args.fraction,
            seed=args.seed,
            weight_override=args.weight,
        )
