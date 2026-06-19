"""
src/train/train_specialist.py
──────────────────────────────
Trains one specialist YOLO model per condition using Ultralytics.
Reads hyperparameters from configs/pipeline.yaml and the per-model YAML.

Usage:
  # Train a single specialist
  python src/train/train_specialist.py --condition caries

  # Train all 6 sequentially
  python src/train/train_specialist.py --condition all

  # Resume an interrupted run
  python src/train/train_specialist.py --condition gingivitis --resume

  # Override any hyperparam on the fly
  python src/train/train_specialist.py --condition plaque --epochs 120 --batch 8
"""

import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PIPELINE_CFG = ROOT / "configs" / "pipeline.yaml"
CONDITIONS = ["caries", "gingivitis", "plaque", "discoloration", "ulcer", "recession"]


def load_pipeline_cfg():
    with open(PIPELINE_CFG) as f:
        return yaml.safe_load(f)


def train_specialist(condition: str, resume: bool = False, overrides: dict = {}):
    cfg = load_pipeline_cfg()
    spec = cfg["specialists"][condition]
    defaults = cfg["train"]

    model_cfg_path = ROOT / spec["config"]
    with open(model_cfg_path) as f:
        model_cfg = yaml.safe_load(f)

    # Merge: pipeline defaults → model-level overrides → CLI overrides
    train_args = {**defaults}
    train_args.update(model_cfg.get("overrides", {}))
    train_args.update(overrides)

    # Resolve dataset yaml path (absolute)
    data_yaml = ROOT / "data" / "processed" / condition / "dataset.yaml"
    if not data_yaml.exists():
        _write_dataset_yaml(condition, model_cfg)

    # Output dir
    run_name = f"specialist_{condition}"
    project_dir = ROOT / "runs" / "specialists"

    print(f"\n{'='*60}")
    print(f"  Training specialist: {condition.upper()}")
    print(f"  Dataset:  {data_yaml}")
    print(f"  Weights:  {spec['weight']}")
    print(f"  Epochs:   {train_args.get('epochs', 80)}")
    print(f"{'='*60}\n")

    if resume:
        last_ckpt = project_dir / run_name / "weights" / "last.pt"
        if not last_ckpt.exists():
            print(f"  [warn] No checkpoint to resume at {last_ckpt}, starting fresh.")
            resume = False

    model = YOLO(str(project_dir / run_name / "weights" / "last.pt")
                 if resume else spec["weight"])

    results = model.train(
        data=str(data_yaml),
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        resume=resume,
        device=cfg["project"]["device"],
        # Core training params
        epochs=train_args.get("epochs", 80),
        imgsz=train_args.get("imgsz", 640),
        batch=train_args.get("batch", 16),
        optimizer=train_args.get("optimizer", "AdamW"),
        lr0=train_args.get("lr0", 0.001),
        lrf=train_args.get("lrf", 0.01),
        momentum=train_args.get("momentum", 0.937),
        weight_decay=train_args.get("weight_decay", 0.0005),
        warmup_epochs=train_args.get("warmup_epochs", 3),
        # Augmentation
        hsv_h=train_args.get("hsv_h", 0.015),
        hsv_s=train_args.get("hsv_s", 0.4),
        hsv_v=train_args.get("hsv_v", 0.3),
        fliplr=train_args.get("fliplr", 0.5),
        flipud=train_args.get("flipud", 0.0),
        degrees=train_args.get("degrees", 5.0),
        translate=train_args.get("translate", 0.1),
        scale=train_args.get("scale", 0.3),
        mosaic=train_args.get("mosaic", 0.5),
        copy_paste=train_args.get("copy_paste", 0.0),
        # Save
        save=True,
        save_period=10,       # checkpoint every 10 epochs
        plots=True,
        val=True,
    )

    best_ckpt = project_dir / run_name / "weights" / "best.pt"
    print(f"\n✓ Specialist [{condition}] trained.")
    print(f"  Best checkpoint: {best_ckpt}")
    print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    return best_ckpt


def _write_dataset_yaml(condition: str, model_cfg: dict):
    """Auto-generate a dataset.yaml for Ultralytics from our processed directory."""
    out_dir = ROOT / "data" / "processed" / condition
    yaml_content = {
        "path":  str(out_dir),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    model_cfg.get("nc", 1),
        "names": model_cfg.get("names", [condition]),
    }
    with open(out_dir / "dataset.yaml", "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="all",
                        choices=CONDITIONS + ["all"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch",  type=int, default=None)
    parser.add_argument("--imgsz",  type=int, default=None)
    args = parser.parse_args()

    # Build CLI overrides dict (only set args)
    cli_overrides = {k: v for k, v in {
        "epochs": args.epochs,
        "batch":  args.batch,
        "imgsz":  args.imgsz,
    }.items() if v is not None}

    targets = CONDITIONS if args.condition == "all" else [args.condition]
    for cond in targets:
        train_specialist(cond, resume=args.resume, overrides=cli_overrides)

    print("\n✓ All specialist training complete.")
