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
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PIPELINE_CFG = ROOT / "configs" / "pipeline.yaml"
CONDITIONS = ["caries", "gingivitis", "plaque", "discoloration", "ulcer", "recession"]

# HPO param keys we accept from runs/hpo/<cond>_best.json
HPO_OVERRIDABLE = ("lr0", "weight_decay", "batch")


def load_pipeline_cfg():
    with open(PIPELINE_CFG, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Arch-aware path conventions ─────────────────────────────────────────────
# When `arch` is None we use the canonical "current winner" locations.
# When `arch` is set (sweep mode), outputs are partitioned by architecture so
# multiple candidates can train and be compared without clobbering each other.


def specialist_project_dir(arch: str | None) -> Path:
    """Where Ultralytics writes runs/<project>/<name>/."""
    if arch:
        return ROOT / "runs" / "sweep" / arch
    return ROOT / "runs" / "specialists"


def specialist_run_dir(condition: str, arch: str | None) -> Path:
    return specialist_project_dir(arch) / f"specialist_{condition}"


def hpo_best_json(condition: str, arch: str | None) -> Path:
    """Per-condition HPO summary; arch-partitioned so a YOLO26x HPO doesn't
    bias a YOLO26s training run."""
    base = ROOT / "runs" / "hpo"
    if arch:
        return base / arch / f"{condition}_best.json"
    return base / f"{condition}_best.json"


def hpo_project_dir(arch: str | None) -> Path:
    base = ROOT / "runs" / "hpo"
    return base / arch if arch else base


def _maybe_apply_hpo_overrides(
    condition: str, train_args: dict, *, arch: str | None = None,
) -> dict:
    """If src/train/hpo.py has produced a top-3 file for (arch, condition),
    silently merge its #1 params into train_args. This is the 'dynamic, not
    hardcoded' HPO link per Generic_Traning_Plan §5.1."""
    hpo_path = hpo_best_json(condition, arch)
    if not hpo_path.exists():
        return train_args
    try:
        hpo = json.loads(hpo_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return train_args
    best = hpo.get("best")
    if not best or not best.get("params"):
        return train_args
    params = best["params"]
    applied = {k: params[k] for k in HPO_OVERRIDABLE if k in params}
    if applied:
        print(f"  ↻ HPO override (trial #{best.get('trial')} "
              f"mAP50={best.get('map50', 0):.3f}): {applied}")
        train_args.update(applied)
    return train_args


def slug_from_weight(weight: str) -> str:
    """Derive an architecture slug from a weight filename.
    'yolo26s.pt' → 'yolo26s'; 'rtdetr-l.pt' → 'rtdetr-l'."""
    return Path(weight).stem


def train_specialist(
    condition: str,
    resume: bool = False,
    overrides: dict = {},
    *,
    arch: str | None = None,
    weight_override: str | None = None,
):
    """Train one specialist via the architecture-appropriate adapter.

    Dispatches every architecture (YOLO/RT-DETR via Ultralytics, Faster R-CNN
    via torchvision, DETR via HuggingFace) through a common SpecialistAdapter
    interface (src/train/adapters/). When `arch` is unset we derive it from
    the configured weight name (e.g. "yolo26s.pt" → "yolo26s").

    Args:
      condition: caries|gingivitis|...
      resume: resume from last.pt of the same (arch, condition) run.
      overrides: explicit dict overrides for train_args (CLI flags).
      arch: optional architecture slug. When set, output goes to
        runs/sweep/<arch>/specialist_<cond>/. When None, output goes to
        runs/specialists/specialist_<cond>/ and `arch` is derived from
        spec["weight"] for HPO lookup purposes.
      weight_override: optional starting weight (defaults to spec["weight"]).
        Typically pass `f"{arch}.pt"` in sweep mode.
    """
    from train.adapters import get_adapter  # lazy import; avoids cycles

    cfg = load_pipeline_cfg()
    spec = cfg["specialists"][condition]
    defaults = cfg["train"]

    model_cfg_path = ROOT / spec["config"]
    with open(model_cfg_path, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    # Merge: pipeline defaults → model-level overrides → HPO best → CLI overrides
    train_args = {**defaults}
    train_args.update(model_cfg.get("overrides", {}))
    train_args = _maybe_apply_hpo_overrides(condition, train_args, arch=arch)
    train_args.update(overrides)

    # Make sure dataset.yaml exists (used by Ultralytics-family adapters).
    data_yaml = ROOT / "data" / "processed" / condition / "dataset.yaml"
    if not data_yaml.exists():
        _write_dataset_yaml(condition, model_cfg)

    weight = weight_override or spec["weight"]
    effective_arch = arch or slug_from_weight(weight)

    # Output dir (canonical vs sweep candidate).
    output_dir = specialist_run_dir(condition, arch)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    arch_tag = f" [arch={arch}]" if arch else f" [arch={effective_arch}]"
    print(f"  Training specialist: {condition.upper()}{arch_tag}")
    print(f"  Dataset:  {data_yaml}")
    print(f"  Weights:  {weight}")
    print(f"  Epochs:   {train_args.get('epochs', 80)}")
    print(f"  Out:      {output_dir}")
    print(f"{'='*60}\n")

    adapter = get_adapter(effective_arch)
    best_ckpt = adapter.train(
        condition=condition,
        train_args=train_args,
        output_dir=output_dir,
        weight=weight,
        resume=resume,
        device=str(cfg["project"]["device"]),
    )
    print(f"\n✓ Specialist [{condition}]{arch_tag} trained.")
    print(f"  Best checkpoint: {best_ckpt}")
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
    with open(out_dir / "dataset.yaml", "w", encoding="utf-8") as f:
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
