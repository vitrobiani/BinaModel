"""
src/validation/kpi_gate.py
──────────────────────────
Phase-1 specialist KPI gate (Generic_Traning_Plan §2.4).

A specialist must clear all three on the **test** set before it is allowed to
generate pseudo-labels for Phase 2:

    mAP@0.5   >= 0.85
    precision >= 0.95   (at the val-calibrated threshold from threshold_finder)
    recall    >= 0.60   (at the same threshold)

Reads:
    runs/specialists/specialist_<cond>/weights/best.pt
    runs/specialists/specialist_<cond>/threshold.json      (must exist; produced
                                                            by threshold_finder)

Writes:
    runs/specialists/specialist_<cond>/kpi_gate.json       (PASS/FAIL manifest)

Usage:
    python src/validation/kpi_gate.py --condition caries
    python src/validation/kpi_gate.py --condition all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train.train_specialist import specialist_run_dir  # noqa: E402
from validation.threshold_finder import (  # noqa: E402
    collect_predictions,
    find_threshold,
)

PIPELINE_CFG = ROOT / "configs" / "pipeline.yaml"
CONDITIONS = ["caries", "gingivitis", "plaque", "discoloration", "ulcer", "recession"]

# Plan §2.4.
TARGET_MAP50 = 0.85
TARGET_PRECISION = 0.95
MIN_RECALL = 0.60


def evaluate_specialist(condition: str, *, arch: str | None = None) -> dict:
    """KPI gate for one (arch, condition) candidate.

    arch=None evaluates the canonical specialist; arch="yolo26s" / ...
    evaluates the corresponding sweep candidate.
    """
    cfg = yaml.safe_load(PIPELINE_CFG.read_text())
    run_dir = specialist_run_dir(condition, arch)
    ckpt = run_dir / "weights" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}")

    threshold_path = run_dir / "threshold.json"
    if not threshold_path.exists():
        raise FileNotFoundError(
            f"No threshold.json at {threshold_path}. "
            "Run threshold_finder first."
        )
    th = json.loads(threshold_path.read_text())
    if not th.get("passed"):
        return {
            "condition": condition,
            "passed": False,
            "reason": "val-calibrated threshold did not pass",
            "threshold_report": th,
        }
    calibrated_conf = float(th["threshold"])

    data_yaml_path = ROOT / "data" / "processed" / condition / "dataset.yaml"
    data_yaml = yaml.safe_load(data_yaml_path.read_text())
    base = Path(data_yaml["path"])
    test_img_rel = data_yaml.get("test", "images/test")
    test_img = base / test_img_rel
    test_lbl = base / test_img_rel.replace("images/", "labels/")

    if not test_img.exists() or not any(test_img.iterdir()):
        return {
            "condition": condition,
            "passed": False,
            "reason": f"no test images at {test_img}",
            "threshold_report": th,
        }

    model = YOLO(str(ckpt))

    # mAP@0.5 on the test split via ultralytics' native validator.
    print(f"  Evaluating mAP@0.5 on test ({test_img})...")
    val_results = model.val(
        data=str(data_yaml_path),
        split="test",
        device=str(cfg["project"].get("device", "0")),
        imgsz=int(cfg["train"].get("imgsz", 640)),
        batch=int(cfg["train"].get("batch", 16)),
        verbose=False,
    )
    # Single-class: take overall mAP50.
    map50 = float(getattr(val_results.box, "map50", 0.0))

    # Precision/recall at the val-calibrated threshold, computed on test.
    print(f"  Computing P/R at calibrated conf>={calibrated_conf:.4f} on test...")
    pred_conf, pred_tp, total_gt = collect_predictions(
        model, test_img, test_lbl,
        device=str(cfg["project"].get("device", "0")),
        imgsz=int(cfg["train"].get("imgsz", 640)),
        batch=int(cfg["train"].get("batch", 16)),
    )
    keep = pred_conf >= calibrated_conf
    tp_test = int(pred_tp[keep].sum())
    fp_test = int((1 - pred_tp[keep]).sum())
    test_precision = tp_test / max(tp_test + fp_test, 1)
    test_recall = tp_test / max(total_gt, 1)

    map_ok = map50 >= TARGET_MAP50
    prec_ok = test_precision >= TARGET_PRECISION
    rec_ok = test_recall >= MIN_RECALL
    passed = map_ok and prec_ok and rec_ok

    result = {
        "condition": condition,
        "passed": passed,
        "calibrated_conf": calibrated_conf,
        "metrics": {
            "mAP50": map50,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_tp": tp_test,
            "test_fp": fp_test,
            "test_gt": int(total_gt),
        },
        "gates": {
            "mAP50_>=0.85": map_ok,
            "precision_>=0.95": prec_ok,
            "recall_>=0.60": rec_ok,
        },
        "threshold_report": th,
    }
    if arch:
        result["arch"] = arch
    return result


def _summarize(r: dict) -> str:
    if not r.get("passed") and "metrics" not in r:
        return f"  [{r['condition']}] FAIL — {r.get('reason')}"
    m = r["metrics"]
    g = r["gates"]
    verdict = "PASS" if r["passed"] else "FAIL"
    return (f"  [{r['condition']}] {verdict}  "
            f"mAP50={m['mAP50']:.3f}{'✓' if g['mAP50_>=0.85'] else '✗'}  "
            f"P={m['test_precision']:.3f}{'✓' if g['precision_>=0.95'] else '✗'}  "
            f"R={m['test_recall']:.3f}{'✓' if g['recall_>=0.60'] else '✗'}  "
            f"(conf>={r['calibrated_conf']:.4f})")


def write_manifest(condition: str, result: dict, *, arch: str | None = None) -> Path:
    out = specialist_run_dir(condition, arch) / "kpi_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="all",
                        choices=CONDITIONS + ["all"])
    parser.add_argument("--arch", default=None,
                        help="sweep-mode architecture slug "
                             "(evaluates runs/sweep/<arch>/specialist_<cond>/)")
    args = parser.parse_args()
    targets = CONDITIONS if args.condition == "all" else [args.condition]
    for cond in targets:
        print(f"\n[{cond}]" + (f" arch={args.arch}" if args.arch else ""))
        try:
            r = evaluate_specialist(cond, arch=args.arch)
            out = write_manifest(cond, r, arch=args.arch)
            print(_summarize(r))
            print(f"  → {out}")
        except FileNotFoundError as e:
            print(f"  skipped: {e}")
