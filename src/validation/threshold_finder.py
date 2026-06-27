"""
src/validation/threshold_finder.py
──────────────────────────────────
Per-specialist confidence-threshold calibration (Generic_Traning_Plan §2.4).

For a trained single-class specialist, sweep the val set across confidence
levels and find the smallest threshold satisfying both:

    precision >= TARGET_PRECISION   (default 0.95, per plan §2.4 / §3.3)
    recall    >= MIN_RECALL         (default 0.60)

The threshold is what the pseudo-label engine should use for that specialist —
not a hand-picked value in pipeline.yaml. Output is written as a JSON sidecar
next to the checkpoint:

    runs/specialists/specialist_<cond>/threshold.json

Usage:
    python src/validation/threshold_finder.py --condition caries
    python src/validation/threshold_finder.py --condition all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from train.train_specialist import specialist_run_dir, slug_from_weight  # noqa: E402
from train.adapters import get_adapter  # noqa: E402

PIPELINE_CFG = ROOT / "configs" / "pipeline.yaml"
CONDITIONS = ["caries", "gingivitis", "plaque", "discoloration", "ulcer", "recession"]

# Plan §2.4 / §3.3.
TARGET_PRECISION = 0.95
MIN_RECALL = 0.60
IOU_MATCH = 0.50  # mAP@0.5 matching


# ── Geometry helpers (normalized cxcywh space) ───────────────────────────────


def _load_yolo_labels(lbl_path: Path) -> np.ndarray:
    """Read YOLO labels → Nx4 array of [cx, cy, w, h] in normalized coords."""
    if not lbl_path.exists():
        return np.zeros((0, 4))
    rows = []
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            rows.append([float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4])])
    return np.array(rows) if rows else np.zeros((0, 4))


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter = (np.maximum(0, inter_x2 - inter_x1) *
             np.maximum(0, inter_y2 - inter_y1))
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


# ── Prediction collection ────────────────────────────────────────────────────


def collect_predictions(
    ckpt: Path,
    img_dir: Path,
    lbl_dir: Path,
    *,
    arch: str,
    conf_min: float = 0.001,
    imgsz: int = 640,
    device: str = "0",
    batch: int = 16,
    chunk_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Run inference at a very low conf via the per-arch adapter, then greedily
    IoU-match each prediction against ground-truth (highest-conf first,
    IoU>=IOU_MATCH). Family-agnostic — works for YOLO, RT-DETR, FRCNN, DETR.

    `chunk_size` bounds peak RAM by passing only N images at a time to
    adapter.predict_batch — Ultralytics' Results objects hold large tensor
    state, and accumulating 244+ of them at conf_min=0.001 (max_det=300/img)
    has produced MemoryError on plaque-sized val sets.

    Returns:
      pred_conf: (K,) confidence per prediction (across the whole val set)
      pred_tp:   (K,) 1 if matched a GT (TP), else 0 (FP)
      total_gt:  total GT box count
    """
    import gc

    adapter = get_adapter(arch)
    img_paths = sorted([f for f in img_dir.iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    confidences: list[float] = []
    tps: list[int] = []
    total_gt = 0

    for chunk_start in range(0, len(img_paths), chunk_size):
        chunk = img_paths[chunk_start:chunk_start + chunk_size]
        predictions = adapter.predict_batch(
            ckpt, chunk,
            conf_min=conf_min,
            imgsz=imgsz,
            batch=batch,
            device=device,
        )
        total_gt += _accumulate_chunk(
            chunk, predictions, lbl_dir,
            confidences, tps,
        )
        # Drop the chunk's Results so the next chunk starts from clean RAM.
        del predictions
        gc.collect()

    return np.asarray(confidences), np.asarray(tps, dtype=np.int64), total_gt


def _accumulate_chunk(
    img_paths: list[Path],
    predictions: list,
    lbl_dir: Path,
    confidences: list[float],
    tps: list[int],
) -> int:
    """Per-image greedy IoU match → append to confidences/tps lists in place.
    Returns the number of GT boxes seen in this chunk."""
    chunk_gt = 0
    for img_path, pred in zip(img_paths, predictions):
        gt = _load_yolo_labels(lbl_dir / f"{img_path.stem}.txt")
        chunk_gt += len(gt)
        gt_xyxy = _cxcywh_to_xyxy(gt)

        if len(pred.boxes_xyxyn) == 0:
            continue

        order = np.argsort(-pred.scores)
        pred_xyxy = pred.boxes_xyxyn[order]
        pred_conf = pred.scores[order]

        if len(gt_xyxy) == 0:
            for c in pred_conf:
                confidences.append(float(c))
                tps.append(0)
            continue

        iou = _iou_matrix(pred_xyxy, gt_xyxy)
        matched_gt = np.zeros(len(gt_xyxy), dtype=bool)
        for i in range(len(pred_xyxy)):
            best_j, best_iou = -1, IOU_MATCH
            for j in range(len(gt_xyxy)):
                if matched_gt[j]:
                    continue
                if iou[i, j] > best_iou:
                    best_iou = iou[i, j]
                    best_j = j
            if best_j >= 0:
                matched_gt[best_j] = True
                tps.append(1)
            else:
                tps.append(0)
            confidences.append(float(pred_conf[i]))

    return chunk_gt


# ── Threshold selection ──────────────────────────────────────────────────────


def find_threshold(
    pred_conf: np.ndarray,
    pred_tp: np.ndarray,
    total_gt: int,
    *,
    target_precision: float = TARGET_PRECISION,
    min_recall: float = MIN_RECALL,
) -> dict:
    """
    Among prediction rankings (high-conf → low-conf), find the smallest
    confidence at which both precision and recall constraints hold.
    """
    if total_gt == 0:
        return {
            "passed": False,
            "mode": "no_gt",
            "reason": "no ground-truth boxes in val set",
            "threshold": 1.0,
            "precision": 0.0,
            "recall": 0.0,
            "n_predictions": int(len(pred_conf)),
        }
    if len(pred_conf) == 0:
        return {
            "passed": False,
            "mode": "no_predictions",
            "reason": "model produced zero predictions even at conf_min=0.001",
            "threshold": 1.0,
            "precision": 0.0,
            "recall": 0.0,
            "total_gt": int(total_gt),
            "n_predictions": 0,
        }

    order = np.argsort(-pred_conf)
    confs = pred_conf[order]
    tps = pred_tp[order]

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(1 - tps)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    recall = cum_tp / total_gt

    # Tier 1 — strict gate: smallest threshold (highest index i, since confs
    # are sorted DESC) where precision[i] >= target AND recall[i] >= min.
    strict_mask = (precision >= target_precision) & (recall >= min_recall)
    if strict_mask.any():
        i = int(np.where(strict_mask)[0][-1])
        return {
            "passed": True,
            "mode": "strict",
            "threshold": float(confs[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "tp": int(cum_tp[i]),
            "fp": int(cum_fp[i]),
            "total_gt": int(total_gt),
            "n_predictions": int(len(pred_conf)),
        }

    # Tier 2 — recall-anchored fallback: the strict gate failed, but the
    # specialist might still produce usable pseudo-labels at a less aggressive
    # threshold. Pick the threshold that MAXIMIZES PRECISION subject to
    # recall >= min_recall. This avoids the old failure mode where the model
    # picked the conf at which precision peaked (often 1 detection in entire
    # val set → P=1.0, R=0.001 — useless for pseudo-labeling).
    recall_mask = recall >= min_recall
    if recall_mask.any():
        precision_when_recall_ok = np.where(recall_mask, precision, -1)
        i = int(np.argmax(precision_when_recall_ok))
        return {
            "passed": False,
            "mode": "recall_anchored",
            "reason": (f"strict gate failed (P>={target_precision} & "
                       f"R>={min_recall}); fell back to max-precision at "
                       f"R>={min_recall}"),
            "threshold": float(confs[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "tp": int(cum_tp[i]),
            "fp": int(cum_fp[i]),
            "total_gt": int(total_gt),
            "n_predictions": int(len(pred_conf)),
        }

    # Tier 3 — F1-optimal last resort: even the recall floor is unreachable.
    # F1 = 2PR/(P+R) gives the most balanced operating point. Still usable as
    # a diagnostic threshold even though pseudo-labels would be sparse.
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-10)
    i = int(np.argmax(f1))
    return {
        "passed": False,
        "mode": "f1_optimal",
        "reason": (f"strict gate failed AND R>={min_recall} unreachable; "
                   f"fell back to F1-optimal threshold"),
        "threshold": float(confs[i]),
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "f1": float(f1[i]),
        "tp": int(cum_tp[i]),
        "fp": int(cum_fp[i]),
        "total_gt": int(total_gt),
        "n_predictions": int(len(pred_conf)),
    }


# ── Per-specialist driver ────────────────────────────────────────────────────


def find_specialist_threshold(condition: str, *, arch: str | None = None) -> dict:
    """Calibrate the per-(arch, condition) confidence threshold.

    arch=None operates on the canonical runs/specialists/specialist_<cond>/
    (and derives the effective arch from pipeline.yaml's spec["weight"]).
    arch="yolo26s" / "rtdetr-l" / "frcnn-r50" / "detr-r50" / ... operates on
    the sweep candidate at runs/sweep/<arch>/specialist_<cond>/.
    """
    cfg = yaml.safe_load(PIPELINE_CFG.read_text(encoding="utf-8"))
    run_dir = specialist_run_dir(condition, arch)
    ckpt = run_dir / "weights" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No trained specialist at {ckpt}")

    # Derive effective arch (for adapter dispatch) when caller didn't pass one.
    # Order of preference for the canonical specialist dir (arch=None):
    #   1. existing kpi_gate.json (copied by promote.py from the winning sweep)
    #   2. runs/sweep/promotions.json
    #   3. pipeline.yaml weight slug (legacy default)
    # Without this, conditions promoted from frcnn-r50 / detr-r50 would crash
    # here trying to YOLO()-load a non-Ultralytics checkpoint.
    effective_arch = arch or _detect_promoted_arch(condition) or slug_from_weight(
        cfg["specialists"][condition]["weight"]
    )

    data_yaml_path = ROOT / "data" / "processed" / condition / "dataset.yaml"
    data_yaml = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    base = Path(data_yaml["path"])
    val_img_rel = data_yaml["val"]
    val_img = base / val_img_rel
    val_lbl = base / val_img_rel.replace("images/", "labels/")

    print(f"  ckpt:  {ckpt}")
    print(f"  val:   {val_img}")
    print(f"  arch:  {effective_arch}")

    pred_conf, pred_tp, total_gt = collect_predictions(
        ckpt, val_img, val_lbl,
        arch=effective_arch,
        device=str(cfg["project"].get("device", "0")),
        imgsz=int(cfg["train"].get("imgsz", 640)),
        batch=int(cfg["train"].get("batch", 16)),
    )

    result = find_threshold(pred_conf, pred_tp, total_gt)
    result["condition"] = condition
    result["arch"] = effective_arch   # always record — downstream depends on it

    out_path = run_dir / "threshold.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  → {out_path}")
    return result


def _detect_promoted_arch(condition: str) -> str | None:
    """Look up which arch was promoted for `condition` into runs/specialists/.
    Returns None if no promotion record exists (caller falls back to pipeline
    yaml). Used by find_specialist_threshold when arch=None so we don't
    YOLO()-load a torchvision/HF checkpoint."""
    canonical_dir = specialist_run_dir(condition, None)
    # Prefer kpi_gate.json next to the checkpoint — promote.py copies it from
    # the winning sweep run, so it always carries the right arch.
    kpi_path = canonical_dir / "kpi_gate.json"
    if kpi_path.exists():
        try:
            kpi = json.loads(kpi_path.read_text(encoding="utf-8"))
            if kpi.get("arch"):
                return str(kpi["arch"])
        except (json.JSONDecodeError, OSError):
            pass
    # Fallback: read the promotions manifest.
    promotions_path = ROOT / "runs" / "sweep" / "promotions.json"
    if promotions_path.exists():
        try:
            pr = json.loads(promotions_path.read_text(encoding="utf-8"))
            entry = pr.get("promotions", {}).get(condition, {})
            if entry.get("arch"):
                return str(entry["arch"])
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _summarize(condition: str, r: dict) -> str:
    if r.get("passed"):
        return (f"  [{condition}] PASS conf>={r['threshold']:.4f}  "
                f"P={r['precision']:.3f}  R={r['recall']:.3f}  "
                f"GT={r['total_gt']}  preds={r['n_predictions']}")
    # New 3-tier fallback modes (recall_anchored, f1_optimal) all carry a
    # usable threshold + precision/recall pair. Show them so the user sees
    # actual numbers instead of just a reason string.
    if "threshold" in r and "precision" in r and "recall" in r:
        mode = r.get("mode", "fallback")
        return (f"  [{condition}] FAIL ({mode}) conf>={r['threshold']:.4f}  "
                f"P={r['precision']:.3f}  R={r['recall']:.3f}  "
                f"GT={r.get('total_gt', 0)}  preds={r.get('n_predictions', 0)}")
    return f"  [{condition}] FAIL — {r.get('reason', 'unknown')}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="all",
                        choices=CONDITIONS + ["all"])
    parser.add_argument("--arch", default=None,
                        help="sweep-mode architecture slug "
                             "(reads runs/sweep/<arch>/specialist_<cond>/)")
    args = parser.parse_args()
    targets = CONDITIONS if args.condition == "all" else [args.condition]
    for cond in targets:
        try:
            print(f"\n[{cond}]" + (f" arch={args.arch}" if args.arch else ""))
            r = find_specialist_threshold(cond, arch=args.arch)
            print(_summarize(cond, r))
        except FileNotFoundError as e:
            print(f"  skipped: {e}")
