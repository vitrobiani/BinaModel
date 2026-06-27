"""
src/inference/ensemble.py
──────────────────────────
Adapter-routed multi-specialist inference for the pseudo-labeling stage.

Each specialist is loaded via `get_adapter(arch).predict_batch(...)`, so the
ensemble works uniformly across YOLO, RT-DETR, Faster R-CNN, and DETR
winners. Per-specialist confidence thresholds are read from each
specialist's `threshold.json` (produced by validation/threshold_finder.py)
— honoring the plan's §3.3 "use the val-calibrated precision threshold,
not a hand-picked value" rule.

Returns the same detection dict shape that merge.py consumes:
  {
    "image":      Path,
    "condition":  str,
    "class_id":   int,         # unified id (0..5)
    "local_cls":  int,         # always 0 for single-class specialists
    "conf":       float,
    "bbox_yolo":  [cx, cy, w, h],   # normalized YOLO format
    "bbox_xyxy":  [x1, y1, x2, y2], # absolute pixels
  }
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from train.adapters import get_adapter  # noqa: E402

PIPELINE_CFG = ROOT / "configs" / "pipeline.yaml"

UNIFIED_CLASS_IDS = {
    "caries":        0,
    "gingivitis":    1,
    "plaque":        2,
    "discoloration": 3,
    "ulcer":         4,
    "recession":     5,
}


def _load_cfg() -> dict:
    with open(PIPELINE_CFG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_specialist_info(condition: str) -> dict:
    """Bundle per-condition info: arch + ckpt + calibrated confidence.

    Resolution order for arch:
      1. kpi_gate.json["arch"]
      2. threshold.json["arch"]
      3. fallback: slug derived from pipeline.yaml's specialists.<cond>.weight

    Resolution order for confidence threshold:
      1. threshold.json["threshold"] (if "passed" — i.e. P>=0.95 & R>=0.60 met)
      2. kpi_gate.json["calibrated_conf"] (same value, from the gate report)
      3. fallback: pipeline.yaml's specialists.<cond>.conf_thresh
    """
    cfg = _load_cfg()
    run_dir = ROOT / "runs" / "specialists" / f"specialist_{condition}"
    ckpt = run_dir / "weights" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No trained checkpoint for [{condition}] at {ckpt}\n"
            f"Run sweep + promote first, or "
            f"`python src/train/train_specialist.py --condition {condition}`"
        )

    arch: str | None = None
    calibrated_conf: float | None = None

    kpi_path = run_dir / "kpi_gate.json"
    if kpi_path.exists():
        try:
            kpi = json.loads(kpi_path.read_text(encoding="utf-8"))
            arch = kpi.get("arch")
            calibrated_conf = kpi.get("calibrated_conf")
        except (json.JSONDecodeError, OSError):
            pass

    th_path = run_dir / "threshold.json"
    if th_path.exists():
        try:
            th = json.loads(th_path.read_text(encoding="utf-8"))
            if not arch:
                arch = th.get("arch")
            # threshold_finder records the value under "threshold" only when
            # the gate passed; otherwise we keep whatever calibrated_conf is.
            if th.get("passed") and th.get("threshold") is not None:
                calibrated_conf = float(th["threshold"])
        except (json.JSONDecodeError, OSError):
            pass

    if not arch:
        weight = cfg["specialists"][condition]["weight"]
        arch = Path(weight).stem
        print(f"  [{condition}] WARN: no kpi_gate.json / threshold.json — "
              f"falling back to arch={arch} from pipeline.yaml weight")

    if calibrated_conf is None:
        calibrated_conf = float(cfg["specialists"][condition].get(
            "conf_thresh", 0.40))
        print(f"  [{condition}] WARN: no calibrated threshold — "
              f"falling back to {calibrated_conf} from pipeline.yaml")

    iou_thresh = float(cfg["specialists"][condition].get("iou_thresh", 0.45))

    return {
        "arch": arch,
        "ckpt": ckpt,
        "calibrated_conf": float(calibrated_conf),
        "iou_thresh": iou_thresh,
    }


def load_specialists(conditions: list[str] | None = None) -> dict[str, dict]:
    """Resolve per-condition info dicts (NOT loaded model objects).

    Returning info dicts (instead of preloaded `YOLO` instances) lets each
    adapter load the model lazily inside its `predict_batch` call, which:
      - keeps VRAM occupancy low across conditions
      - lets non-Ultralytics adapters (FRCNN, DETR) load their own way
      - prints what arch + threshold will be used per specialist up-front
    """
    cfg = _load_cfg()
    targets = conditions or list(cfg["specialists"].keys())
    info: dict[str, dict] = {}
    for cond in targets:
        i = _load_specialist_info(cond)
        info[cond] = i
        print(f"  [{cond}] arch={i['arch']}  conf≥{i['calibrated_conf']:.4f}")
    return info


def run_ensemble(
    image_dir: Path,
    models: dict[str, dict],
    batch_size: int = 16,
    img_exts: tuple = (".jpg", ".jpeg", ".png", ".JPG", ".PNG"),
) -> dict[str, list[dict]]:
    """Adapter-routed ensemble over `image_dir`.

    `models` is the dict returned by `load_specialists()` — kept named `models`
    for backwards compatibility with pipeline.py's call site.
    """
    cfg = _load_cfg()
    device = str(cfg["project"].get("device", "0"))
    imgsz = int(cfg["train"].get("imgsz", 640))

    images = sorted([p for p in image_dir.iterdir() if p.suffix in img_exts])
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    print(f"\n[Ensemble] {len(images)} images × {len(models)} specialists")

    # Cache (W, H) per image — needed to convert normalized xyxy to absolute,
    # which merge.py's cross-class IoU computation requires.
    sizes_by_stem: dict[str, tuple[int, int]] = {}
    for img in tqdm(images, desc="  reading sizes", leave=False):
        with Image.open(img) as pil:
            sizes_by_stem[img.stem] = pil.size  # (W, H)

    detections_by_image: dict[str, list[dict]] = {img.stem: [] for img in images}

    for condition, info in models.items():
        unified_id = UNIFIED_CLASS_IDS[condition]
        arch = info["arch"]
        ckpt = info["ckpt"]
        conf = info["calibrated_conf"]

        print(f"\n  ▶ {condition}  arch={arch}  conf≥{conf:.4f}")
        t0 = time.time()

        adapter = get_adapter(arch)
        preds = adapter.predict_batch(
            ckpt, images,
            conf_min=conf,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
        )

        n_dets = 0
        for img_path, pred in zip(images, preds):
            W, H = sizes_by_stem[img_path.stem]
            for k in range(len(pred.boxes_xyxyn)):
                x1n, y1n, x2n, y2n = pred.boxes_xyxyn[k]
                score = float(pred.scores[k])
                label = int(pred.labels[k])
                # Normalized cxcywh for the YOLO label format used downstream
                cx = float((x1n + x2n) / 2)
                cy = float((y1n + y2n) / 2)
                w_n = float(x2n - x1n)
                h_n = float(y2n - y1n)
                # Absolute pixel xyxy for merge.py's IoU math
                x1 = float(x1n) * W
                y1 = float(y1n) * H
                x2 = float(x2n) * W
                y2 = float(y2n) * H

                detections_by_image[img_path.stem].append({
                    "image":     img_path,
                    "condition": condition,
                    "class_id":  unified_id,
                    "local_cls": label,
                    "conf":      score,
                    "bbox_yolo": [cx, cy, w_n, h_n],
                    "bbox_xyxy": [x1, y1, x2, y2],
                })
                n_dets += 1

        elapsed = time.time() - t0
        print(f"    {n_dets} detections in {elapsed:.1f}s "
              f"({n_dets / max(elapsed, 0.001):.1f} dets/s)")

    return detections_by_image
