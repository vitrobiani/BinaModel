"""
DETRAdapter — DETR-R50 via HuggingFace transformers.

The plan (§2.1) lists transformer-based detectors as candidates; DETR-R50 is
the canonical reference. We start from the COCO-pretrained checkpoint
("facebook/detr-resnet-50") and replace the classification head for single-
class fine-tuning (1 foreground class; DETR keeps an implicit "no object"
class internally).

DETR is data-hungry. Conditions with fewer than DETR_MIN_TRAIN_IMAGES (1000)
train images cause the adapter to raise SkipArchitecture — the sweep then
records "skipped" for that (arch, condition) pair instead of training a
guaranteed-poor model. Current dataset state:

    caries     ~3,900 ✓     plaque         ~5,400 ✓
    gingivitis ~1,100 ✓     recession      ~1,960 ✓
    ulcer        ~260 ✗     discoloration    ~185 ✗

Known plan deferrals (deliberate):
  - Cutout/CutMix (§2.3) not implemented; using stock HF augmentation.
  - Hard negative mining (§2.2) — DETR's set-prediction loss makes this less
    directly applicable; deferred until the comparison clarifies whether it
    matters.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import Prediction, SpecialistAdapter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from train.yolo_dataset import (  # noqa: E402
    YoloDirDataset, collate_detr_pre_processor,
)


PRETRAINED = "facebook/detr-resnet-50"
DETR_MIN_TRAIN_IMAGES = 1000


class SkipArchitecture(Exception):
    """Raised when an adapter declines a (arch, condition) pair."""


def _resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def _load_detr(num_labels: int = 1):
    """Load DETR pretrained on COCO and replace the class head."""
    from transformers import DetrForObjectDetection
    model = DetrForObjectDetection.from_pretrained(
        PRETRAINED,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    return model


def _load_processor():
    from transformers import DetrImageProcessor
    return DetrImageProcessor.from_pretrained(PRETRAINED)


def _save_checkpoint(model, path: Path, *, epoch: int, map50: float, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "epoch": epoch,
        "map50": map50,
        "meta": meta,
    }, path)


def _load_checkpoint_into(model, path: Path) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("map50", 0.0))


# ── Train loop ──────────────────────────────────────────────────────────────


def _process_batch(processor, pil_imgs, hf_targets):
    """Use DetrImageProcessor to produce model inputs + COCO-format labels."""
    enc = processor(images=pil_imgs, annotations=hf_targets,
                    return_tensors="pt")
    return enc  # has pixel_values, pixel_mask, labels


def _train_one_epoch(model, processor, loader, optimizer, device, *,
                     log_every: int = 50):
    model.train()
    total_loss = 0.0
    seen = 0
    for i, (pil_imgs, hf_targets) in enumerate(loader):
        enc = _process_batch(processor, pil_imgs, hf_targets)
        pixel_values = enc["pixel_values"].to(device)
        pixel_mask = enc["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in enc["labels"]]
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask,
                        labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        seen += 1
        if (i + 1) % log_every == 0:
            print(f"    step {i+1}/{len(loader)}  loss={total_loss/max(seen,1):.4f}")
    return total_loss / max(seen, 1)


@torch.no_grad()
def _eval_map50(model, processor, loader, device) -> float:
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", iou_thresholds=[0.5])
    model.eval()
    for pil_imgs, hf_targets in loader:
        enc = _process_batch(processor, pil_imgs, hf_targets)
        pixel_values = enc["pixel_values"].to(device)
        pixel_mask = enc["pixel_mask"].to(device)
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        sizes = torch.tensor(
            [(im.size[1], im.size[0]) for im in pil_imgs],   # (h, w)
            device=device,
        )
        results = processor.post_process_object_detection(
            outputs, target_sizes=sizes, threshold=0.0,
        )

        preds, gts = [], []
        for res, tg, pil in zip(results, hf_targets, pil_imgs):
            preds.append({
                "boxes": res["boxes"].cpu(),
                "scores": res["scores"].cpu(),
                "labels": res["labels"].cpu(),
            })
            anns = tg["annotations"]
            if anns:
                gt_boxes = torch.tensor(
                    [[a["bbox"][0], a["bbox"][1],
                      a["bbox"][0] + a["bbox"][2],
                      a["bbox"][1] + a["bbox"][3]] for a in anns],
                    dtype=torch.float32,
                )
                gt_labels = torch.zeros(len(anns), dtype=torch.int64)
            else:
                gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
                gt_labels = torch.zeros((0,), dtype=torch.int64)
            gts.append({"boxes": gt_boxes, "labels": gt_labels})
        metric.update(preds, gts)
    result = metric.compute()
    return float(result.get("map_50", torch.tensor(0.0)).item())


# ── Adapter ──────────────────────────────────────────────────────────────────


class DETRAdapter(SpecialistAdapter):
    arch = "detr-r50"

    def train(
        self,
        condition: str,
        train_args: dict,
        output_dir: Path,
        weight: str,        # interpreted (default: PRETRAINED)
        *,
        resume: bool = False,
        device: str = "0",
    ) -> Path:
        dev = _resolve_device(device)

        ds_train = YoloDirDataset(condition, "train", "detr")
        ds_val = YoloDirDataset(condition, "val", "detr")
        if len(ds_train) < DETR_MIN_TRAIN_IMAGES:
            raise SkipArchitecture(
                f"detr-r50: {condition} has only {len(ds_train)} train images "
                f"(min {DETR_MIN_TRAIN_IMAGES}); DETR is data-hungry."
            )

        epochs = int(train_args.get("epochs", 80))
        batch = int(train_args.get("batch", 2))   # DETR is VRAM-heavy
        lr0 = float(train_args.get("lr0", 1e-4))
        weight_decay = float(train_args.get("weight_decay", 1e-4))
        warmup_epochs = int(train_args.get("warmup_epochs", 3))
        num_workers = int(train_args.get("num_workers", 2))

        loader_train = DataLoader(
            ds_train, batch_size=batch, shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_detr_pre_processor,
            persistent_workers=num_workers > 0,
        )
        loader_val = DataLoader(
            ds_val, batch_size=batch, shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_detr_pre_processor,
            persistent_workers=num_workers > 0,
        )

        processor = _load_processor()
        model = _load_detr(num_labels=1)
        start_epoch = 0
        if resume:
            last_ckpt = output_dir / "weights" / "last.pt"
            if last_ckpt.exists():
                start_epoch, _ = _load_checkpoint_into(model, last_ckpt)
                print(f"  resumed from epoch {start_epoch}")
        model.to(dev)

        # AdamW (plan §5.2 for transformer-style training).
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr0, weight_decay=weight_decay,
        )
        warmup_steps = max(warmup_epochs, 1)

        def lr_lambda(ep):
            if ep < warmup_steps:
                return (ep + 1) / (warmup_steps + 1)
            progress = (ep - warmup_steps) / max(epochs - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        run_dir = output_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        best_map = 0.0
        best_ckpt = run_dir / "weights" / "best.pt"
        last_ckpt = run_dir / "weights" / "last.pt"

        print(f"  DETR train: epochs={epochs} batch={batch} lr0={lr0} "
              f"wd={weight_decay} dev={dev}  train={len(ds_train)} val={len(ds_val)}")
        t0 = time.time()
        for epoch in range(start_epoch, epochs):
            print(f"\n  [epoch {epoch+1}/{epochs}] "
                  f"lr={optimizer.param_groups[0]['lr']:.5f}")
            train_loss = _train_one_epoch(model, processor, loader_train,
                                          optimizer, dev)
            map50 = _eval_map50(model, processor, loader_val, dev)
            print(f"    train_loss={train_loss:.4f}  val_mAP50={map50:.4f}")

            meta = {"arch": self.arch, "condition": condition,
                    "lr": optimizer.param_groups[0]["lr"]}
            _save_checkpoint(model, last_ckpt, epoch=epoch+1,
                             map50=map50, meta=meta)
            if map50 > best_map:
                best_map = map50
                _save_checkpoint(model, best_ckpt, epoch=epoch+1,
                                 map50=map50, meta=meta)
                print(f"    ✓ new best (mAP50={best_map:.4f})")
            scheduler.step()
        elapsed = (time.time() - t0) / 60
        print(f"\n  Training done in {elapsed:.1f} min. Best mAP50={best_map:.4f}")
        return best_ckpt

    # ── predict ──

    @torch.no_grad()
    def predict_batch(
        self,
        ckpt: Path,
        img_paths: list[Path],
        *,
        conf_min: float = 0.001,
        imgsz: int = 640,    # unused; HF processor handles resize
        batch: int = 4,
        device: str = "0",
    ) -> list[Prediction]:
        from PIL import Image

        dev = _resolve_device(device)
        processor = _load_processor()
        model = _load_detr(num_labels=1)
        _load_checkpoint_into(model, ckpt)
        model.to(dev).eval()

        outputs: list[Prediction] = []
        for i in range(0, len(img_paths), batch):
            chunk = img_paths[i:i + batch]
            pil_imgs = [Image.open(p).convert("RGB") for p in chunk]
            enc = processor(images=pil_imgs, return_tensors="pt")
            pixel_values = enc["pixel_values"].to(dev)
            pixel_mask = enc["pixel_mask"].to(dev)
            out = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            sizes = torch.tensor(
                [(im.size[1], im.size[0]) for im in pil_imgs], device=dev,
            )
            results = processor.post_process_object_detection(
                out, target_sizes=sizes, threshold=conf_min,
            )
            for res, pil in zip(results, pil_imgs):
                W, H = pil.size
                boxes = res["boxes"].cpu().numpy()
                scores = res["scores"].cpu().numpy()
                if len(boxes):
                    boxes_n = boxes / np.array([W, H, W, H], dtype=np.float32)
                else:
                    boxes_n = boxes.astype(np.float32)
                outputs.append(Prediction(
                    boxes_xyxyn=boxes_n.astype(np.float32),
                    scores=scores.astype(np.float32),
                    labels=np.zeros(len(boxes_n), dtype=np.int64),
                ))
        return outputs

    # ── mAP@0.5 ──

    def compute_map50(
        self,
        ckpt: Path,
        data_yaml: Path,
        split: str,
        *,
        imgsz: int = 640,
        batch: int = 2,
        device: str = "0",
    ) -> float:
        dev = _resolve_device(device)
        condition = data_yaml.parent.name
        ds = YoloDirDataset(condition, split, "detr")
        loader = DataLoader(
            ds, batch_size=batch, shuffle=False, num_workers=0,
            collate_fn=collate_detr_pre_processor,
        )
        processor = _load_processor()
        model = _load_detr(num_labels=1)
        _load_checkpoint_into(model, ckpt)
        model.to(dev)
        return _eval_map50(model, processor, loader, dev)

    # ── ONNX export ──

    def export_onnx(self, ckpt: Path, *, imgsz: int = 640) -> Path:
        """ONNX export for DETR is supported but requires careful handling
        of dynamic input sizes and the HF post-processing. We export the
        model's forward pass at a fixed imgsz; post-processing remains in
        PyTorch / Python at inference."""
        dev = torch.device("cpu")
        model = _load_detr(num_labels=1)
        _load_checkpoint_into(model, ckpt)
        model.to(dev).eval()
        dummy_px = torch.rand(1, 3, imgsz, imgsz)
        dummy_mask = torch.ones(1, imgsz, imgsz, dtype=torch.long)
        onnx_path = ckpt.with_suffix(".onnx")
        torch.onnx.export(
            model,
            (dummy_px, dummy_mask),
            str(onnx_path),
            opset_version=14,
            do_constant_folding=True,
            input_names=["pixel_values", "pixel_mask"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "pixel_mask": {0: "batch"},
            },
        )
        return onnx_path
