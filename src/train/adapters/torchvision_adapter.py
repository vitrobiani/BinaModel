"""
FasterRCNNAdapter — Faster R-CNN ResNet-50-FPN via torchvision.

The plan (§2.1) explicitly calls for ResNet-50 + FPN with COCO/ImageNet
pretraining. We replace the box predictor with a 2-class head (background +
single target) and fine-tune end-to-end.

Known deviations from the plan, deliberate:
  - Focal Loss (§2.2) is not implemented; torchvision's stock RoI sampling
    is used. Adding Focal Loss requires monkey-patching `fastrcnn_loss` in
    torchvision and is deferred until we see whether it actually moves the
    needle on KPI metrics.
  - Progressive unfreezing (§2.1) is not implemented; we fine-tune all
    parameters from the start at a single LR (HPO picks). Easier to compare
    architectures apples-to-apples without per-arch unfreezing logic.

These are good targets if Phase 1 results show FRCNN underperforming.
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

from train.yolo_dataset import YoloDirDataset, collate_frcnn  # noqa: E402


def _build_model(num_classes: int = 2, pretrained: bool = True):
    """num_classes = background + target. For single-class specialists → 2."""
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _resolve_device(device: str) -> torch.device:
    """Accept Ultralytics-style device strings ('0', 'cpu', 'cuda:0', ...)."""
    if device in ("cpu",):
        return torch.device("cpu")
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def _save_checkpoint(model, path: Path, *, epoch: int, map50: float, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "epoch": epoch,
        "map50": map50,
        "meta": meta,
    }, path)


def _load_checkpoint(model, path: Path) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("map50", 0.0))


def _to_device(tensor_list, device):
    return [t.to(device) for t in tensor_list]


def _targets_to_device(targets, device):
    return [{k: v.to(device) if torch.is_tensor(v) else v
             for k, v in t.items()} for t in targets]


# ── Train ────────────────────────────────────────────────────────────────────


def _train_one_epoch(model, loader, optimizer, device, *, log_every: int = 50):
    model.train()
    running = 0.0
    seen = 0
    for i, (imgs, targets) in enumerate(loader):
        imgs = _to_device(imgs, device)
        targets = _targets_to_device(targets, device)
        # Skip empty-target images (FRCNN errors on them)
        ok_pairs = [(im, tg) for im, tg in zip(imgs, targets)
                    if tg["boxes"].numel() > 0]
        if not ok_pairs:
            continue
        imgs_ok, targets_ok = zip(*ok_pairs)
        loss_dict = model(list(imgs_ok), list(targets_ok))
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        running += float(loss.detach().cpu())
        seen += 1
        if (i + 1) % log_every == 0:
            print(f"    step {i+1}/{len(loader)}  loss={running/max(seen,1):.4f}")
    return running / max(seen, 1)


@torch.no_grad()
def _eval_map50(model, loader, device) -> float:
    """mAP@0.5 via torchmetrics."""
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", iou_thresholds=[0.5],
                                  class_metrics=False)
    model.eval()
    for imgs, targets in loader:
        imgs = _to_device(imgs, device)
        outputs = model(imgs)
        preds = []
        gts = []
        for out, tg in zip(outputs, targets):
            preds.append({
                "boxes": out["boxes"].cpu(),
                "scores": out["scores"].cpu(),
                "labels": out["labels"].cpu(),
            })
            gts.append({
                "boxes": tg["boxes"].cpu(),
                "labels": tg["labels"].cpu(),
            })
        metric.update(preds, gts)
    result = metric.compute()
    return float(result.get("map_50", torch.tensor(0.0)).item())


# ── Adapter ──────────────────────────────────────────────────────────────────


class FasterRCNNAdapter(SpecialistAdapter):
    arch = "frcnn-r50"

    # ── train ──

    def train(
        self,
        condition: str,
        train_args: dict,
        output_dir: Path,
        weight: str,  # interpreted as "pretrained-coco" / path to .pt resume
        *,
        resume: bool = False,
        device: str = "0",
    ) -> Path:
        dev = _resolve_device(device)
        epochs = int(train_args.get("epochs", 80))
        batch = int(train_args.get("batch", 4))   # FRCNN is heavy; default 4
        lr0 = float(train_args.get("lr0", 0.005))  # plan §5.2: SGD-style LR
        momentum = float(train_args.get("momentum", 0.937))
        weight_decay = float(train_args.get("weight_decay", 0.0005))
        warmup_epochs = int(train_args.get("warmup_epochs", 3))
        num_workers = int(train_args.get("num_workers", 2))

        ds_train = YoloDirDataset(condition, "train", "frcnn")
        ds_val = YoloDirDataset(condition, "val", "frcnn")
        loader_train = DataLoader(
            ds_train, batch_size=batch, shuffle=True,
            num_workers=num_workers, collate_fn=collate_frcnn,
            persistent_workers=num_workers > 0,
        )
        loader_val = DataLoader(
            ds_val, batch_size=1, shuffle=False,
            num_workers=num_workers, collate_fn=collate_frcnn,
            persistent_workers=num_workers > 0,
        )

        model = _build_model(num_classes=2, pretrained=True)
        start_epoch = 0
        if resume:
            last_ckpt = output_dir / "weights" / "last.pt"
            if last_ckpt.exists():
                start_epoch, _ = _load_checkpoint(model, last_ckpt)
                print(f"  resumed from epoch {start_epoch}")
        model.to(dev)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=lr0, momentum=momentum, weight_decay=weight_decay,
        )
        # Linear warmup → cosine annealing (plan §5.2).
        total_steps = max(epochs, 1)
        warmup_steps = max(warmup_epochs, 1)

        def lr_lambda(ep):
            if ep < warmup_steps:
                return (ep + 1) / (warmup_steps + 1)
            progress = (ep - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        run_dir = output_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        best_map = 0.0
        best_ckpt = run_dir / "weights" / "best.pt"
        last_ckpt = run_dir / "weights" / "last.pt"

        print(f"  Faster R-CNN train: epochs={epochs} batch={batch} "
              f"lr0={lr0} momentum={momentum} wd={weight_decay} dev={dev}")
        t0 = time.time()
        for epoch in range(start_epoch, epochs):
            print(f"\n  [epoch {epoch+1}/{epochs}] lr={optimizer.param_groups[0]['lr']:.5f}")
            train_loss = _train_one_epoch(model, loader_train, optimizer, dev)
            map50 = _eval_map50(model, loader_val, dev)
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
        imgsz: int = 640,    # unused — FRCNN handles native resolution
        batch: int = 4,
        device: str = "0",
    ) -> list[Prediction]:
        import cv2

        dev = _resolve_device(device)
        model = _build_model(num_classes=2, pretrained=False)
        _load_checkpoint(model, ckpt)
        model.to(dev).eval()

        outputs_all: list[Prediction] = []
        for i in range(0, len(img_paths), batch):
            chunk = img_paths[i:i + batch]
            imgs = []
            sizes = []
            for p in chunk:
                bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if bgr is None:
                    imgs.append(torch.zeros(3, 32, 32))
                    sizes.append((32, 32))
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                imgs.append(
                    torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                )
                sizes.append((h, w))
            imgs = [im.to(dev) for im in imgs]
            outs = model(imgs)
            for out, (h, w) in zip(outs, sizes):
                boxes = out["boxes"].cpu().numpy()
                scores = out["scores"].cpu().numpy()
                labels = out["labels"].cpu().numpy()
                keep = scores >= conf_min
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                if len(boxes):
                    # absolute xyxy → normalized xyxy
                    boxes_n = boxes / np.array([w, h, w, h], dtype=np.float32)
                else:
                    boxes_n = boxes.astype(np.float32)
                # torchvision uses label 1 for foreground; remap to 0 for the
                # rest of the pipeline (single-class).
                labels0 = np.zeros_like(labels)
                outputs_all.append(Prediction(
                    boxes_xyxyn=boxes_n.astype(np.float32),
                    scores=scores.astype(np.float32),
                    labels=labels0.astype(np.int64),
                ))
        return outputs_all

    # ── mAP@0.5 ──

    def compute_map50(
        self,
        ckpt: Path,
        data_yaml: Path,           # unused; condition encoded via dirname
        split: str,
        *,
        imgsz: int = 640,
        batch: int = 4,
        device: str = "0",
    ) -> float:
        dev = _resolve_device(device)
        # data_yaml.parent.name == condition (data/processed/<cond>/dataset.yaml)
        condition = data_yaml.parent.name
        ds = YoloDirDataset(condition, split, "frcnn")
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=collate_frcnn)
        model = _build_model(num_classes=2, pretrained=False)
        _load_checkpoint(model, ckpt)
        model.to(dev)
        return _eval_map50(model, loader, dev)

    # ── ONNX export ──

    def export_onnx(self, ckpt: Path, *, imgsz: int = 640) -> Path:
        """ONNX export for torchvision Faster R-CNN is supported via opset>=11
        with caveats (NMS plugins). We export and let onnxruntime sort it out;
        if the resulting graph fails to load, prefer a TorchScript or
        ONNX with a static-shape input from a wrapper module."""
        dev = torch.device("cpu")
        model = _build_model(num_classes=2, pretrained=False)
        _load_checkpoint(model, ckpt)
        model.to(dev).eval()
        dummy = torch.rand(1, 3, imgsz, imgsz)
        onnx_path = ckpt.with_suffix(".onnx")
        torch.onnx.export(
            model, [dummy], str(onnx_path),
            opset_version=16,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["detections"],
            dynamic_axes={"images": {0: "batch", 2: "h", 3: "w"}},
        )
        return onnx_path
