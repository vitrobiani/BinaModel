"""
SpecialistAdapter dispatch by architecture slug.

The Phase-1 sweep treats every architecture (Ultralytics YOLO/RT-DETR,
torchvision Faster R-CNN, HuggingFace DETR) uniformly through this interface.
get_adapter() routes by slug; adapter modules are imported lazily so users
without `transformers` (or without `torchvision`) can still use the Ultralytics
path without missing-dependency errors.
"""
from __future__ import annotations

from .base import SpecialistAdapter, Prediction

_ULTRALYTICS_PREFIXES = ("yolo", "rtdetr")
_TORCHVISION_SLUGS = {"frcnn-r50"}
_HF_DETR_SLUGS = {"detr-r50"}


def get_adapter(arch: str) -> SpecialistAdapter:
    if any(arch.startswith(p) for p in _ULTRALYTICS_PREFIXES):
        from .ultralytics_adapter import UltralyticsAdapter
        return UltralyticsAdapter(arch)
    if arch in _TORCHVISION_SLUGS:
        from .torchvision_adapter import FasterRCNNAdapter
        return FasterRCNNAdapter()
    if arch in _HF_DETR_SLUGS:
        from .hf_detr_adapter import DETRAdapter
        return DETRAdapter()
    raise ValueError(
        f"unknown architecture: {arch!r}. "
        f"Known: yolo*, rtdetr*, {_TORCHVISION_SLUGS}, {_HF_DETR_SLUGS}"
    )


def list_known_archs() -> list[str]:
    """For docs/argparse choices."""
    return ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
            "rtdetr-l", "rtdetr-x", "frcnn-r50", "detr-r50"]


__all__ = ["SpecialistAdapter", "Prediction", "get_adapter", "list_known_archs"]
