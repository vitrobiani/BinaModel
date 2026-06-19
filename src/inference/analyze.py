"""
src/inference/analyze.py
────────────────────────
Analyze a dental image using all trained specialist models.

Usage:
  # Analyze a single image
  python src/inference/analyze.py path/to/image.jpg

  # Analyze and save output (instead of displaying)
  python src/inference/analyze.py path/to/image.jpg --output result.jpg

  # Analyze multiple images
  python src/inference/analyze.py img1.jpg img2.jpg img3.jpg

  # Analyze all images in a directory
  python src/inference/analyze.py path/to/folder/ --output output_folder/

  # Adjust confidence threshold
  python src/inference/analyze.py image.jpg --conf 0.3

  # Use specific conditions only
  python src/inference/analyze.py image.jpg --conditions caries plaque
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PIPELINE_CFG = ROOT / "configs" / "pipeline.yaml"
CONDITIONS = ["caries", "gingivitis", "plaque", "discoloration", "ulcer", "recession"]

# Colors for each condition (BGR format)
COLORS = {
    "caries":        (0, 0, 255),      # Red
    "gingivitis":    (0, 165, 255),    # Orange
    "plaque":        (0, 255, 255),    # Yellow
    "discoloration": (255, 0, 255),    # Magenta
    "ulcer":         (255, 0, 0),      # Blue
    "recession":     (0, 255, 0),      # Green
}


def load_config() -> dict:
    """Load pipeline configuration."""
    with open(PIPELINE_CFG) as f:
        return yaml.safe_load(f)


def find_best_checkpoint(condition: str) -> Optional[Path]:
    """Find the best.pt checkpoint for a condition."""
    ckpt = ROOT / "runs" / "specialists" / f"specialist_{condition}" / "weights" / "best.pt"
    if ckpt.exists():
        return ckpt
    # Try last.pt as fallback
    last = ckpt.parent / "last.pt"
    if last.exists():
        return last
    return None


def load_specialists(conditions: list[str]) -> dict[str, YOLO]:
    """Load all available specialist models."""
    models = {}
    for cond in conditions:
        ckpt = find_best_checkpoint(cond)
        if ckpt:
            print(f"  Loading {cond}: {ckpt.name}")
            models[cond] = YOLO(str(ckpt))
        else:
            print(f"  Skipping {cond}: no checkpoint found")
    return models


def analyze_image(
    image_path: Path,
    models: dict[str, YOLO],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> tuple[np.ndarray, list[dict]]:
    """
    Run all specialist models on an image.

    Returns:
        annotated_image: Image with drawn bounding boxes
        detections: List of detection dicts with condition, confidence, bbox
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    annotated = img.copy()
    all_detections = []
    cfg = load_config()

    for condition, model in models.items():
        # Get condition-specific thresholds from config
        spec_cfg = cfg["specialists"].get(condition, {})
        conf = spec_cfg.get("conf_thresh", conf_threshold)
        iou = spec_cfg.get("iou_thresh", iou_threshold)

        # Run inference
        results = model.predict(
            img,
            conf=conf,
            iou=iou,
            verbose=False,
        )

        # Process detections
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])

                all_detections.append({
                    "condition": condition,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                })

                # Draw on image
                color = COLORS.get(condition, (255, 255, 255))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Label
                label = f"{condition}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(
                    annotated,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

    return annotated, all_detections


def print_report(detections: list[dict], image_path: Path) -> None:
    """Print a summary of detections."""
    print(f"\n{'─' * 50}")
    print(f"  Analysis: {image_path.name}")
    print(f"{'─' * 50}")

    if not detections:
        print("  No conditions detected.")
        return

    # Group by condition
    by_condition = {}
    for det in detections:
        cond = det["condition"]
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(det)

    for cond in CONDITIONS:
        if cond in by_condition:
            dets = by_condition[cond]
            avg_conf = sum(d["confidence"] for d in dets) / len(dets)
            print(f"  {cond:15} : {len(dets)} detection(s), avg conf: {avg_conf:.2f}")

    print(f"\n  Total detections: {len(detections)}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dental images with trained specialist models"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image file(s) or directory to analyze",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file or directory for annotated images",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=CONDITIONS,
        help="Conditions to check (default: all)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display results (just save/print)",
    )
    args = parser.parse_args()

    # Collect input images
    image_paths = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            image_paths.extend(p.glob("*.jpg"))
            image_paths.extend(p.glob("*.jpeg"))
            image_paths.extend(p.glob("*.png"))
        elif p.is_file():
            image_paths.append(p)
        else:
            print(f"Warning: {inp} not found, skipping")

    if not image_paths:
        print("No images found to analyze.")
        sys.exit(1)

    # Load models
    print("\nLoading specialist models...")
    models = load_specialists(args.conditions)

    if not models:
        print("\nNo trained models found!")
        print("Train models first with: python src/pipeline.py --phase train")
        sys.exit(1)

    print(f"\nLoaded {len(models)} model(s): {', '.join(models.keys())}")

    # Setup output
    output_dir = None
    if args.output:
        output_path = Path(args.output)
        if len(image_paths) > 1 or output_path.suffix == "":
            output_dir = output_path
            output_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    for img_path in image_paths:
        try:
            annotated, detections = analyze_image(
                img_path,
                models,
                conf_threshold=args.conf,
            )
            print_report(detections, img_path)

            # Save or display
            if args.output:
                if output_dir:
                    out_path = output_dir / f"analyzed_{img_path.name}"
                else:
                    out_path = Path(args.output)
                cv2.imwrite(str(out_path), annotated)
                print(f"  Saved: {out_path}")
            elif not args.no_display:
                cv2.imshow(f"Analysis: {img_path.name}", annotated)
                print("\n  Press any key to continue, 'q' to quit...")
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
                if key == ord('q'):
                    break

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
