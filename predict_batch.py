"""
Run the student model on N random test images, save annotated outputs as
test1.jpg, test2.jpg, ... in runs/sanity/batch/. Bypasses Ultralytics' weird
auto-save path mangling by drawing + saving the result manually.

Usage:
    python predict_batch.py            # 10 random test images
    python predict_batch.py --n 20     # 20 random test images
    python predict_batch.py --conf 0.1 # lower confidence threshold
    python predict_batch.py --src data/unlabeled  # different source dir
"""
import argparse
import random
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10,
                        help="number of random images")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="confidence threshold")
    parser.add_argument("--src", default="data/student/images/test",
                        help="source dir to pick images from")
    parser.add_argument("--model",
                        default="runs/student/bina_v1/weights/best.pt")
    parser.add_argument("--out", default="runs/sanity/batch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-classes", type=int, default=1,
                        help="only save images where the model detects at "
                             "least this many distinct classes (e.g. --min-classes 2 "
                             "to find images with caries+plaque or similar)")
    parser.add_argument("--scan", type=int, default=None,
                        help="when --min-classes>1: how many candidate images "
                             "to scan to find --n multi-class hits (default: 10x --n)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale tests from prior runs so file count = current run.
    for old in out_dir.glob("test*.jpg"):
        old.unlink()

    src_dir = Path(args.src)
    imgs = sorted(p for p in src_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not imgs:
        raise SystemExit(f"no images found in {src_dir}")

    random.seed(args.seed)
    random.shuffle(imgs)

    model = YOLO(args.model)

    # If user wants multi-class images, scan more candidates than they
    # asked to save; otherwise just take the first --n.
    if args.min_classes > 1:
        scan_budget = args.scan if args.scan else args.n * 10
        candidates = imgs[:min(scan_budget, len(imgs))]
        print(f"\n  scanning {len(candidates)} candidate images for "
              f"≥{args.min_classes} distinct classes at conf>={args.conf}\n")
    else:
        candidates = imgs[:args.n]
        print(f"\n  predicting {len(candidates)} images at conf>={args.conf}\n")

    print(f"  {'#':<4} {'detections':<24} source")
    print("  " + "-" * 80)

    saved = 0
    for src in candidates:
        results = model.predict(source=str(src), imgsz=640, conf=args.conf,
                                verbose=False)
        n_det = len(results[0].boxes) if results[0].boxes is not None else 0
        if n_det == 0:
            continue
        names = results[0].names
        classes = [int(c) for c in results[0].boxes.cls.cpu().numpy()]
        unique_classes = set(classes)
        if len(unique_classes) < args.min_classes:
            continue
        # Ultralytics' .plot() returns a numpy BGR image with boxes + labels
        # already drawn — saves us writing our own renderer.
        annotated = results[0].plot()
        saved += 1
        out_path = out_dir / f"test{saved}.jpg"
        cv2.imwrite(str(out_path), annotated)
        counts = {names[c]: classes.count(c) for c in unique_classes}
        det_str = " ".join(f"{k}:{v}" for k, v in counts.items())
        print(f"  test{saved:<3} {det_str[:24]:<24} {src.name}")
        if saved >= args.n:
            break

    if saved == 0:
        print(f"\n  no images matched --min-classes={args.min_classes}. Try "
              f"lowering --conf or raising --scan.")
    else:
        print(f"\n  saved {saved}/{args.n} images → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
