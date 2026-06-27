"""
Quick visual sanity-check: render pseudo-labels as boxes on the unlabeled
images so you can eyeball whether the specialists are firing on the right
regions. Outputs to data/pseudo_labeled/viz/. Run from project root:

    python viz_pseudo.py            # 20 random images
    python viz_pseudo.py --n 50     # 50 random images
"""
import argparse
import random
from pathlib import Path

import cv2

NAMES = ["caries", "gingivitis", "plaque", "discoloration", "ulcer", "recession"]
# BGR for cv2 — distinct, high-contrast.
COLORS = [
    (0, 255, 0),      # caries — green
    (0, 165, 255),    # gingivitis — orange
    (255, 255, 0),    # plaque — cyan
    (0, 0, 255),      # discoloration — red (likely wrong — eyeball these!)
    (255, 0, 255),    # ulcer — magenta
    (255, 0, 0),      # recession — blue
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--src", default="data/pseudo_labeled")
    args = parser.parse_args()

    src = Path(args.src)
    out = src / "viz"
    out.mkdir(exist_ok=True)
    imgs = sorted(p for p in (src / "images").iterdir() if p.is_file())
    random.seed(args.seed)
    sample = random.sample(imgs, min(args.n, len(imgs)))

    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl = src / "labels" / f"{p.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = int(parts[0])
                cx, cy, bw, bh = (float(parts[i]) for i in range(1, 5))
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                color = COLORS[c]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, NAMES[c], (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(str(out / p.name), img)

    print(f"wrote {len(sample)} annotated images to {out.resolve()}")


if __name__ == "__main__":
    main()
