"""
Filter pseudo-labels by class id. Strips boxes for classes whose specialists
are too weak to trust (default: discoloration=3) and writes a clean copy at
data/pseudo_labeled_clean/. Images with no remaining boxes are still copied —
Ultralytics treats them as 'negative examples' which is healthy training signal.

Usage:
    python filter_pseudo.py                  # drop class 3 (discoloration)
    python filter_pseudo.py --drop 3 5       # drop discoloration AND recession
"""
import argparse
import shutil
from pathlib import Path


NAMES = ["caries", "gingivitis", "plaque", "discoloration", "ulcer", "recession"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/pseudo_labeled")
    parser.add_argument("--dst", default="data/pseudo_labeled_clean")
    parser.add_argument("--drop", type=int, nargs="+", default=[3],
                        help="class ids to remove from labels (default: 3 = discoloration)")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    drop = set(args.drop)

    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)

    # Per-class counters
    kept = {i: 0 for i in range(6)}
    dropped = {i: 0 for i in range(6)}
    images_copied = 0
    images_with_labels = 0

    for img_path in sorted((src / "images").iterdir()):
        if not img_path.is_file():
            continue
        shutil.copy2(img_path, dst / "images" / img_path.name)
        images_copied += 1
        lbl_path = src / "labels" / f"{img_path.stem}.txt"
        out_lines = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(parts[0])
                except ValueError:
                    continue
                if cls in drop:
                    dropped[cls] += 1
                else:
                    kept[cls] += 1
                    out_lines.append(line)
        # Always write a label file, even if empty (counts as "negative" image).
        (dst / "labels" / f"{img_path.stem}.txt").write_text(
            "\n".join(out_lines) + ("\n" if out_lines else "")
        )
        if out_lines:
            images_with_labels += 1

    print(f"\n  source:       {src.resolve()}")
    print(f"  output:       {dst.resolve()}")
    print(f"  images:       {images_copied}  ({images_with_labels} with boxes, "
          f"{images_copied - images_with_labels} negatives)")
    print(f"  dropped:      {sorted(drop)}")
    print("")
    print("  class            kept    dropped")
    print("  " + "-" * 36)
    for i, name in enumerate(NAMES):
        marker = " (dropped)" if i in drop else ""
        print(f"  {i} {name:<14}  {kept[i]:>4}    {dropped[i]:>4}{marker}")
    total_kept = sum(kept.values())
    total_dropped = sum(dropped.values())
    print("  " + "-" * 36)
    print(f"  TOTAL          {total_kept:>5}   {total_dropped:>5}")


if __name__ == "__main__":
    main()
