# python
from pathlib import Path
import shutil

def normalize_yolo_labels(labels_dir: Path) -> int:
    """
    Ensure each label line has exactly 5 columns: class cx cy w h.
    If a line has 6+ tokens, keep only the first 5. Lines with 5 are unchanged.
    Returns the number of files modified.
    """
    if not labels_dir.exists():
        raise FileNotFoundError(f'Labels directory not found: {labels_dir}')

    modified_files = 0
    for lbl_file in sorted(labels_dir.glob('*.txt')):
        lines = lbl_file.read_text(encoding='utf-8').splitlines()
        changed = False
        out_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                out_lines.append(stripped)
                continue

            parts = stripped.split()
            if len(parts) >= 6:
                kept = ' '.join(parts[:5])
                if kept != stripped:
                    changed = True
                out_lines.append(kept)
            else:
                out_lines.append(stripped)

        if changed:
            lbl_file.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
            modified_files += 1

    return modified_files


def write_to_save_list(
    images_dir: Path = Path('new_images'),
    to_save_file: Path = Path('to_save.txt')
) -> int:
    """
    Write all *.jpg base names (without extension) from images_dir into to_save.txt.
    Overwrites the file on each run. Returns the number of entries written.
    """
    if not images_dir.exists():
        raise FileNotFoundError(f'Images directory not found: {images_dir}')

    jpgs = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() == '.jpg')
    names = [p.stem for p in jpgs]  # base names without '.jpg'
    to_save_file.write_text('\n'.join(names) + ('\n' if names else ''), encoding='utf-8')
    return len(names)


def build_cvat_bundle(
    labels_dir=Path('new_images/labels'),
    cvat_dir=Path('cvat_labels'),
    images_prefix=Path('data/images/train'),
    image_ext='.jpg',
    normalize_labels=True,
    images_dir=Path('new_images'),
    to_save_file=Path('to_save.txt'),
    write_to_save=True
):
    if not labels_dir.exists():
        raise FileNotFoundError(f'Labels directory not found: {labels_dir}')

    # Optional: normalize labels in-place before packaging
    if normalize_labels:
        changed = normalize_yolo_labels(labels_dir)
        print(f'Normalized {changed} label file(s) in {labels_dir}')

    # Optional: write image names to to_save.txt
    if write_to_save:
        written = write_to_save_list(images_dir=images_dir, to_save_file=to_save_file)
        print(f'Wrote {written} image name(s) to {to_save_file}')

    # Recreate cvat_labels directory
    if cvat_dir.exists():
        shutil.rmtree(cvat_dir)
    cvat_dir.mkdir(parents=True, exist_ok=True)

    # Prepare labels subdirs: cvat_labels/labels/train
    cvat_labels_subdir_labels = cvat_dir / 'labels'
    cvat_labels_subdir_labels.mkdir(parents=True, exist_ok=True)
    cvat_labels_subdir = cvat_labels_subdir_labels / 'train'
    cvat_labels_subdir.mkdir(parents=True, exist_ok=True)

    # Collect label files
    label_files = sorted(labels_dir.glob('*.txt'))

    # Build train.txt lines from label stems
    train_lines = []
    for lbl in label_files:
        name = lbl.stem
        img_path = (images_prefix / f'{name}{image_ext}').as_posix()
        train_lines.append(img_path)

    # Write train.txt inside cvat_labels
    train_txt_path = cvat_dir / 'train.txt'
    train_txt_path.write_text('\n'.join(train_lines) + '\n', encoding='utf-8')

    # Write data.yaml inside cvat_labels
    data_yaml_path = cvat_dir / 'data.yaml'
    data_yaml_content = (
        "names:\n"
        "  0: Caries\n"
        "  1: Ulcer\n"
        "  2: Tooth Discoloration\n"
        "  3: Gingivitis\n"
        "path: .\n"
        "train: train.txt\n"
    )
    data_yaml_path.write_text(data_yaml_content, encoding='utf-8')

    # Copy label files to cvat_labels/labels/train
    for lbl in label_files:
        shutil.copy2(lbl, cvat_labels_subdir / lbl.name)

    print(f'Wrote {len(train_lines)} entries to {train_txt_path}')
    print(f'Wrote YAML to {data_yaml_path}')
    print(f'Copied {len(label_files)} label files to {cvat_labels_subdir}')


if __name__ == '__main__':
    build_cvat_bundle()