# Bina Pipeline — Dental Diagnosis Multi-Model Training System

A 4-phase pipeline that trains 6 specialist YOLO models (one per dental condition),
runs them over unlabeled RGB intraoral images to generate pseudo-annotations,
then trains a final unified multi-class student model.

```
BinaDatasets/ (7 source datasets)
     │
     ▼
┌─────────────┐
│  normalize  │  → data/processed/<condition>/  (unified YOLO format)
└─────────────┘
     │
     ▼
┌──────────────────────────────────────────────────┐
│  6 Specialist YOLO Models (trained separately)   │
│  caries · gingivitis · plaque · discoloration    │
│  ulcer  · recession                              │
└──────────────────────────────────────────────────┘
     │
     ▼  (run over unlabeled pool)
┌──────────────────────────────────────────────────┐
│  Ensemble Inference                              │
│  → Intra-class NMS                               │
│  → Cross-model deduplication (IoU-based)         │
│  → Global confidence filter                      │
└──────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────┐
│  Student Model (YOLO26m, 6-class)                │
│  Real data (oversampled) + Pseudo labels         │
└──────────────────────────────────────────────────┘
```

---

## Table of Contents

- [Setup](#setup)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Running the Pipeline](#running-the-pipeline)
- [Specialist Architecture Sweep](#specialist-architecture-sweep)
- [Analyzing Images](#analyzing-images)
- [Adding New Data](#adding-new-data)
- [Model Checkpoints](#model-checkpoints)
- [Configuration](#configuration)
- [Tuning the Merge Pipeline](#tuning-the-merge-pipeline)
- [Key Design Decisions](#key-design-decisions)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)

---

## Setup

```bash
# NixOS: packages are managed via configuration
# Required in python.nix: torch, opencv-python, ultralytics, tqdm, pyyaml

# Other systems:
pip install -r requirements.txt
```

**Required packages:** `ultralytics`, `torch`, `torchvision`, `opencv-python`, `numpy`, `PyYAML`, `tqdm`

---

## Quick Start

```bash
# 1. Normalize all datasets (converts to YOLO format)
python src/data/normalize.py --condition all

# 2. Train all specialist models
python src/pipeline.py --phase train

# 3. Analyze a dental image
python src/inference/analyze.py path/to/dental_photo.jpg
```

---

## Datasets

Source datasets are stored in `BinaDatasets/` and normalized into `data/processed/`.

### Current Dataset Summary

| Condition | Sources | Train | Val | Test | Total |
|-----------|---------|-------|-----|------|-------|
| **caries** | Carries_Dataset + Carries_abrasion (classes 3-8) + extensive (class 0) | 2,932 | 587 | 393 | 3,912 |
| **gingivitis** | Gingivites_Dataset (classes 3,4) + extensive (class 3) | 818 | 164 | 111 | 1,093 |
| **plaque** | mendeley-dataset + CALCULUS_Dataset | 4,091 | 819 | 547 | 5,457 |
| **discoloration** | extensive (class 2) | 137 | 28 | 20 | 185 |
| **ulcer** | extensive (class 1) | 196 | 40 | 27 | 263 |
| **recession** | gum_recession_dataset (class 0) + Spot (class 5 polygons) + big_gum_dataset (class 1 receding_gum) | ~1,470 | ~294 | ~196 | ~1,961 |

### Source Dataset Locations

```
BinaDatasets/
├── Carries_Dataset/              # YOLO format, classes 0,1
├── Carries_abrasion_restoration_dataset/  # YOLO polygon format
├── Gingivites_Dataset/           # YOLO format with Labels/ folder
├── mendeley-dataset-materials_Part_2/     # Custom 6-value format
├── CALCULUS_Dataset/             # YOLO format from Roboflow
├── gum_recession_dataset/        # YOLO format
├── Spot/                         # YOLO polygon (9 classes; class 5 = recession source)
├── big_gum_dataset/              # YOLO bbox (2 classes; class 1 receding_gum = recession source)
└── extensive_dataset/            # Multi-class YOLO (0=caries, 1=ulcer, 2=discoloration, 3=gingivitis)
```

### Format Conversions (handled by normalize.py)

| Dataset | Original Format | Conversion |
|---------|-----------------|------------|
| Carries_Dataset | YOLO bbox | Class remap (0,1 → 0) |
| Carries_abrasion | YOLO polygon | Polygon → bbox, classes 3-8 → 0 |
| Gingivites_Dataset | YOLO bbox | Classes 3,4 → 0 |
| mendeley-dataset | `flag cx cy w h tooth_id` | Keep flag=1, convert to YOLO |
| CALCULUS_Dataset | YOLO bbox | Direct copy (class 0) |
| gum_recession | YOLO bbox | Class 0 only + augmentation |
| Spot | YOLO polygon | Polygon → bbox, keep only class 5 ("Caries 5 class") → 0, prefix `spot_` |
| big_gum_dataset | YOLO bbox | Keep only class 1 (`receding_gum`) → 0, drop `diseased_gum`, prefix `big_gum_` |
| extensive | YOLO bbox | Filter by class per condition |

---

## Dataset Download (if needed)

See each `configs/models/<condition>.yaml` for the exact sources.
Quick start with Roboflow:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace().project("oral-dis")
dataset = project.version(2).download("yolov8", location="data/raw/discoloration/oral_dis_roboflow")
```

For Kaggle datasets:
```bash
kaggle datasets download bavithravairam/oral-ulcer -p data/raw/ulcer/kaggle_oral_ulcer --unzip
```

Drop any unlabeled intraoral RGB images into `data/unlabeled/`.

---

## Running the Pipeline

```bash
# Full end-to-end
python src/pipeline.py --phase all

# Individual phases
python src/pipeline.py --phase normalize
python src/pipeline.py --phase train
python src/pipeline.py --phase pseudo
python src/pipeline.py --phase student

# Plaque with domain-shift augmentation
python src/pipeline.py --phase normalize --conditions plaque --domain-shift

# Train only two conditions
python src/pipeline.py --phase train --conditions caries gingivitis

# Resume interrupted training
python src/pipeline.py --phase train --resume
```

---

## Specialist Architecture Sweep

Phase 1 of the multi-model training plan picks the best specialist architecture
per condition by running every candidate through the same HPO → train →
threshold → KPI gate pipeline. Five architectures are wired up:

| Slug | Family | Adapter | Notes |
|---|---|---|---|
| `yolo26s` | Ultralytics YOLO | `UltralyticsAdapter` | Baseline (current default). |
| `yolo26x` | Ultralytics YOLO | `UltralyticsAdapter` | Heavy YOLO sanity-check. |
| `rtdetr-l` | Ultralytics RT-DETR | `UltralyticsAdapter` | Transformer real-time. |
| `frcnn-r50` | torchvision | `FasterRCNNAdapter` | Faster R-CNN ResNet-50-FPN (plan §2.1). |
| `detr-r50` | HuggingFace transformers | `DETRAdapter` | Pure transformer. Auto-skipped on conditions with <1000 train images (currently discoloration and ulcer). |

### Sweep flow

```bash
# Full 5-arch × 6-condition sweep (HEAVY — multi-day on a single GPU)
python src/pipeline.py --phase sweep

# Restrict scope (recommended first run)
python src/pipeline.py --phase sweep --archs yolo26s --conditions recession \
                                    --hpo-trials 3 --hpo-epochs 3

# After the sweep, pick the per-condition winner (must clear KPI gate)
python src/pipeline.py --phase promote

# Downstream phases then read winners from runs/specialists/specialist_<c>/
python src/pipeline.py --phase pseudo
python src/pipeline.py --phase student
python src/pipeline.py --phase export
```

Each candidate's artifacts live under `runs/sweep/<arch>/specialist_<cond>/`
(`weights/best.pt`, `threshold.json`, `kpi_gate.json`). The sweep writes
`runs/sweep/results.json` incrementally after every pair, so the run is
safely interruptible. `promote.py` reads that file, picks the highest-mAP@0.5
candidate per condition that passed the KPI gate, and copies its files into
the canonical `runs/specialists/specialist_<cond>/` location.

Per-arch HPO results live under `runs/hpo/<arch>/<cond>_best.json` and are
picked up automatically by `train_specialist` for the matching `(arch, cond)`
run.

---

## Analyzing Images

Use trained models to analyze dental photos:

```bash
# Analyze single image (opens display window)
python src/inference/analyze.py photo.jpg

# Save annotated result
python src/inference/analyze.py photo.jpg --output result.jpg

# Analyze multiple images
python src/inference/analyze.py img1.jpg img2.jpg img3.jpg --output results/

# Analyze entire folder
python src/inference/analyze.py photos/ --output analyzed/

# Only use specific models
python src/inference/analyze.py photo.jpg --conditions caries plaque gingivitis

# Adjust confidence threshold (default: 0.25)
python src/inference/analyze.py photo.jpg --conf 0.3

# No display, just print results
python src/inference/analyze.py photo.jpg --no-display
```

### Detection Colors

| Color | Condition |
|-------|-----------|
| Red | Caries |
| Orange | Gingivitis |
| Yellow | Plaque |
| Magenta | Discoloration |
| Blue | Ulcer |
| Green | Recession |

### Sample Output

```
──────────────────────────────────────────────────
  Analysis: dental_photo.jpg
──────────────────────────────────────────────────
  caries          : 2 detection(s), avg conf: 0.78
  plaque          : 1 detection(s), avg conf: 0.65
  gingivitis      : 3 detection(s), avg conf: 0.71

  Total detections: 6
```

---

## Adding New Data

### Option 1: Add Directly to Processed Data (Quick)

```bash
# Copy image to training set
cp new_image.jpg data/processed/caries/images/train/

# Create matching label file (YOLO format)
echo "0 0.45 0.32 0.12 0.08" > data/processed/caries/labels/train/new_image.txt
```

### Option 2: Add to Source Datasets (Recommended for bulk)

```bash
# 1. Create custom folder
mkdir -p BinaDatasets/custom_caries/images
mkdir -p BinaDatasets/custom_caries/labels

# 2. Add your images and YOLO labels
cp my_images/*.jpg BinaDatasets/custom_caries/images/
cp my_labels/*.txt BinaDatasets/custom_caries/labels/

# 3. Edit src/data/normalize.py to include your folder:
#    Add to normalize_caries():
#      custom_dir = BINA_DATASETS_DIR / "custom_caries"
#      if custom_dir.exists():
#          n = copy_yolo_with_remap(...)

# 4. Re-run normalization
python src/data/normalize.py --condition caries
```

### YOLO Label Format

Each image needs a `.txt` file with the same name:

```
<class_id> <cx> <cy> <width> <height>
```

- `class_id`: Always `0` for specialist models
- `cx`, `cy`: Bounding box center (normalized 0-1)
- `width`, `height`: Bounding box size (normalized 0-1)

**Example:** For a 640x480 image with a box at pixels (100,120) to (200,180):
```
0 0.234375 0.3125 0.15625 0.125
```

### Labeling Tools

- [LabelImg](https://github.com/heartexlabs/labelImg) — outputs YOLO format directly
- [CVAT](https://cvat.ai/) — export as YOLO
- [Roboflow](https://roboflow.com/) — annotate and export

### After Adding Data

```bash
# Re-train the model
python src/pipeline.py --phase train --conditions caries

# Or resume from checkpoint
python src/pipeline.py --phase train --conditions caries --resume
```

---

## Model Checkpoints

Trained models are saved as PyTorch `.pt` files:

```
runs/specialists/
├── specialist_caries/
│   └── weights/
│       ├── best.pt      # Best validation mAP
│       └── last.pt      # Latest epoch
├── specialist_gingivitis/
│   └── weights/
│       ├── best.pt
│       └── last.pt
├── specialist_plaque/
├── specialist_discoloration/
├── specialist_ulcer/
└── specialist_recession/
```

### Using Checkpoints Directly

```python
from ultralytics import YOLO

# Load a specialist model
model = YOLO("runs/specialists/specialist_caries/weights/best.pt")

# Run inference
results = model.predict("dental_photo.jpg", conf=0.4)

# Access detections
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    confidence = box.conf[0]
    print(f"Caries detected: conf={confidence:.2f}")
```

---

## Configuration

### Main Config: `configs/pipeline.yaml`

```yaml
project:
  device: "0"           # GPU index, or "cpu"

train:
  epochs: 80
  imgsz: 640
  batch: 16             # Reduce to 8 or 4 if GPU memory is limited
  optimizer: AdamW
  lr0: 0.001

specialists:
  caries:
    config: configs/models/caries.yaml
    weight: yolo11s.pt
    conf_thresh: 0.40   # Inference threshold
    iou_thresh: 0.45    # NMS threshold
```

### Per-Condition Configs: `configs/models/<condition>.yaml`

```yaml
path: ../../data/processed/caries
train: images/train
val: images/val
test: images/test

nc: 1
names: ["caries"]

overrides:
  epochs: 80
  batch: 16
```

---

## Tuning the Merge Pipeline

Edit `configs/pipeline.yaml`:

| Parameter | Effect |
|---|---|
| `specialists.<cond>.conf_thresh` | Per-model detection threshold before merging |
| `cross_model_iou_threshold` | IoU above which cross-condition boxes are deduped |
| `student.pseudo_label_min_conf` | Second-pass filter on merged pseudo-labels |
| `student.real_data_weight` | How much to oversample real annotated data |

**If too few pseudo-labels survive:** lower `conf_thresh` and/or `pseudo_label_min_conf`.  
**If pseudo-labels are noisy:** raise both. Start at 0.45 and tune from there.

---

## Key Design Decisions

### Why separate models instead of multi-label from the start?
Each condition has a different dataset distribution, class balance, and even annotation
style. A joint model would have to reconcile all of that simultaneously.
Separate specialists learn cleaner features per condition, then the student
distills them into a single efficient model.

### Cross-model deduplication
Standard NMS only deduplicates within the same class. Here, a tooth region might
be flagged as both "discoloration" and "plaque". The cross-model NMS in `merge.py`
handles this: when two different-condition boxes overlap above the IoU threshold,
only the higher-confidence one survives. This is intentional — conditions that
genuinely co-occur (e.g. gingivitis + recession) will usually have *non-overlapping*
boxes, so they both survive.

### Plaque domain gap
The best plaque dataset uses disclosing gel (blue-stained teeth). `normalize.py`'s
`--domain-shift` flag applies HSV perturbations to partially bridge the gap.
It's not perfect — consider fine-tuning the plaque specialist on even 50–100 natural-
light images if you can get them from your intraoral camera.

### Gum recession (now broadly covered)
Originally the weakest condition (~187 images from `gum_recession_dataset` alone),
recession is now backed by three sources merged inside `normalize_recession`:

- `gum_recession_dataset` — class 0, native bbox.
- `Spot` — class 5 ("Caries 5 class"), YOLO polygon. Only lines tagged as class 5
  are kept; all other Spot classes (Abrasion, other Caries levels, Crown, Filling)
  are dropped per-line, and any image left with zero lines is skipped. Polygons
  are reduced to enclosing bboxes by `polygon_to_bbox_yolo`.
- `big_gum_dataset` — class 1 (`receding_gum`), YOLO bbox. Class 0 (`diseased_gum`)
  is dropped; images without `receding_gum` are skipped.

The Spot and big_gum imports get `spot_` / `big_gum_` filename prefixes when
copied into the temp pool, so filename collisions across sources can't silently
overwrite each other. Total recession pool is now ~1,961 images before re-split,
so the `augment_small_dataset(target=300)` step in `normalize_recession` is now
typically a no-op — bump the target up or remove the call once you've verified
the real-data counts are sufficient. The pseudo-labeling pass from the other 5
specialists still adds further annotations from unlabeled data.

---

## File Structure

```
bina-pipeline/
├── BinaDatasets/              ← source datasets (various formats)
│   ├── Carries_Dataset/
│   ├── Carries_abrasion_restoration_dataset/
│   ├── Gingivites_Dataset/
│   ├── mendeley-dataset-materials_Part_2/
│   ├── CALCULUS_Dataset/
│   ├── gum_recession_dataset/
│   ├── Spot/                  ← polygon; class 5 → recession
│   ├── big_gum_dataset/       ← bbox; class 1 (receding_gum) → recession
│   └── extensive_dataset/
│
├── configs/
│   ├── pipeline.yaml          ← master config (thresholds, paths, hyperparams)
│   └── models/
│       ├── caries.yaml
│       ├── gingivitis.yaml
│       ├── plaque.yaml
│       ├── discoloration.yaml
│       ├── ulcer.yaml
│       └── recession.yaml
│
├── data/
│   ├── processed/             ← normalized YOLO datasets (auto-generated)
│   │   ├── caries/
│   │   │   ├── images/{train,val,test}/
│   │   │   ├── labels/{train,val,test}/
│   │   │   └── dataset.yaml
│   │   ├── gingivitis/
│   │   ├── plaque/
│   │   ├── discoloration/
│   │   ├── ulcer/
│   │   └── recession/
│   ├── unlabeled/             ← unlabeled images to pseudo-label
│   └── pseudo_labeled/        ← merged pseudo-annotations (auto-generated)
│
├── src/
│   ├── data/
│   │   └── normalize.py       ← dataset normalization & format conversion
│   ├── train/
│   │   └── train_specialist.py
│   ├── inference/
│   │   ├── analyze.py         ← analyze images with trained models
│   │   ├── ensemble.py        ← multi-model inference
│   │   └── merge.py           ← NMS, cross-model dedup, pseudo-label writing
│   └── pipeline.py            ← main orchestrator
│
├── runs/                      ← training outputs (auto-generated)
│   └── specialists/
│       ├── specialist_caries/
│       │   └── weights/{best.pt, last.pt}
│       ├── specialist_gingivitis/
│       └── ...
│
└── requirements.txt
```

---

## Troubleshooting

### "No module named 'ultralytics'"

```bash
pip install ultralytics torch torchvision
```

### "CUDA out of memory"

Reduce batch size in `configs/pipeline.yaml`:
```yaml
train:
  batch: 8    # or 4
```

### "No trained models found" (when analyzing)

Train the models first:
```bash
python src/pipeline.py --phase train
```

### Training not improving

- Check dataset quality (correct labels, clear images)
- Try more epochs: `python src/train/train_specialist.py --condition caries --epochs 120`
- Check class balance in training data
- Review training curves in `runs/specialists/specialist_<condition>/results.png`

### New data doesn't seem to help

- Ensure labels are in correct YOLO format (class 0, normalized coordinates)
- Verify image and label filenames match (e.g., `img001.jpg` ↔ `img001.txt`)
- Check that new images are in `train/` directory
- Re-run normalization if adding to `BinaDatasets/`

### Hardware Recommendations

| GPU VRAM | Recommended Batch Size |
|----------|------------------------|
| 4GB | 4 |
| 8GB | 16 |
| 12GB+ | 32 |

Training time per specialist (80 epochs, ~3000 images):
- RTX 4060 (8GB): ~30-60 min
- RTX 3090 (24GB): ~15-30 min
