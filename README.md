# Project Antigravity — YOLOv11 Training & Ring Inspection Pipeline

End-to-end computer vision system for industrial ring component inspection.
Two YOLO detectors (product + defect) work in tandem with a geometric
`RingInspectionModel` to identify, segment, and quality-inspect 4 concentric
boundaries on ring parts.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Dataset Layout](#dataset-layout)
4. [Pipeline Overview](#pipeline-overview)
5. [Training Commands](#training-commands)
   - [Stage 0 — Dataset Preparation](#stage-0--dataset-preparation-augmentpy)
   - [Stage 1 — Product Detector](#stage-1--product-detector)
   - [Stage 2 — Defect Detector](#stage-2--defect-detector)
   - [Stage 3 — Ring Inspection Model](#stage-3--ring-inspection-model-geometric)
6. [Inference Commands](#inference-commands)
   - [Two-Phase Defect Detection (images)](#two-phase-defect-detection-images)
   - [Two-Phase Defect Detection (video)](#two-phase-defect-detection-video)
   - [Circle QA](#circle-qa)
7. [Output Weights Reference](#output-weights-reference)
8. [Key CLI Flags Summary](#key-cli-flags-summary)

---

## Project Structure

```
yolo_v8_test/
├── train.py                     # Combined YOLO11n training (augment → train)
├── train_product_detector.py    # Product detector training (single-class)
├── train_defect_detector.py     # Defect detector training (2-class)
├── augment.py                   # Dataset preparation / polygon→bbox conversion
├── crop_products.py             # Crop product ROIs from full images
├── detect_two_phase.py          # Two-phase inference on image directories
├── detect_two_phase_video.py    # Two-phase inference on video files
├── run_ring_inspection.py       # Train & run geometric ring inspection model
├── run_circle_qa.py             # Run circle QA on cropped product images
├── ring_inspection/
│   ├── model.py                 # RingInspectionModel class
│   ├── pipeline.py              # Core geometric pipeline
│   ├── visualise.py             # Visualisation helpers
│   └── default_weights.json     # Default parameter seed
├── circle_qa/
│   ├── evaluator.py             # CircleQA evaluator
│   └── visualise.py
└── output/
    ├── product_detector/weights/best.pt
    ├── defect_detector/weights/best.pt
    └── yolo11n/weights/best.pt
```

---

## Prerequisites

```bash
pip install ultralytics opencv-python-headless numpy scipy pyyaml
```

> **GPU:** All scripts auto-select CUDA if available (`device=""`).
> No manual device flag is required.

---

## Dataset Layout

### YOLO Detector Datasets

```
data_products_only/          # or new_data_products_only/
    train/images/  train/labels/
    valid/images/  valid/labels/
    test/images/   test/labels/
    data.yaml

data_defects_only/           # or new_data_defects_only/
    train/images/  train/labels/
    valid/images/  valid/labels/
    test/images/   test/labels/
    data.yaml
```

### Ring Inspection Dataset

```
type_wise_all_products/
    <ProductType>/           # e.g. SAP87c2, SAP119, SAP58c1 …
        normal/              # golden training images (PASS samples)
        defective/           # optional known-bad samples for validation
```

---

## Pipeline Overview

```
Full Images (all_products/)
       │
       ▼
[Stage 1] train_product_detector.py   →  product detector weights
       │
       ▼
[crop_products.py]                    →  cropped_all_products/
       │
       ▼
[Stage 2] train_defect_detector.py    →  defect detector weights
       │
       ▼
[detect_two_phase.py]                 →  annotated output images
       │
       ▼
[Stage 3] run_ring_inspection.py      →  per-product-type weights.json
                                         + polar/overlay visualisations
```

---

## Training Commands

### Stage 0 — Dataset Preparation (`augment.py`)

Merges validation into training, converts polygon/segmentation labels to
YOLO bounding boxes, and verifies all class IDs are present.

> `train.py` runs this automatically — only call it manually if needed.

```bash
# Run dataset preparation only (operates on data/ directory)
python3 augment.py
```

---

### Stage 1 — Product Detector

Trains a single-class YOLO11n model to locate the product in a full-frame image.

**Defaults:** 100 epochs · batch 8 · imgsz 640 · dataset `new_data_products_only/`

```bash
# Quick start — all defaults
python3 train_product_detector.py

# Custom dataset directory
python3 train_product_detector.py --data_dir new_data_products_only

# Custom hyperparameters
python3 train_product_detector.py \
    --epochs 150 \
    --batch 16 \
    --imgsz 640 \
    --data_dir new_data_products_only

# Fine-tune from a specific checkpoint
python3 train_product_detector.py \
    --weights output/product_detector/train_results/weights/best.pt \
    --epochs 50

# Save to a new named run (avoids overwriting train_results/)
python3 train_product_detector.py \
    --name train_results_red_cam \
    --exist_ok
```

**Output weights:**
```
output/product_detector/<name>/weights/best.pt
output/product_detector/<name>/weights/last.pt
```

---

### Stage 2 — Defect Detector

Trains a 2-class YOLO11n model on **cropped** product images.
Classes: `dentporosity` (0), `unclearsurface` (1).

**Defaults:** 150 epochs · batch 8 · imgsz 640 · dataset `data_defects_only/`

```bash
# Quick start — all defaults (auto fine-tunes from best.pt if it exists)
python3 train_defect_detector.py

# Custom dataset
python3 train_defect_detector.py --dataset new_data_defects_only

# Custom hyperparameters
python3 train_defect_detector.py \
    --epochs 200 \
    --batch 8 \
    --imgsz 640

# Fine-tune from a specific checkpoint
python3 train_defect_detector.py \
    --weights output/defect_detector/train_results/weights/best.pt \
    --epochs 50

# Save to a new named run
python3 train_defect_detector.py \
    --name train_new_products \
    --exist_ok

# Full example: new dataset, new run name, 200 epochs
python3 train_defect_detector.py \
    --dataset new_data_defects_only \
    --epochs 200 \
    --name train_results_red_cam_defects \
    --exist_ok
```

**Output weights:**
```
output/defect_detector/<name>/weights/best.pt
output/defect_detector/<name>/weights/last.pt
output/defect_detector/<name>/weights/best.onnx  (if exported)
```

---

#### Combined Training Script (`train.py`)

Runs `augment.py` first, then trains YOLO11n on the `data/` v8 dataset.
Uses `output/yolo11n/` as the project directory.

```bash
# Default: 100 epochs, batch 8, imgsz 640
python3 train.py

# Custom
python3 train.py --epochs 150 --batch 16 --imgsz 640
```

**Output weights:**
```
output/yolo11n/train_results/weights/best.pt
output/yolo11n/train_results/weights/last.pt
```

---

### Stage 3 — Ring Inspection Model (Geometric)

`RingInspectionModel` is an ML-like class that **auto-tunes** geometric
parameters (CLAHE, Canny, polar-unwrap thresholds, Circle 3 variance ε)
by learning from golden (PASS) product images.

One model is trained **per product type**. Learned weights are saved as
`weights.json` inside `ring_inspection_output/<ProductType>/`.

```bash
# Train all product types found in type_wise_all_products/
# and run inference on normal/ + defective/ images
python3 run_ring_inspection.py

# Custom source and output directories
python3 run_ring_inspection.py \
    --source type_wise_all_products \
    --output ring_inspection_output

# Enable Circle 3 fallback for specific product types
# (PASS the product even if Circle 3 cannot be tracked)
python3 run_ring_inspection.py \
    --fallback SAP5751 SAP119

# Enable Circle 3 fallback for ALL product types
python3 run_ring_inspection.py --fallback-all

# Skip re-training; load existing weights.json files
python3 run_ring_inspection.py --skip-train

# Skip training + custom source
python3 run_ring_inspection.py \
    --skip-train \
    --source PRODUCTSNEW \
    --output ring_inspection_output
```

**Output per product type:**
```
ring_inspection_output/
    <ProductType>/
        weights.json          ← learned model weights (committed to git)
        normal/<img>_overlay.jpg
        normal/<img>_polar.jpg
        defective/<img>_overlay.jpg
    report.json               ← full structured JSON results
```

---

## Inference Commands

### Two-Phase Defect Detection (images)

Runs Phase 1 (product detection) → crops ROI → runs Phase 2 (defect detection)
→ saves annotated images.

```bash
# Default source: all_products/ → output: two_phase_output/
python3 detect_two_phase.py

# Custom source directory
python3 detect_two_phase.py --source new_all_products --output two_phase_output

# Custom model weights
python3 detect_two_phase.py \
    --product-weights output/product_detector/train_results_red_cam/weights/best.pt \
    --defect-weights  output/defect_detector/train_results_red_cam_defects/weights/best.pt

# Tune confidence thresholds
python3 detect_two_phase.py \
    --product-conf 0.30 \
    --defect-conf 0.45

# Increase crop buffer (more context around the product)
python3 detect_two_phase.py --buffer 0.10

# Hide the product bounding box, show only defect boxes
python3 detect_two_phase.py --no-product-box

# Full custom run
python3 detect_two_phase.py \
    --source        PRODUCTSNEW \
    --output        two_phase_output \
    --product-weights output/product_detector/train_results/weights/best.pt \
    --defect-weights  output/defect_detector/train_new_products/weights/best.pt \
    --product-conf  0.25 \
    --defect-conf   0.40 \
    --buffer        0.05 \
    --jpeg-quality  95
```

---

### Two-Phase Defect Detection (video)

Same pipeline as above, operating on video files.

```bash
python3 detect_two_phase_video.py --source <video_file_or_dir>
```

---

### Crop Products

Use the product detector to crop the product ROI from full-frame images.
Required before running the defect detector on a new image set.

```bash
# Default: source=new_all_products/ → output=new_all_products_cropped/
python3 crop_products.py

# Custom directories
python3 crop_products.py \
    --source_dir PRODUCTSNEW \
    --output_dir PRODUCTSNEW_CROPPED

# Custom weights and confidence
python3 crop_products.py \
    --weights output/product_detector/train_results/weights/best.pt \
    --conf 0.30 \
    --buffer 0.05
```

---

### Circle QA

Evaluates concentricity / circularity of cropped product images.

```bash
# Default: source=cropped_all_products/ → output=circle_qa_output/
python3 run_circle_qa.py

# Custom source
python3 run_circle_qa.py --source new_all_products_cropped

# Override Canny / detection parameters at runtime
python3 run_circle_qa.py \
    --params canny_lo=15 canny_hi=70 elongation_std_frac=0.05

# Custom panel size and JPEG quality
python3 run_circle_qa.py \
    --panel-size 640 \
    --jpeg-quality 95
```

---

## Output Weights Reference

| Model | Weights Path |
|---|---|
| Product Detector (latest) | `output/product_detector/train_results/weights/best.pt` |
| Defect Detector (latest) | `output/defect_detector/train_results/weights/best.pt` |
| Defect Detector (ONNX) | `output/defect_detector/train_new_products/weights/best.onnx` |
| Combined YOLO11n | `output/yolo11n/train_results/weights/best.pt` |
| Ring Inspection (per product) | `ring_inspection_output/<ProductType>/weights.json` |
| Ring Inspection (default seed) | `ring_inspection/default_weights.json` |

---

## Key CLI Flags Summary

| Script | Key Flags |
|---|---|
| `train_product_detector.py` | `--epochs` `--batch` `--imgsz` `--data_dir` `--weights` `--name` `--exist_ok` |
| `train_defect_detector.py` | `--epochs` `--batch` `--imgsz` `--dataset` `--weights` `--name` `--exist_ok` |
| `train.py` | `--epochs` `--batch` `--imgsz` |
| `run_ring_inspection.py` | `--source` `--output` `--fallback` `--fallback-all` `--skip-train` |
| `detect_two_phase.py` | `--source` `--output` `--product-weights` `--defect-weights` `--product-conf` `--defect-conf` `--buffer` `--no-product-box` |
| `crop_products.py` | `--source_dir` `--output_dir` `--weights` `--conf` `--buffer` |
| `run_circle_qa.py` | `--source` `--output` `--params` `--panel-size` `--jpeg-quality` |
