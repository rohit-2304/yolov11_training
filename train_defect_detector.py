import sys
import time
import argparse
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# YOLO11n — Defect Detector Training Script
# ============================================================
# Trains on data_defects_only dataset.
# Classes (from Roboflow export):
#   0: dentporosity  (merged dent + porosity)
#   1: unclearsurface
#
# Input images: cropped product images (from crop_products.py)
# ============================================================


def verify_dataset(base: Path, names: list):
    """Quick sanity check — count annotations per class per split."""
    from collections import Counter
    print("\nDataset verification:")
    for split in ["train", "valid", "test"]:
        lbl_dir = base / split / "labels"
        if not lbl_dir.exists():
            continue
        counts = Counter()
        files = 0
        for f in lbl_dir.glob("*.txt"):
            files += 1
            for line in f.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
        total = sum(counts.values())
        img_dir = base / split / "images"
        imgs = len(list(img_dir.glob("*"))) if img_dir.exists() else "?"
        print(f"  {split}: {imgs} images, {files} label files, {total} annotations")
        for i, name in enumerate(names):
            print(f"    class {i} ({name}): {counts.get(i, 0)}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11n Defect Detector")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs (default 150)")
    parser.add_argument("--batch",  type=int, default=8,   help="Batch size (default 8)")
    parser.add_argument("--imgsz",  type=int, default=640, help="Image size (default 640)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to weights to fine-tune from (default: yolo11n.pt pretrained)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset directory (default: data_defects_only)")
    parser.add_argument("--name", type=str, default="train_results",
                        help="Run name for output directory (default: train_results)")
    parser.add_argument("--exist_ok", action="store_true", default=False,
                        help="Overwrite existing output directory if it exists (default: False)")
    args = parser.parse_args()

    print("=" * 60)
    print("   YOLO11n — Defect Detector Training")
    print("=" * 60)

    workspace_dir = Path(__file__).resolve().parent
    if args.dataset:
        dataset_dir = Path(args.dataset).resolve()
    else:
        dataset_dir = workspace_dir / "data_defects_only"

    # Read class names from data.yaml
    import yaml
    with open(dataset_dir / "data.yaml") as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg.get("names", [])
    nc = data_cfg.get("nc", len(class_names))
    print(f"Classes ({nc}): {class_names}")

    # Generate absolute-path data_local.yaml so YOLO resolves paths correctly
    data_yaml_path = dataset_dir / "data_local.yaml"
    data_yaml_content = f"""path: {dataset_dir.resolve()}
train: train/images
val: valid/images
test: test/images

nc: {nc}
names: {class_names}
"""
    with open(data_yaml_path, "w") as f:
        f.write(data_yaml_content)
    print(f"Generated data_local.yaml -> {data_yaml_path}")

    verify_dataset(dataset_dir, class_names)
    print("-" * 60)

    # Load model weights
    if args.weights:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            print(f"ERROR: weights not found at {weights_path}")
            sys.exit(1)
        print(f"Fine-tuning from specified weights: {weights_path}")
    else:
        best_pt = workspace_dir / "output" / "defect_detector" / "train_results" / "weights" / "best.pt"
        local_pt = workspace_dir / "yolo11n.pt"
        if best_pt.exists():
            weights_path = best_pt
            print(f"Fine-tuning from existing best model: {weights_path}")
        elif local_pt.exists():
            weights_path = local_pt
            print(f"Starting from pretrained base: {weights_path}")
        else:
            weights_path = Path("yolo11n.pt")
            print(f"Starting from default yolo11n.pt")

    try:
        model = YOLO(str(weights_path))
        print("Model loaded.\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Output paths
    project_dir = str(workspace_dir / "output" / "defect_detector")
    name = args.name

    print(f"Training config:")
    print(f"  imgsz   : {args.imgsz}")
    print(f"  batch   : {args.batch}")
    print(f"  epochs  : {args.epochs}")
    print(f"  output  : {project_dir}/{name}")
    print(f"  exist_ok: {args.exist_ok}")
    print("-" * 60)

    start_time = time.time()
    try:
        model.train(
            data=str(data_yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device="",
            project=project_dir,
            name=name,
            exist_ok=args.exist_ok,
            plots=True,
            workers=1,
            cache=True,
            verbose=True,
            auto_augment=False,

            # Regularisation — more aggressive than product detector
            # because defect dataset is smaller and harder
            dropout=0.2,
            patience=30,

            # Loss weights — raise cls penalty to reduce false positives
            cls=1.5,
            box=7.5,

            # Augmentations — modest, preserve defect appearance
            fliplr=0.5,
            flipud=0.0,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            hsv_h=0.01,          # very small hue shift
            hsv_s=0.15,          # low saturation variance
            hsv_v=0.1,           # low brightness variance
            degrees=5.0,
            translate=0.05,      # small translation
            scale=0.1,           # small scale variance
            shear=0.0,
            perspective=0.0,
        )

        duration = time.time() - start_time
        print("-" * 60)
        print(f"Training complete! ({duration/60:.1f} min)")

        best  = Path(project_dir) / name / "weights" / "best.pt"
        last  = Path(project_dir) / name / "weights" / "last.pt"
        print(f"Best weights : {best}")
        print(f"Last weights : {last}")
        print("=" * 60)

    except Exception as e:
        print(f"\nTraining failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
