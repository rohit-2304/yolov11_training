import os
import sys
import time
import argparse
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# YOLO11n — Product Detector Training Script
# ============================================================
# Trains a single-class YOLO11n model on the data_products_only
# dataset. Labels have been cleaned in-place to contain only
# the 'product' class (remapped to class 0).
# ============================================================


def verify_dataset(base: Path):
    """Quick sanity check — count product annotations per split."""
    print("\nDataset verification:")
    for split in ["train", "valid", "test"]:
        lbl_dir = base / split / "labels"
        if not lbl_dir.exists():
            continue
        total = 0
        files = 0
        for f in lbl_dir.glob("*.txt"):
            files += 1
            with open(f) as fp:
                for line in fp:
                    if line.strip():
                        total += 1
        print(f"  {split}: {files} label files, {total} product annotations")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11n Product Detector (single class)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch",  type=int, default=8,   help="Batch size")
    parser.add_argument("--imgsz",  type=int, default=640, help="Image size (default 640)")
    parser.add_argument("--data_dir", type=str, default="new_data_products_only", help="Path or folder name of the dataset")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to weights to fine-tune from (default: output/product_detector/train_results/weights/best.pt or yolo11n.pt)")
    parser.add_argument("--name", type=str, default="train_results",
                        help="Run name for output directory (default: train_results)")
    parser.add_argument("--exist_ok", action="store_true", default=False,
                        help="Overwrite existing output directory if it exists (default: False)")
    args = parser.parse_args()

    print("=" * 60)
    print("   YOLO11n — Product Detector Training")
    print("=" * 60)

    workspace_dir = Path(__file__).resolve().parent
    dataset_dir   = Path(args.data_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = workspace_dir / dataset_dir

    if not dataset_dir.exists():
        print(f"ERROR: dataset directory not found at {dataset_dir}")
        sys.exit(1)

    print(f"Dataset directory: {dataset_dir}")

    # Generate an absolute-path data.yaml at runtime so YOLO resolves paths correctly
    data_yaml_path = dataset_dir / "data_local.yaml"
    data_yaml_content = f"""path: {dataset_dir.resolve()}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['product']
"""
    with open(data_yaml_path, "w") as f:
        f.write(data_yaml_content)
    print(f"Generated data_local.yaml -> {data_yaml_path}")

    verify_dataset(dataset_dir)
    print("-" * 60)

    # Load model weights
    if args.weights:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            print(f"ERROR: weights not found at {weights_path}")
            sys.exit(1)
        print(f"Fine-tuning from specified weights: {weights_path}")
    else:
        best_pt = workspace_dir / "output" / "product_detector" / "train_results" / "weights" / "best.pt"
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
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Train
    project_dir = str(workspace_dir / "output" / "product_detector")
    name = args.name

    print(f"\nTraining config:")
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
            auto_augment="False",

            # Regularisation
            dropout=0.1,
            patience=20,

            # Augmentations — conservative, preserve product shape
            fliplr=0.5,
            flipud=0.0,
            mosaic=0.0,
            mixup=0.0,
            hsv_h=0.015,
            hsv_s=0.2,
            hsv_v=0.1,
            degrees=5.0,
            translate=0.1,
            scale=0.2,
            shear=0.0,
            perspective=0.0,

            # Loss weights
            cls=0.5,
            box=7.5,
        )

        duration = time.time() - start_time
        print("-" * 60)
        print(f"Training complete! ({duration/60:.1f} min)")
        best_weights = Path(project_dir) / name / "weights" / "best.pt"
        print(f"Best weights: {best_weights}")
        print("=" * 60)

    except Exception as e:
        print(f"\nTraining failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

