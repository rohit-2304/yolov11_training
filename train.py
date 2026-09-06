import os
import sys
import time
import subprocess
import argparse
import shutil
from ultralytics import YOLO

# ============================================================
# YOLO11 Nano Training Script — v8 Dataset
# ============================================================
# Runs augment.py first, then trains/fine-tunes YOLO11n on the
# v8 dataset.
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 Nano on v8 dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    print("=" * 60)
    print("   YOLO11 Nano Training Script — Defect Detection v8")
    print("=" * 60)

    # Step 1: Run the dataset preparation tool
    print("Preparing and validating dataset...")
    try:
        subprocess.run([sys.executable, "augment.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running dataset preparation. Aborting training.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running dataset preparation: {e}")
        sys.exit(1)

    print("-" * 60)

    # Define paths
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(workspace_dir, "data_local.yaml")
    project_dir    = os.path.join(workspace_dir, "output/yolo11n")
    name           = "train_results"

    # Step 1.5: Create data_local.yaml dynamically with absolute paths
    print("Generating data_local.yaml dynamically...")
    data_yaml_content = f"""path: {os.path.join(workspace_dir, 'data')}
train: train/images
val: valid/images
test: test/images

nc: 5
names: ['Porosity', 'chipoff', 'dent', 'product', 'surface unclean']
"""
    with open(data_yaml_path, "w") as f:
        f.write(data_yaml_content)

    print(f"Dataset Config : {data_yaml_path}")
    print(f"Output Project : {project_dir}")
    print(f"Run Name       : {name}")
    print("-" * 60)

    # Step 2: Load the YOLO11 model
    print("Loading YOLO11 model...")
    try:
        local_weights = os.path.join(workspace_dir, "yolo11n.pt")
        # Try copying pretrained weights from yolo_v5_test if not present locally
        if not os.path.exists(local_weights):
            v5_weights = os.path.abspath(os.path.join(workspace_dir, "../yolo_v5_test/yolo11n.pt"))
            if os.path.exists(v5_weights):
                print(f"  Found yolo11n.pt in yolo_v5_test — copying: {v5_weights}")
                shutil.copy2(v5_weights, local_weights)
            
        model = YOLO(local_weights if os.path.exists(local_weights) else "yolo11n.pt")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("-" * 60)
    print("Starting training process...")
    start_time = time.time()

    # Step 3: Train the model
    try:
        epochs = args.epochs
        batch_size = args.batch
        imgsz = args.imgsz
        dropout_val = 0.2
        patience_val = 25
        device = ""  # Let ultralytics auto-select the best device (CUDA GPU or CPU)

        print(f"Device Selected : auto")
        print(f"Epochs Configured: {epochs}")
        print(f"Batch Size      : {batch_size}")
        print(f"Image Size      : {imgsz}")
        print("-" * 60)

        results = model.train(
            auto_augment=False,
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=project_dir,
            name=name,
            exist_ok=True,
            plots=True,
            workers=1,
            verbose=True,
            cache=True,
            
            # --- Regularization ---
            dropout=dropout_val,     
            patience=patience_val,   
            
            # --- Basic and very few Augmentations ---
            fliplr=0.5,              # random horizontal flip
            flipud=0.0,
            mosaic=0.0,              # heavily reduced mosaic
            mixup=0.0,
            hsv_h=0.015,             # standard small hue shift
            hsv_s=0.2,               # reduced saturation variance
            hsv_v=0.1,               # reduced intensity variance
            degrees=5.0,             # basic rotation
            translate=0.1,           # basic translation
            scale=0.1,               # basic scaling
            shear=0.0,
            perspective=0.0
        )

        duration = time.time() - start_time
        print("-" * 60)
        print("Training completed successfully!")
        print(f"Total training time: {duration:.2f}s ({duration/60:.2f} min)")

        # Report weight paths
        weights_dir  = os.path.join(project_dir, name, "weights")
        best_weights = os.path.join(weights_dir, "best.pt")
        last_weights = os.path.join(weights_dir, "last.pt")

        print(f"Best weights : {best_weights}")
        print(f"Last weights : {last_weights}")
        print("=" * 60)

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
