import os
import sys
import json
from ultralytics import YOLO

# ============================================================
# YOLO11 Nano Testing & Evaluation Script — v8 Dataset
# ============================================================
# Evaluates the trained model on the test split and generates
# annotated prediction images saved to output/yolo11n/.
# ============================================================

def main():
    print("=" * 60)
    print("   YOLO11 Nano Testing & Evaluation Script — v8")
    print("=" * 60)

    # Paths
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    best_weights_path = os.path.join(workspace_dir, "output/yolo11n/train_results/weights/best.pt")
    data_yaml_path    = os.path.join(workspace_dir, "data_local.yaml")
    output_dir        = os.path.join(workspace_dir, "output/yolo11n")
    test_images_dir   = os.path.join(workspace_dir, "data/test/images")

    print(f"Best Weights : {best_weights_path}")
    print(f"Dataset YAML : {data_yaml_path}")
    print(f"Test Images  : {test_images_dir}")
    print("-" * 60)

    # Verify weights exist
    if not os.path.exists(best_weights_path):
        print(f"ERROR: Trained model weights not found at {best_weights_path}.")
        print("Please run train.py first to generate weights.")
        sys.exit(1)

    # Step 1: Load trained model
    print("Loading best trained model...")
    try:
        model = YOLO(best_weights_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("-" * 60)
    print("Running evaluation on 'test' split...")

    # Step 2: Run evaluation
    metrics_dict = {}
    try:
        device = ""  # auto
        metrics = model.val(
            data=data_yaml_path,
            split="test",
            device=device,
            plots=True,
            project=output_dir,
            name="test_evaluation",
            exist_ok=True,
        )

        map50  = float(metrics.box.map50)
        map95  = float(metrics.box.map)
        mp     = float(metrics.box.mp)
        mr     = float(metrics.box.mr)
        fitness = float(metrics.fitness)

        print("\nTest Evaluation Summary:")
        print(f"  - mean Precision (mP) : {mp:.4f}")
        print(f"  - mean Recall (mR)    : {mr:.4f}")
        print(f"  - mAP50               : {map50:.4f}")
        print(f"  - mAP50-95            : {map95:.4f}")
        print(f"  - Fitness Score       : {fitness:.4f}")

        # Class-wise metrics
        class_metrics = {}
        for i, name in model.names.items():
            try:
                # Class index can map to lists inside metrics.box
                class_metrics[name] = {
                    "precision": float(metrics.box.class_result(i)[0]),
                    "recall": float(metrics.box.class_result(i)[1]),
                    "mAP50": float(metrics.box.class_result(i)[2]),
                    "mAP50_95": float(metrics.box.class_result(i)[3]),
                }
            except Exception as ex:
                class_metrics[name] = f"Error extracting: {ex}"

        metrics_dict = {
            "model_version": "YOLO11 Nano v8 (yolo11n)",
            "metrics": {
                "mean_precision": mp,
                "mean_recall":    mr,
                "mAP50":          map50,
                "mAP50_95":       map95,
                "fitness_score":  fitness,
            },
            "class_metrics": class_metrics,
            "class_indices_to_names": model.names,
        }

        metrics_json_path = os.path.join(output_dir, "test_metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"\nSaved test metrics JSON to: {metrics_json_path}")

    except Exception as e:
        print(f"Warning: Evaluation failed or test labels incomplete: {e}")
        print("Proceeding with visual predictions anyway...")
        metrics_dict = {"status": "Evaluation skipped", "error": str(e)}

    print("-" * 60)
    print("Generating annotated prediction images for test set...")

    # Step 3: Perform prediction and save annotated images
    try:
        device = ""
        results = model.predict(
            source=test_images_dir,
            save=True,
            imgsz=640,
            conf=0.25,
            project=output_dir,
            name="test_annotated",
            exist_ok=True,
            device=device,
        )

        annotated_dir = os.path.join(output_dir, "test_annotated")
        print("\nPrediction visual results:")
        if os.path.exists(annotated_dir):
            for file in sorted(os.listdir(annotated_dir)):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    print(f"  - {os.path.join(annotated_dir, file)}")
        else:
            print("  Check output/yolo11n/test_annotated/ for saved predictions.")

        print("=" * 60)
        print("Inference completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Prediction failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
