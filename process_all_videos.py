import os
import sys
import yaml
from pathlib import Path
from ultralytics import YOLO
import shutil
import gc
import torch

def main():
    print("=" * 60)
    print("   YOLO Batch Video Inference Script")
    print("=" * 60)

    # Paths
    workspace_dir = Path(__file__).resolve().parent
    app_dir = workspace_dir.parent / "app"
    config_path = app_dir / "config.yaml"
    
    best_weights_path = workspace_dir / "output/yolo11n/train_results/weights/best.pt"
    source_dir = workspace_dir.parent / "all_products" / "mp4_videos"
    output_base_dir = workspace_dir / "all_products_output"
    
    print(f"Source Directory: {source_dir}")
    print(f"Output Directory: {output_base_dir}")
    print(f"Weights Path    : {best_weights_path}")
    print("-" * 60)

    # Validations
    if not source_dir.exists():
        print(f"ERROR: Source directory {source_dir} not found.")
        sys.exit(1)
        
    if not best_weights_path.exists():
        print(f"ERROR: Trained model weights not found at {best_weights_path}.")
        sys.exit(1)

    # Step 1: Load config to get image size
    imgsz = 640 # User requested 640 size
    conf_thresh = 0.25
    
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            model_cfg = config.get("model", {})
            conf_thresh = model_cfg.get("confidence_threshold", 0.25)
            # We stick to imgsz=640 as explicitly requested by user recently
            print(f"Loaded config: conf={conf_thresh}, imgsz={imgsz}")
        except Exception as e:
            print(f"Warning: Failed to read config: {e}")
    else:
        print(f"Warning: Config not found at {config_path}")

    # Step 2: Load trained model
    print("\nLoading best trained model...")
    try:
        model = YOLO(best_weights_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("-" * 60)
    print("Running batch inference on videos...")

    # Step 3: Find all videos
    video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    video_files = []
    for ext in video_extensions:
        video_files.extend(source_dir.rglob(f"*{ext}"))
        video_files.extend(source_dir.rglob(f"*{ext.upper()}"))
        
    # Remove duplicates if any (due to case-insensitivity on Windows, less likely on Linux, but safe)
    video_files = list(set(video_files))
    
    if not video_files:
        print(f"No videos found in {source_dir}.")
        sys.exit(0)
        
    print(f"Found {len(video_files)} video files to process.")

    # Step 4: Process videos
    for i, video_path in enumerate(video_files, 1):
        rel_path = video_path.relative_to(source_dir)
        target_dir = output_base_dir / rel_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[{i}/{len(video_files)}] Processing: {rel_path}")
        
        try:
            # Run predict. 
            # Note: setting name="" might not work in all Ultralytics versions (sometimes defaults to 'predict').
            # To be absolutely sure about the output file location, we run predict in a temporary directory
            # and move the generated file to our precise target structure.
            temp_output_dir = output_base_dir / ".temp_predict"
            if temp_output_dir.exists():
                shutil.rmtree(temp_output_dir)
                
            # Use stream=True to prevent accumulating results in RAM and crashing the system
            results = model.predict(
                source=str(video_path),
                save=True,
                imgsz=imgsz,
                conf=conf_thresh,
                project=str(temp_output_dir),
                name="run",
                exist_ok=True,
                device="",
                stream=True
            )
            
            # Iterate through the generator to process and save the video frame-by-frame
            for _ in results:
                pass
            
            # The output video should be at temp_output_dir / "run" / video_path.name
            # Sometimes YOLO appends AVI for videos, but usually keeps MP4 if we use OpenCV with MP4V.
            generated_file = None
            run_dir = temp_output_dir / "run"
            if run_dir.exists():
                for f in run_dir.iterdir():
                    if f.is_file() and f.stem == video_path.stem:
                        generated_file = f
                        break
                        
            if generated_file:
                # Move to the correct spot
                final_dest = target_dir / generated_file.name
                if final_dest.exists():
                    final_dest.unlink() # replace
                shutil.move(str(generated_file), str(final_dest))
                print(f"  -> Saved annotated video to: {final_dest.relative_to(workspace_dir)}")
            else:
                print(f"  -> Warning: Could not find generated file in {run_dir}")
                
            # Cleanup temp dir
            if temp_output_dir.exists():
                shutil.rmtree(temp_output_dir)
                
            # Clear memory to prevent slow leaks between videos
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing {rel_path}: {e}")

    print("=" * 60)
    print("All video processing completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
