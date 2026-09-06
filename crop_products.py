import sys
import cv2
import gc
import torch
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# Product Crop Script
# ============================================================
# Runs the product detector on every JPG in all_products/,
# crops the highest-confidence bounding box with a configurable
# buffer margin, and saves the result to cropped_all_products/
# mirroring the original directory structure.
# ============================================================

import argparse

# --- Config Defaults ---
BUFFER_FRACTION  = 0.05   # 5% of the bbox width/height added as padding on each side
CONF_THRESHOLD   = 0.25   # minimum confidence to accept a detection
IMGSZ            = 640    # inference image size
JPEG_QUALITY     = 100    # 0-100; 100 = maximum quality, no compression loss


def crop_with_buffer(img, x1, y1, x2, y2, buf_frac=BUFFER_FRACTION):
    """
    Expand bbox by buf_frac of its dimensions on each side,
    clamped to image boundaries.
    """
    h, w = img.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * buf_frac)
    pad_y = int(bh * buf_frac)

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    return img[cy1:cy2, cx1:cx2]


def main():
    workspace_dir  = Path(__file__).resolve().parent
    
    parser = argparse.ArgumentParser(description="Crop product images using YOLO product detector")
    parser.add_argument("--source_dir", type=str, default="new_all_products", help="Source image directory")
    parser.add_argument("--output_dir", type=str, default="new_all_products_cropped", help="Output cropped directory")
    parser.add_argument("--weights", type=str, default=None, help="Weights path (default: output/product_detector/train_results/weights/best.pt)")
    parser.add_argument("--buffer", type=float, default=BUFFER_FRACTION, help="Padding buffer fraction (default: 0.05)")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.is_absolute():
        source_dir = workspace_dir / source_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = workspace_dir / output_dir

    if args.weights:
        weights_path = Path(args.weights)
        if not weights_path.is_absolute():
            weights_path = workspace_dir / weights_path
    else:
        weights_path = workspace_dir / "output/product_detector/train_results/weights/best.pt"

    buf_frac = args.buffer
    conf_thresh = args.conf

    print("=" * 60)
    print("   Product Crop Script")
    print("=" * 60)
    print(f"Weights   : {weights_path}")
    print(f"Source    : {source_dir}")
    print(f"Output    : {output_dir}")
    print(f"Buffer    : {buf_frac * 100:.1f}% padding")
    print("-" * 60)

    # Validate
    if not weights_path.exists():
        print(f"ERROR: Model weights not found at {weights_path}")
        print("       Run train_product_detector.py first.")
        sys.exit(1)

    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        sys.exit(1)

    # Load model
    print("Loading product detector model...")
    model = YOLO(str(weights_path))
    print("Model loaded.\n")

    # Gather all images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
    image_files = sorted([p for p in source_dir.rglob("*") if p.suffix in valid_exts and p.is_file()])
    if not image_files:
        print(f"No images found in {source_dir}")
        sys.exit(0)

    print(f"Found {len(image_files)} images to process.\n")

    skipped   = 0
    processed = 0
    no_detect = 0

    for i, img_path in enumerate(image_files, 1):
        # Mirror the relative path in the output directory
        rel_path   = img_path.relative_to(source_dir)
        output_path = output_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{i:>3}/{len(image_files)}] {rel_path}", end="  ")

        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print("→ SKIP (could not read)")
            skipped += 1
            continue

        # Run inference
        results = model.predict(
            source=str(img_path),
            imgsz=IMGSZ,
            conf=conf_thresh,
            verbose=False,
            device=""
        )

        # Pick the highest-confidence product detection
        best_box  = None
        best_conf = -1.0

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box  = box

        if best_box is None:
            # No product detected — save the original image unchanged
            print(f"→ no detection (saved original)")
            cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            no_detect += 1
        else:
            # Crop with buffer
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cropped = crop_with_buffer(img, x1, y1, x2, y2, buf_frac=buf_frac)
            cv2.imwrite(str(output_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            print(f"→ cropped {x2-x1}×{y2-y1}px  conf={best_conf:.2f}")
            processed += 1

        # Free memory between images
        del results, img
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f"Done.")
    print(f"  Cropped successfully : {processed}")
    print(f"  No detection (original saved) : {no_detect}")
    print(f"  Skipped (read error) : {skipped}")
    print(f"  Output directory     : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
