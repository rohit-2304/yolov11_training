import cv2
import random
from pathlib import Path

def main():
    base_dir = Path("/home/rohit/Rohit/Product_QA/yolo_v8_test/new_all_products")
    
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist!")
        return

    # Find all image files
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
    image_files = [p for p in base_dir.rglob("*") if p.suffix in valid_exts and p.is_file()]
    
    total_images = len(image_files)
    print(f"Found {total_images} images in {base_dir}")
    
    if total_images == 0:
        print("No images found to transform.")
        return

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(image_files)

    # Calculate 20% split sizes
    chunk_size = total_images // 5
    
    # 5 buckets:
    # 0: Original (0-20%)
    # 1: 90 deg Clockwise (20-40%)
    # 2: 90 deg Anti-Clockwise (40-60%)
    # 3: 180 deg (60-80%)
    # 4: Horizontal Flip (80-100%)
    
    counts = {
        "original": 0,
        "rotate_90_cw": 0,
        "rotate_90_ccw": 0,
        "rotate_180": 0,
        "flip_horizontal": 0
    }

    for i, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        if i < chunk_size:
            # Original - no change
            counts["original"] += 1
            continue
        elif i < chunk_size * 2:
            # 90 deg clockwise
            transformed = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            counts["rotate_90_cw"] += 1
        elif i < chunk_size * 3:
            # 90 deg anticlockwise
            transformed = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            counts["rotate_90_ccw"] += 1
        elif i < chunk_size * 4:
            # 180 deg
            transformed = cv2.rotate(img, cv2.ROTATE_180)
            counts["rotate_180"] += 1
        else:
            # Horizontal flip
            transformed = cv2.flip(img, 1)
            counts["flip_horizontal"] += 1

        # Overwrite file with transformed image
        cv2.imwrite(str(img_path), transformed, [cv2.IMWRITE_JPEG_QUALITY, 100])

    print("\nTransformation Summary:")
    print(f"  - Original (unchanged): {counts['original']}")
    print(f"  - Rotated 90° Clockwise: {counts['rotate_90_cw']}")
    print(f"  - Rotated 90° Anti-clockwise: {counts['rotate_90_ccw']}")
    print(f"  - Rotated 180°: {counts['rotate_180']}")
    print(f"  - Flipped Horizontally: {counts['flip_horizontal']}")
    print(f"Total processed: {sum(counts.values())} / {total_images}")

if __name__ == "__main__":
    main()
