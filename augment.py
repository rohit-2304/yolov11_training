import os
import shutil
from collections import Counter

def clean_and_convert_labels():
    """
    Scans all label files in train, valid, and test splits and converts
    any polygon/segmentation annotations to clean YOLO bounding boxes.
    """
    print("Converting any polygon annotations to clean bounding boxes...")
    converted_lines_total = 0

    for split in ["train", "valid", "test"]:
        lbl_dir = f"data/{split}/labels"
        if not os.path.exists(lbl_dir):
            print(f"Directory {lbl_dir} does not exist.")
            continue

        for f in os.listdir(lbl_dir):
            if f.endswith(".txt") and not f.startswith(".") and "_aug" not in f:
                lbl_path = os.path.join(lbl_dir, f)
                with open(lbl_path, "r") as file:
                    lines = file.readlines()

                new_lines = []
                file_changed = False

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 5:
                        # Polygon/segmentation: class x1 y1 x2 y2 ... xN yN
                        cls = parts[0]
                        coords = list(map(float, parts[1:]))

                        # Separate X and Y coordinates
                        xs = coords[0::2]
                        ys = coords[1::2]

                        xmin = min(xs)
                        xmax = max(xs)
                        ymin = min(ys)
                        ymax = max(ys)

                        # Clamp to [0.0, 1.0]
                        xmin = max(0.0, min(1.0, xmin))
                        xmax = max(0.0, min(1.0, xmax))
                        ymin = max(0.0, min(1.0, ymin))
                        ymax = max(0.0, min(1.0, ymax))

                        w = xmax - xmin
                        h = ymax - ymin
                        x = xmin + w / 2.0
                        y = ymin + h / 2.0

                        new_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                        file_changed = True
                        converted_lines_total += 1
                    else:
                        new_lines.append(line)

                if file_changed:
                    with open(lbl_path, "w") as file:
                        file.writelines(new_lines)

    print(f"Successfully converted {converted_lines_total} polygon annotations to bounding boxes.")


def verify_class_presence():
    """
    Checks the training labels and counts instances of each class
    to verify presence of all classes (0 to 4) in the training data.
    """
    print("\nVerifying class presence in training data...")
    train_lbl_dir = "data/train/labels"
    class_names = ['Porosity', 'chipoff', 'dent', 'product', 'surface unclean']
    
    if not os.path.exists(train_lbl_dir):
        print(f"Error: Training labels directory not found at {train_lbl_dir}")
        return False

    class_counts = Counter()
    total_files = 0
    
    for f in os.listdir(train_lbl_dir):
        if f.endswith(".txt") and not f.startswith("."):
            total_files += 1
            lbl_path = os.path.join(train_lbl_dir, f)
            with open(lbl_path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    if parts:
                        cls = int(parts[0])
                        class_counts[cls] += 1
                        
    print(f"Scanned {total_files} training label files.")
    print("Class distribution in training data:")
    for idx, name in enumerate(class_names):
        count = class_counts.get(idx, 0)
        print(f"  - Class {idx} ({name}): {count} instances")
        if count == 0:
            print(f"  WARNING: Class {idx} ({name}) is MISSING from training data!")

    missing_classes = [idx for idx in range(len(class_names)) if class_counts.get(idx, 0) == 0]
    if not missing_classes:
        print("Success: Each class is present in the training data.")
        return True
    else:
        print(f"Error: Missing classes in training split: {missing_classes}")
        return False


def include_validation_in_training():
    """
    Copies all validation images and labels into the training directories
    so they are included in training. Also clears labels.cache files to force YOLO to rebuild them.
    """
    print("Merging validation dataset into training dataset...")
    valid_img_dir = "data/valid/images"
    valid_lbl_dir = "data/valid/labels"
    train_img_dir = "data/train/images"
    train_lbl_dir = "data/train/labels"
    
    if not os.path.exists(valid_img_dir) or not os.path.exists(valid_lbl_dir):
        print("Validation directories not found. Skipping merge.")
        return
        
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    
    copied_images = 0
    copied_labels = 0
    
    for filename in os.listdir(valid_img_dir):
        if filename.startswith('.'):
            continue
        src = os.path.join(valid_img_dir, filename)
        dst = os.path.join(train_img_dir, filename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied_images += 1
            
    for filename in os.listdir(valid_lbl_dir):
        if filename.startswith('.'):
            continue
        src = os.path.join(valid_lbl_dir, filename)
        dst = os.path.join(train_lbl_dir, filename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied_labels += 1
            
    print(f"Copied {copied_images} validation images and {copied_labels} validation labels to training set.")
    
    # Remove labels.cache files to force YOLO to re-scan
    for cache_path in ["data/train/labels.cache", "data/valid/labels.cache", "data/test/labels.cache"]:
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                print(f"Cleared cache file: {cache_path}")
            except Exception as e:
                print(f"Warning: Could not clear cache file {cache_path}: {e}")


def main():
    print("=" * 60)
    # 0. Include validation photos in training too
    include_validation_in_training()
    print("-" * 60)
    # 1. Clean and convert polygon/segmentation labels to bounding boxes
    clean_and_convert_labels()
    print("-" * 60)
    # 2. Verify class presence
    if not verify_class_presence():
        exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()
