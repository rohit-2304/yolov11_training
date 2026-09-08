import os
import shutil
import argparse
from pathlib import Path
from collections import Counter
import yaml

# ============================================================
# Prepare Defect Dataset without Chipoff
# ============================================================
# Original classes in data/defects/data.yaml:
#   0: chippoff  --> REMOVED
#   1: dent      --> 0
#   2: porosity  --> 1
#   3: scratch   --> 2
#   4: surface_unclean --> 3
#   5: white_mark      --> 4
# ============================================================

NEW_CLASS_NAMES = ['dent', 'porosity', 'scratch', 'surface_unclean', 'white_mark']
EXCLUDED_CLASS_ID = 0  # chippoff


def copy_or_link(src: Path, dst: Path):
    """Link file if possible to save disk space and time, else copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def process_dataset(src_dir: Path, dst_dir: Path):
    print("=" * 60)
    print(f"Preparing dataset without 'chipoff'")
    print(f"Source: {src_dir}")
    print(f"Target: {dst_dir}")
    print("=" * 60)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    # Read original data.yaml
    src_yaml_path = src_dir / "data.yaml"
    if src_yaml_path.exists():
        with open(src_yaml_path) as f:
            src_cfg = yaml.safe_load(f)
        orig_names = src_cfg.get("names", [])
        print(f"Original classes ({len(orig_names)}): {orig_names}")
    else:
        orig_names = ['chippoff', 'dent', 'porosity', 'scratch', 'surface_unclean', 'white_mark']

    dst_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "valid", "test"]
    stats = {}

    for split in splits:
        src_split = src_dir / split
        dst_split = dst_dir / split
        if not src_split.exists():
            continue

        src_img_dir = src_split / "images"
        src_lbl_dir = src_split / "labels"
        dst_img_dir = dst_split / "images"
        dst_lbl_dir = dst_split / "labels"

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        orig_counts = Counter()
        new_counts = Counter()
        chipoff_removed_count = 0
        total_images = 0
        empty_label_files = 0

        # Process all label files
        label_files = list(src_lbl_dir.glob("*.txt")) if src_lbl_dir.exists() else []
        for lbl_file in label_files:
            new_lines = []
            for line in lbl_file.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                orig_counts[cls_id] += 1
                if cls_id == EXCLUDED_CLASS_ID:
                    chipoff_removed_count += 1
                    continue
                # Remap: cls_id - 1
                remapped_id = cls_id - 1
                new_counts[remapped_id] += 1
                new_lines.append(f"{remapped_id} " + " ".join(parts[1:]))

            target_lbl = dst_lbl_dir / lbl_file.name
            if new_lines:
                target_lbl.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            else:
                # Write empty file to serve as negative/background sample
                target_lbl.write_text("", encoding="utf-8")
                empty_label_files += 1

        # Link/copy images
        if src_img_dir.exists():
            for img_file in src_img_dir.glob("*"):
                if img_file.is_file():
                    total_images += 1
                    target_img = dst_img_dir / img_file.name
                    copy_or_link(img_file, target_img)

        stats[split] = {
            "total_images": total_images,
            "label_files": len(label_files),
            "empty_labels": empty_label_files,
            "chipoff_removed": chipoff_removed_count,
            "orig_counts": orig_counts,
            "new_counts": new_counts,
        }

    # Write new data.yaml
    new_yaml = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": len(NEW_CLASS_NAMES),
        "names": NEW_CLASS_NAMES,
    }
    with open(dst_dir / "data.yaml", "w") as f:
        yaml.dump(new_yaml, f, sort_keys=False)

    # Write local data_local.yaml
    data_local_content = f"""path: {dst_dir.resolve()}
train: train/images
val: valid/images
test: test/images

nc: {len(NEW_CLASS_NAMES)}
names: {NEW_CLASS_NAMES}
"""
    with open(dst_dir / "data_local.yaml", "w") as f:
        f.write(data_local_content)

    print("\nDataset Conversion Summary:")
    print("-" * 60)
    for split, s in stats.items():
        print(f"\nSplit: [{split.upper()}]")
        print(f"  Images: {s['total_images']} | Label files: {s['label_files']} (Empty/Background: {s['empty_labels']})")
        print(f"  Chipoff annotations removed: {s['chipoff_removed']}")
        print(f"  New class distributions:")
        for idx, name in enumerate(NEW_CLASS_NAMES):
            print(f"    Class {idx} ({name}): {s['new_counts'].get(idx, 0)} (was class {idx+1}: {s['orig_counts'].get(idx+1, 0)})")

    print("\n" + "=" * 60)
    print(f"Created: {dst_dir / 'data.yaml'}")
    print(f"Created: {dst_dir / 'data_local.yaml'}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter out chipoff from defect dataset")
    parser.add_argument("--src", type=str, default="data/defects", help="Source defects directory")
    parser.add_argument("--dst", type=str, default="data/defects_no_chipoff", help="Target defects directory")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent
    src = (workspace / args.src).resolve() if not Path(args.src).is_absolute() else Path(args.src).resolve()
    dst = (workspace / args.dst).resolve() if not Path(args.dst).is_absolute() else Path(args.dst).resolve()

    process_dataset(src, dst)
