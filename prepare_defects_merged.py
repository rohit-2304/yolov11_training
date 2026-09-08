import os
import shutil
import argparse
from pathlib import Path
from collections import Counter
import yaml

# ============================================================
# Prepare Defect Dataset with Merged Classes
# ============================================================
# Merges dent + porosity + (optionally) chipoff into a single class.
#
# Original classes in data/defects/data.yaml:
#   0: chippoff
#   1: dent
#   2: porosity
#   3: scratch
#   4: surface_unclean
#   5: white_mark
# ============================================================

def copy_or_link(src: Path, dst: Path):
    """Link file if possible to save disk space and time, else copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def process_dataset(src_dir: Path, dst_dir: Path, include_chipoff: bool = True, merged_name: str = "dent_porosity_chipoff"):
    if not include_chipoff and merged_name == "dent_porosity_chipoff":
        merged_name = "dent_porosity"

    new_class_names = [merged_name, "scratch", "surface_unclean", "white_mark"]

    print("=" * 60)
    print(f"Preparing MERGED dataset: {new_class_names}")
    print(f"Include chipoff in merged class: {include_chipoff}")
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
        merged_source_counts = Counter()
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

                # Mapping logic:
                # 0: chippoff
                # 1: dent
                # 2: porosity
                # 3: scratch
                # 4: surface_unclean
                # 5: white_mark
                if cls_id == 0:  # chippoff
                    if include_chipoff:
                        target_id = 0
                        merged_source_counts["chippoff"] += 1
                        new_counts[target_id] += 1
                        new_lines.append(f"{target_id} " + " ".join(parts[1:]))
                    else:
                        # Drop chipoff completely
                        continue
                elif cls_id in (1, 2):  # dent (1) or porosity (2)
                    target_id = 0
                    src_label = "dent" if cls_id == 1 else "porosity"
                    merged_source_counts[src_label] += 1
                    new_counts[target_id] += 1
                    new_lines.append(f"{target_id} " + " ".join(parts[1:]))
                elif cls_id == 3:  # scratch -> 1
                    target_id = 1
                    new_counts[target_id] += 1
                    new_lines.append(f"{target_id} " + " ".join(parts[1:]))
                elif cls_id == 4:  # surface_unclean -> 2
                    target_id = 2
                    new_counts[target_id] += 1
                    new_lines.append(f"{target_id} " + " ".join(parts[1:]))
                elif cls_id == 5:  # white_mark -> 3
                    target_id = 3
                    new_counts[target_id] += 1
                    new_lines.append(f"{target_id} " + " ".join(parts[1:]))

            target_lbl = dst_lbl_dir / lbl_file.name
            if new_lines:
                target_lbl.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            else:
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
            "orig_counts": orig_counts,
            "new_counts": new_counts,
            "merged_sources": merged_source_counts,
        }

    # Write new data.yaml
    new_yaml = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": len(new_class_names),
        "names": new_class_names,
    }
    with open(dst_dir / "data.yaml", "w") as f:
        yaml.dump(new_yaml, f, sort_keys=False)

    # Write local data_local.yaml
    data_local_content = f"""path: {dst_dir.resolve()}
train: train/images
val: valid/images
test: test/images

nc: {len(new_class_names)}
names: {new_class_names}
"""
    with open(dst_dir / "data_local.yaml", "w") as f:
        f.write(data_local_content)

    print("\nDataset Conversion Summary:")
    print("-" * 60)
    for split, s in stats.items():
        print(f"\nSplit: [{split.upper()}]")
        print(f"  Images: {s['total_images']} | Label files: {s['label_files']} (Empty/Background: {s['empty_labels']})")
        print(f"  Class 0 ({new_class_names[0]}): {s['new_counts'].get(0, 0)} total instances")
        print(f"    Composition: {dict(s['merged_sources'])}")
        for idx in range(1, len(new_class_names)):
            print(f"  Class {idx} ({new_class_names[idx]}): {s['new_counts'].get(idx, 0)}")

    print("\n" + "=" * 60)
    print(f"Created: {dst_dir / 'data.yaml'}")
    print(f"Created: {dst_dir / 'data_local.yaml'}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create merged defect dataset")
    parser.add_argument("--src", type=str, default="data/defects", help="Source defects directory")
    parser.add_argument("--dst", type=str, default="data/defects_merged", help="Target defects directory")
    parser.add_argument("--no-chipoff", action="store_true", help="Exclude chipoff from the merged class")
    parser.add_argument("--name", type=str, default="dent_porosity_chipoff", help="Name for merged class 0")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent
    src = (workspace / args.src).resolve() if not Path(args.src).is_absolute() else Path(args.src).resolve()
    dst = (workspace / args.dst).resolve() if not Path(args.dst).is_absolute() else Path(args.dst).resolve()

    process_dataset(src, dst, include_chipoff=not args.no_chipoff, merged_name=args.name)
