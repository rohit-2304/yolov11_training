"""
run_ring_inspection.py
======================
Train one RingInspectionModel per product type (on the `normal/` directory),
then run inference on every image in both `normal/` and `defective/` directories.

Directory layout expected:
    type_wise_all_products/
        <ProductType>/
            normal/       ← golden training images
            defective/    ← (optional) known-bad samples

Outputs to ring_inspection_output/:
    <ProductType>/
        weights.json            ← learned model weights for this product type
        normal/
            <img>_overlay.jpg   ← cartesian arc overlay
            <img>_polar.jpg     ← polar Sobel strip
        defective/
            <img>_overlay.jpg
            <img>_polar.jpg
    report.json                 ← full structured results

Usage:
    python3 run_ring_inspection.py
    python3 run_ring_inspection.py --source type_wise_all_products --output ring_inspection_output
    python3 run_ring_inspection.py --fallback SAP5751         # enable C3 fallback for a specific type
    python3 run_ring_inspection.py --fallback-all             # enable C3 fallback for ALL types
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ring_inspection import RingInspectionModel

# ──────────────────────────────────────────────────────────────
# DEFAULTS
# ──────────────────────────────────────────────────────────────
DEFAULT_SOURCE = "type_wise_all_products"
DEFAULT_OUTPUT = "ring_inspection_output"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
JPEG_QUALITY = 92


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────


def collect_images(directory: Path) -> list:
    if not directory.exists():
        return []
    return sorted([f for f in directory.rglob("*") if f.suffix in IMAGE_EXTS])


def safe_metrics(metrics: dict) -> dict:
    """Convert any numpy types to plain Python for JSON serialisation."""
    out = {}
    for k, v in metrics.items():
        if v is None:
            out[k] = None
        elif isinstance(v, dict):
            out[k] = safe_metrics(v)
        elif hasattr(v, "tolist"):
            out[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            out[k] = [float(x) if hasattr(x, "item") else x for x in v]
        elif isinstance(v, float):
            out[k] = round(v, 6)
        else:
            out[k] = v
    return out


def run_inference_on_dir(
    model: RingInspectionModel, img_dir: Path, out_dir: Path, tag: str
) -> list:
    """
    Run predict() on every image in img_dir.
    Save overlay + polar panel to out_dir.
    Return list of per-image result dicts.
    """
    images = collect_images(img_dir)
    results = []

    if not images:
        return results

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [{tag}] {len(images)} images …")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            results.append(
                {
                    "image": str(img_path.name),
                    "status": "ERROR",
                    "metrics": {"defects": ["Could not read image"]},
                }
            )
            continue

        status, metrics, (polar_vis, cart_overlay) = model.predict(img)

        stem = img_path.stem

        # Save cartesian overlay
        if cart_overlay is not None:
            overlay_path = out_dir / f"{stem}_overlay.jpg"
            cv2.imwrite(
                str(overlay_path),
                cart_overlay,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

        # Save polar strip
        if polar_vis is not None:
            polar_path = out_dir / f"{stem}_polar.jpg"
            cv2.imwrite(
                str(polar_path), polar_vis, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

        result_entry = {
            "image": str(img_path.name),
            "status": status,
            "metrics": safe_metrics(metrics),
        }
        results.append(result_entry)

        defect_hint = (
            f"  [{metrics['defects'][0][:50]}]" if metrics.get("defects") else ""
        )
        print(f"    {img_path.name:<35} → {status}{defect_hint}")

    return results


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train RingInspectionModel per product type and run inference"
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"Root directory with product-type folders (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--fallback",
        nargs="*",
        default=[],
        metavar="PRODUCT_TYPE",
        help="Enable Circle-3 fallback for these product types, "
        "e.g. --fallback SAP5751 SAP119",
    )
    parser.add_argument(
        "--fallback-all",
        action="store_true",
        help="Enable Circle-3 fallback for ALL product types",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; load existing weights.json if present",
    )
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent
    source_dir = (
        workspace / args.source
        if not Path(args.source).is_absolute()
        else Path(args.source)
    )
    output_dir = (
        workspace / args.output
        if not Path(args.output).is_absolute()
        else Path(args.output)
    )
    fallback_types = set(args.fallback or [])

    print("=" * 64)
    print("   Ring Inspection — Train & Inference Pipeline")
    print("=" * 64)
    print(f"Source  : {source_dir}")
    print(f"Output  : {output_dir}")
    print(f"C3 fallback: {'ALL' if args.fallback_all else (fallback_types or 'none')}")
    print("-" * 64)

    if not source_dir.exists():
        print(f"ERROR: source directory not found: {source_dir}")
        sys.exit(1)

    # Collect product type directories
    product_types = sorted(
        [d for d in source_dir.iterdir() if d.is_dir() and (d / "normal").exists()]
    )

    if not product_types:
        print("No product type directories with a 'normal/' sub-folder found.")
        sys.exit(1)

    print(
        f"Found {len(product_types)} product types: "
        f"{[d.name for d in product_types]}\n"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    full_report = {
        "run_timestamp": run_ts,
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "summary": {
            "total_images": 0,
            "PASS": 0,
            "REJECT": 0,
            "ERROR": 0,
        },
        "product_types": {},
    }
    report_path = output_dir / "report.json"

    # ── Per-product-type loop ──────────────────────────────────
    for pt_dir in product_types:
        pt_name = pt_dir.name
        normal_dir = pt_dir / "normal"
        defect_dir = pt_dir / "defective"
        pt_out_dir = output_dir / pt_name
        weights_path = pt_out_dir / "weights.json"

        use_fallback = args.fallback_all or (pt_name in fallback_types)

        print(f"\n{'━'*64}")
        print(f"  Product Type : {pt_name}")
        print(
            f"  Normal dir   : {normal_dir}  ({len(collect_images(normal_dir))} images)"
        )
        print(
            f"  Defective dir: {defect_dir}  ({len(collect_images(defect_dir))} images)"
        )
        print(f"  C3 fallback  : {use_fallback}")
        print(f"{'━'*64}")

        # ── TRAIN ──────────────────────────────────────────────
        model = RingInspectionModel()

        if args.skip_train and weights_path.exists():
            print(
                f"  [TRAIN] Skipping — loading existing weights from {weights_path.name}"
            )
            model.load_model_weights(str(weights_path))
        else:
            n_train = len(collect_images(normal_dir))
            if n_train == 0:
                print(f"  [TRAIN] No normal images found — using default weights.")
            else:
                print(f"  [TRAIN] Training on {n_train} golden images …")
                model.train(str(normal_dir), allow_circle_3_fallback=use_fallback)

            pt_out_dir.mkdir(parents=True, exist_ok=True)
            model.save_model_weights(str(weights_path))

        # ── INFERENCE ──────────────────────────────────────────
        pt_report = {
            "product_type": pt_name,
            "weights_file": str(weights_path.relative_to(output_dir)),
            "circle_3_fallback": use_fallback,
            "trained_params_summary": {
                "nominal_gap_12_frac": model.params["geometric_constraints"].get(
                    "nominal_gap_12_frac"
                ),
                "nominal_gap_24_frac": model.params["geometric_constraints"].get(
                    "nominal_gap_24_frac"
                ),
                "nominal_ring_width_px": model.params["geometric_constraints"].get(
                    "nominal_ring_width_px"
                ),
                "learned_ratio_3_to_24": model.params["circle_3_inspection_parameters"][
                    "learned_ratio_3_to_24"
                ],
                "max_variance_epsilon": model.params["defect_thresholds"][
                    "max_variance_epsilon_frac"
                ],
            },
            "summary": {"total": 0, "PASS": 0, "REJECT": 0, "ERROR": 0},
            "normal": [],
            "defective": [],
        }

        # Normal images
        normal_results = run_inference_on_dir(
            model, normal_dir, pt_out_dir / "normal", "normal"
        )
        pt_report["normal"] = normal_results

        # Defective images (if directory exists)
        defective_results = run_inference_on_dir(
            model, defect_dir, pt_out_dir / "defective", "defective"
        )
        pt_report["defective"] = defective_results

        # Per-type summary
        for entry in normal_results + defective_results:
            s = entry["status"]
            pt_report["summary"]["total"] += 1
            pt_report["summary"][s] = pt_report["summary"].get(s, 0) + 1
            full_report["summary"]["total_images"] += 1
            full_report["summary"][s] = full_report["summary"].get(s, 0) + 1

        full_report["product_types"][pt_name] = pt_report

        n = pt_report["summary"]
        print(
            f"  Result → Total={n['total']}  PASS={n.get('PASS',0)}  "
            f"REJECT={n.get('REJECT',0)}  ERROR={n.get('ERROR',0)}"
        )

        # Flush JSON after each product type (crash-safe)
        with open(report_path, "w") as f:
            json.dump(full_report, f, indent=2)

    # ── Final summary ──────────────────────────────────────────
    s = full_report["summary"]
    print(f"\n{'='*64}")
    print(f"  All done.")
    print(f"  Total images  : {s['total_images']}")
    print(f"  PASS          : {s.get('PASS',  0)}")
    print(f"  REJECT        : {s.get('REJECT', 0)}")
    print(f"  ERROR         : {s.get('ERROR',  0)}")
    print(f"  Output dir    : {output_dir}")
    print(f"  JSON report   : {report_path}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
