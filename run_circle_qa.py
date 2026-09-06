"""
run_circle_qa.py
================
Run the circle_qa module on all images in cropped_all_products/ and:
  - Save annotated panel images to circle_qa_output/ (same directory structure)
  - Write a JSON report to circle_qa_output/circle_qa_report_<timestamp>.json

Usage:
    python3 run_circle_qa.py
    python3 run_circle_qa.py --source cropped_all_products --output circle_qa_output
    python3 run_circle_qa.py --params canny_lo=15 canny_hi=70 elongation_std_frac=0.05
"""

import sys
import json
import argparse
import cv2
from datetime import datetime
from pathlib import Path

# Allow running from yolo_v8_test/ directly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from circle_qa import CircleQA, DEFAULT_PARAMS, draw_result

# ──────────────────────────────────────────────────────────────
# DEFAULTS
# ──────────────────────────────────────────────────────────────
DEFAULT_SOURCE = "cropped_all_products"
DEFAULT_OUTPUT = "circle_qa_output"
JPEG_QUALITY   = 95


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def parse_param_overrides(raw: list) -> dict:
    """
    Parse CLI param overrides like  key=value  into a dict.
    Numeric strings are auto-converted to int or float.
    'true'/'false' strings become booleans.
    """
    out = {}
    for token in raw:
        if "=" not in token:
            print(f"  WARNING: ignoring malformed param token '{token}' (expected key=value)")
            continue
        key, val = token.split("=", 1)
        key = key.strip()
        val = val.strip()
        # Type coercion
        if val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass   # keep as string
        out[key] = val
    return out


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Circle QA inference on cropped product images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parameter overrides (--params key=value ...):
""" + "\n".join(f"  {k:<30} default={v}" for k, v in DEFAULT_PARAMS.items())
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help=f"Source directory of cropped images (default: {DEFAULT_SOURCE})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--params", nargs="*", default=[],
                        metavar="key=value",
                        help="Override CircleQA params, e.g. canny_lo=15 elongation_std_frac=0.05")
    parser.add_argument("--panel-size", type=int, default=480,
                        help="Side length of the annotated panel image in px (default: 480)")
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY,
                        help="JPEG quality for saved panels (default: 95)")
    args = parser.parse_args()

    workspace_dir = Path(__file__).resolve().parent
    source_dir    = (workspace_dir / args.source
                     if not Path(args.source).is_absolute()
                     else Path(args.source))
    output_dir    = (workspace_dir / args.output
                     if not Path(args.output).is_absolute()
                     else Path(args.output))

    param_overrides = parse_param_overrides(args.params or [])

    print("=" * 60)
    print("   Circle QA — Batch Inference")
    print("=" * 60)
    print(f"Source       : {source_dir}")
    print(f"Output       : {output_dir}")
    print(f"Panel size   : {args.panel_size}×{args.panel_size}px")
    if param_overrides:
        print(f"Param overrides:")
        for k, v in param_overrides.items():
            print(f"  {k} = {v}  (default: {DEFAULT_PARAMS.get(k, 'N/A')})")
    else:
        print("Params       : all defaults")
    print("-" * 60)

    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        sys.exit(1)

    # Build evaluator
    try:
        qa = CircleQA(params=param_overrides)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Gather all images
    image_files = list(source_dir.rglob("*.jpg")) + list(source_dir.rglob("*.JPG")) + \
                  list(source_dir.rglob("*.png")) + list(source_dir.rglob("*.PNG"))
    if not image_files:
        print(f"No images found in {source_dir}")
        sys.exit(0)

    print(f"Found {len(image_files)} images.\n")
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON report skeleton
    run_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    report = {
        "run_timestamp": run_ts,
        "source_dir":    str(source_dir),
        "params_used":   qa.p,
        "summary": {"total": 0, "PASS": 0, "FAIL": 0, "ERROR": 0},
        "images":  {},
    }
    json_path = output_dir / f"circle_qa_report_{run_ts.replace(':', '-')}.json"

    panel_wh = (args.panel_size, args.panel_size)

    for i, img_path in enumerate(image_files, 1):
        rel_path    = img_path.relative_to(source_dir)
        # Output panel: same relative path, JPEG
        out_name    = img_path.stem + "_circle_qa.jpg"
        output_path = output_dir / rel_path.parent / out_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{i:>3}/{len(image_files)}] {rel_path}", end="  ")

        img = cv2.imread(str(img_path))
        if img is None:
            print("→ SKIP (could not read)")
            result = CircleQA._error("Could not read image file.")
        else:
            result = qa.run(img)

        status = result["status"]
        print(f"→ {status}", end="")
        if result["defects"]:
            print(f"  [{result['defects'][0].split(':')[0].strip()}]", end="")
        print()

        # Save annotated panel
        if img is not None:
            panel = draw_result(img, result, panel_size=panel_wh)
            cv2.imwrite(str(output_path), panel, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        else:
            # Write a blank error panel
            blank = cv2.putText(
                255 * __import__("numpy").ones((args.panel_size, args.panel_size, 3),
                                              dtype=__import__("numpy").uint8),
                "READ ERROR", (40, args.panel_size // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2
            )
            cv2.imwrite(str(output_path), blank,
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

        # Record in report
        report["summary"]["total"]  += 1
        report["summary"][status]   = report["summary"].get(status, 0) + 1
        report["images"][str(rel_path)] = {
            "status":       status,
            "defects":      result["defects"],
            "center":       list(result["center"]) if result["center"] else None,
            "ring_width":   result["ring_width"],
            "panel_saved":  str(output_path.relative_to(output_dir)),
        }

        # Flush JSON after every image (crash-safe)
        with open(json_path, "w") as jf:
            json.dump(report, jf, indent=2)

    # ── Summary ─────────────────────────────────────────────────
    s = report["summary"]
    print("\n" + "=" * 60)
    print("Done.")
    print(f"  Total   : {s['total']}")
    print(f"  PASS    : {s.get('PASS',  0)}")
    print(f"  FAIL    : {s.get('FAIL',  0)}")
    print(f"  ERROR   : {s.get('ERROR', 0)}")
    print(f"Output    : {output_dir}")
    print(f"JSON      : {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
