import sys
import cv2
import gc
import json
import argparse
import torch
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Re-use shared detection logic from detect_two_phase.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_two_phase import (
    ProductDetectorConfig, DefectDetectorConfig, VisualisationConfig,
    compute_crop_box, crop_to_original, draw_box,
    detect_product, detect_defects,
)
from ultralytics import YOLO

# ============================================================
# Two-Phase Defect Detection — Video Pipeline
# ============================================================
# Processes every MP4/MOV/AVI video in source_dir frame by frame:
#   Phase 1: product detector → product bbox → buffered crop
#   Phase 2: defect detector on crop → remap to original frame
#   Output : annotated video saved alongside original as
#            <name>_annotated.mp4, mirroring directory structure
# ============================================================


# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────

@dataclass
class VideoConfig:
    source_dir:             str   = "/home/rohit/Rohit/Product_QA/all_products/mp4_videos"
    output_dir:             str   = "two_phase_video_output"  # relative to yolo_v8_test/
    output_suffix:          str   = "_annotated"          # appended before extension
    output_ext:             str   = ".mp4"                # output container
    fourcc:                 str   = "mp4v"                # codec
    draw_product_box:       bool  = True
    skip_every_n:           int   = 1                     # process every N-th frame (1=all, 2=half)
    defect_frame_threshold: float = 0.10  # min fraction of frames with defects to call DEFECT_FOUND
    product:                ProductDetectorConfig = None
    defect:                 DefectDetectorConfig  = None
    vis:                    VisualisationConfig   = None

    def __post_init__(self):
        if self.product is None: self.product = ProductDetectorConfig()
        if self.defect  is None: self.defect  = DefectDetectorConfig()
        if self.vis     is None: self.vis     = VisualisationConfig()


# ──────────────────────────────────────────────────────────────
# PRODUCT BBOX CACHE — reuse last good detection per video
# ──────────────────────────────────────────────────────────────

class ProductBoxCache:
    """
    Keeps the last confident product bbox so that frames where
    the product detector fires with low confidence still get a crop.
    Falls back to the cached box rather than skipping the frame.
    """
    def __init__(self):
        self._box = None  # (x1, y1, x2, y2, conf)

    def update(self, box: Optional[tuple]):
        if box is not None:
            self._box = box

    def get(self) -> Optional[tuple]:
        return self._box

    def reset(self):
        self._box = None


# ──────────────────────────────────────────────────────────────
# PROCESS ONE VIDEO
# ──────────────────────────────────────────────────────────────

def process_video(video_path: Path, output_path: Path,
                  product_model: YOLO, defect_model: YOLO,
                  cfg: VideoConfig) -> dict:
    """
    Process a single video with the two-phase pipeline.
    Returns a stats dict.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path}")
        return {"frames": 0, "detected": 0, "defects": 0, "error": True}

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Resolution  : {width}×{height}  FPS: {fps:.1f}  Frames: {total_frames}")

    # Set up output video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*cfg.fourcc)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"  ERROR: cannot open VideoWriter for {output_path}")
        cap.release()
        return {"frames": 0, "detected": 0, "defects": 0, "error": True}

    stats = {"frames": 0, "product_detected": 0, "frames_with_defects": 0,
             "total_defects": 0, "error": False}
    defect_log = []   # list of {frame, class, confidence, bbox} events
    cache = ProductBoxCache()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        stats["frames"] += 1

        # ── Phase 1: Product Detection (every N frames) ─────────
        if frame_idx % cfg.skip_every_n == 0:
            result = detect_product(product_model, frame, cfg.product)
            cache.update(result)

        product_box = cache.get()

        annotated = frame.copy()

        if product_box is not None:
            px1, py1, px2, py2, p_conf = product_box
            h, w = frame.shape[:2]

            # Compute buffered crop region
            cx1, cy1, cx2, cy2 = compute_crop_box(
                h, w, px1, py1, px2, py2, cfg.product.buffer_fraction
            )

            # ── Phase 2: Defect Detection on crop ───────────────
            crop = frame[cy1:cy2, cx1:cx2].copy()
            defects = detect_defects(defect_model, crop, cfg.defect)

            # Remap defect boxes to original frame coordinates
            v = cfg.vis
            stats["product_detected"] += 1

            if cfg.draw_product_box:
                draw_box(
                    annotated, cx1, cy1, cx2, cy2,
                    label=f"product {p_conf:.2f}",
                    color=v.product_color,
                    thickness=v.product_thickness,
                    font_scale=v.font_scale,
                    font_thickness=v.font_thickness,
                    label_bg=v.label_bg,
                )

            for d in defects:
                ox1, oy1, ox2, oy2 = crop_to_original(
                    d["x1"], d["y1"], d["x2"], d["y2"], cx1, cy1
                )
                color = (v.defect_colors[d["cls_id"]]
                         if d["cls_id"] < len(v.defect_colors)
                         else (255, 255, 0))
                draw_box(
                    annotated, ox1, oy1, ox2, oy2,
                    label=f"{d['cls_name']} {d['conf']:.2f}",
                    color=color,
                    thickness=v.defect_thickness,
                    font_scale=v.font_scale,
                    font_thickness=v.font_thickness,
                    label_bg=v.label_bg,
                )

            if defects:
                stats["frames_with_defects"] += 1
                stats["total_defects"] += len(defects)
                for d in defects:
                    ox1, oy1, ox2, oy2 = crop_to_original(
                        d["x1"], d["y1"], d["x2"], d["y2"], cx1, cy1
                    )
                    defect_log.append({
                        "frame":      frame_idx,
                        "class":      d["cls_name"],
                        "confidence": round(d["conf"], 4),
                        "bbox_orig":  [ox1, oy1, ox2, oy2],
                    })

        writer.write(annotated)

        # Progress every 50 frames
        if frame_idx % 50 == 0:
            pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
            print(f"  [{frame_idx}/{total_frames}  {pct:.0f}%]  "
                  f"defects so far: {stats['total_defects']}")

    cap.release()
    writer.release()

    # Free GPU memory between videos
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stats, defect_log


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def run_video_pipeline(cfg: VideoConfig):
    workspace_dir    = Path(__file__).resolve().parent
    product_weights  = workspace_dir / cfg.product.weights
    defect_weights   = workspace_dir / cfg.defect.weights
    source_dir = Path(cfg.source_dir)
    # If output_dir is relative, resolve it from workspace (yolo_v8_test/)
    output_dir = (workspace_dir / cfg.output_dir
                  if not Path(cfg.output_dir).is_absolute()
                  else Path(cfg.output_dir))

    print("=" * 60)
    print("   Two-Phase Defect Detection — Video Pipeline")
    print("=" * 60)
    print(f"Source dir      : {source_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"Product model   : {product_weights}  conf={cfg.product.conf_threshold}")
    print(f"Defect model    : {defect_weights}  conf={cfg.defect.conf_threshold}")
    print(f"Buffer          : {cfg.product.buffer_fraction * 100:.0f}%")
    print(f"Frame skip      : every {cfg.skip_every_n} frame(s)")
    print("-" * 60)

    for path, label in [(product_weights, "Product weights"),
                        (defect_weights,  "Defect weights"),
                        (source_dir,      "Source directory")]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    print("Loading models...")
    product_model = YOLO(str(product_weights))
    defect_model  = YOLO(str(defect_weights))
    print("Models loaded.\n")

    # Collect all videos
    video_exts = {".mp4", ".mov", ".avi", ".mkv"}
    video_files = [
        f for f in source_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in video_exts
        and cfg.output_suffix not in f.stem   # don't re-process already annotated
    ]

    if not video_files:
        print(f"No video files found in {source_dir}")
        sys.exit(0)

    print(f"Found {len(video_files)} video(s) to process.\n")

    # Prepare JSON report
    run_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    report = {
        "run_timestamp": run_ts,
        "config": {
            "product_conf":           cfg.product.conf_threshold,
            "defect_conf":            cfg.defect.conf_threshold,
            "buffer":                 cfg.product.buffer_fraction,
            "product_imgsz":          cfg.product.imgsz,
            "defect_imgsz":           cfg.defect.imgsz,
            "defect_frame_threshold": cfg.defect_frame_threshold,
        },
        "products": {},
    }
    json_path = output_dir / f"defect_report_{run_ts.replace(':', '-')}.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_defects = 0
    for i, video_path in enumerate(video_files, 1):
        # Output path: same relative structure, with _annotated suffix
        rel_path    = video_path.relative_to(source_dir)
        out_name    = video_path.stem + cfg.output_suffix + cfg.output_ext
        output_path = output_dir / rel_path.parent / out_name

        print(f"\n[{i}/{len(video_files)}] {rel_path}")
        print(f"  → Output: {output_path.relative_to(output_dir)}")

        stats, defect_log = process_video(video_path, output_path, product_model, defect_model, cfg)

        print(f"  Frames processed     : {stats['frames']}")
        print(f"  Product detected in  : {stats['product_detected']} frames")
        print(f"  Frames with defects  : {stats['frames_with_defects']}")
        print(f"  Total defect boxes   : {stats['total_defects']}")
        overall_defects += stats["total_defects"]

        # Build per-class summary from defect_log
        class_summary: dict = {}
        for event in defect_log:
            cls = event["class"]
            if cls not in class_summary:
                class_summary[cls] = {"count": 0, "max_confidence": 0.0, "min_confidence": 1.0}
            class_summary[cls]["count"]          += 1
            class_summary[cls]["max_confidence"]  = round(max(class_summary[cls]["max_confidence"], event["confidence"]), 4)
            class_summary[cls]["min_confidence"]  = round(min(class_summary[cls]["min_confidence"], event["confidence"]), 4)

        product_key = str(rel_path.parent / video_path.stem)   # e.g. "SAP117c2/1"
        total_frames = stats["frames"]
        frames_with_defects = stats["frames_with_defects"]
        defect_frac = frames_with_defects / total_frames if total_frames > 0 else 0.0

        status = ("ERROR"         if stats["error"]
                  else "NO_PRODUCT"  if stats["product_detected"] == 0
                  else "DEFECT_FOUND" if defect_frac >= cfg.defect_frame_threshold
                  else "CLEAN")

        print(f"  Defect frame ratio   : {frames_with_defects}/{total_frames} "
              f"= {defect_frac*100:.1f}%  (threshold: {cfg.defect_frame_threshold*100:.0f}%)")
        print(f"  Status               : {status}")

        report["products"][product_key] = {
            "status":                    status,
            "source_video":              str(rel_path),
            "annotated_video":           str(output_path.relative_to(output_dir)),
            "total_frames":              stats["frames"],
            "frames_with_product":       stats["product_detected"],
            "frames_with_defects":       stats["frames_with_defects"],
            "defect_frame_ratio":        round(defect_frac, 4),
            "defect_frame_threshold":    cfg.defect_frame_threshold,
            "total_defect_detections":   stats["total_defects"],
            "defect_summary":            class_summary,
            "defect_events":             defect_log,
        }

        # Write JSON after every video so partial results are saved
        with open(json_path, "w") as jf:
            json.dump(report, jf, indent=2)
        print(f"  Report updated → {json_path.name}")

    print("\n" + "=" * 60)
    print(f"All videos processed.")
    print(f"Total defect detections across all videos: {overall_defects}")
    print(f"Output directory: {output_dir}")
    print(f"JSON report     : {json_path}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Two-Phase Video Defect Detection")
    p.add_argument("--source",
                   default="/home/rohit/Rohit/Product_QA/all_products/mp4_videos")
    p.add_argument("--output",
                   default="two_phase_video_output",
                   help="Output directory relative to yolo_v8_test/ (default: two_phase_video_output)")
    p.add_argument("--product-weights", default="output/product_detector/train_results/weights/best.pt")
    p.add_argument("--defect-weights",  default="output/defect_detector/train_results/weights/best.pt")
    p.add_argument("--product-conf",    type=float, default=0.25)
    p.add_argument("--defect-conf",     type=float, default=0.40)
    p.add_argument("--product-imgsz",   type=int,   default=640)
    p.add_argument("--defect-imgsz",    type=int,   default=640)
    p.add_argument("--buffer",          type=float, default=0.05)
    p.add_argument("--skip-every-n",    type=int,   default=1,
                   help="Run detection on every N-th frame (1=all, 2=half). "
                        "Higher values = faster but less temporal coverage.")
    p.add_argument("--defect-threshold", type=float, default=0.10,
                   help="Min fraction of frames with defects to classify as "
                        "DEFECT_FOUND (default: 0.10 = 10%%)")
    p.add_argument("--no-product-box",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = VideoConfig(
        source_dir              = args.source,
        output_dir              = args.output,
        draw_product_box        = not args.no_product_box,
        skip_every_n            = args.skip_every_n,
        defect_frame_threshold  = args.defect_threshold,
        product=ProductDetectorConfig(
            weights         = args.product_weights,
            imgsz           = args.product_imgsz,
            conf_threshold  = args.product_conf,
            buffer_fraction = args.buffer,
        ),
        defect=DefectDetectorConfig(
            weights         = args.defect_weights,
            imgsz           = args.defect_imgsz,
            conf_threshold  = args.defect_conf,
        ),
    )

    run_video_pipeline(cfg)
