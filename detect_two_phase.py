import sys
import cv2
import gc
import argparse
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from ultralytics import YOLO

# ============================================================
# Two-Phase Defect Detection Pipeline
# ============================================================
# Phase 1 — Product Detector:
#   Finds the product bounding box in the full original image.
#
# Phase 2 — Defect Detector:
#   Crops the product region (+ buffer), runs defect detection
#   on the crop, then maps defect coordinates back to the
#   original image space.
#
# Final output: original image with product bbox + defect bboxes
# drawn on top, saved to two_phase_output/ (mirrored structure).
# ============================================================


# ──────────────────────────────────────────────────────────────
# CONFIGURATION — all tunable parameters in one place
# ──────────────────────────────────────────────────────────────

@dataclass
class ProductDetectorConfig:
    weights:         str   = "output/product_detector/train_results/weights/best.pt"
    imgsz:           int   = 640
    conf_threshold:  float = 0.25   # min confidence to accept a product detection
    buffer_fraction: float = 0.05   # fraction of bbox size added as padding on each side

@dataclass
class DefectDetectorConfig:
    weights:         str   = "output/defect_detector/train_results/weights/best.pt"
    imgsz:           int   = 640
    conf_threshold:  float = 0.40   # higher than product — reduce defect false positives

@dataclass
class VisualisationConfig:
    # Product bbox
    product_color:      tuple = (0, 255, 0)     # green
    product_thickness:  int   = 3

    # Defect bbox colours per class index
    defect_colors: tuple = (
        (0,  80, 255),   # class 0: dentporosity  — orange-red
        (255, 50,  50),  # class 1: unclearsurface — blue
    )
    defect_thickness: int   = 2
    font_scale:       float = 0.65
    font_thickness:   int   = 2
    label_bg:         bool  = True   # draw filled background behind label text

@dataclass
class PipelineConfig:
    product:       ProductDetectorConfig = None
    defect:        DefectDetectorConfig  = None
    vis:           VisualisationConfig   = None
    source_dir:    str  = "all_products"          # input directory (full original images)
    output_dir:    str  = "two_phase_output"      # where annotated images are saved
    jpeg_quality:  int  = 95
    draw_product_box: bool = True   # set False to only draw defect boxes on output

    def __post_init__(self):
        if self.product is None: self.product = ProductDetectorConfig()
        if self.defect  is None: self.defect  = DefectDetectorConfig()
        if self.vis     is None: self.vis     = VisualisationConfig()


# ──────────────────────────────────────────────────────────────
# COORDINATE UTILITIES
# ──────────────────────────────────────────────────────────────

def compute_crop_box(img_h: int, img_w: int,
                     x1: int, y1: int, x2: int, y2: int,
                     buf_frac: float) -> tuple[int, int, int, int]:
    """
    Expand a bounding box by buf_frac of its own size on each side,
    clamped to the image boundary.
    Returns (cx1, cy1, cx2, cy2) — the actual crop coordinates in
    the original image.
    """
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * buf_frac)
    pad_y = int(bh * buf_frac)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(img_w, x2 + pad_x)
    cy2 = min(img_h, y2 + pad_y)
    return cx1, cy1, cx2, cy2


def crop_to_original(dx1: int, dy1: int, dx2: int, dy2: int,
                     crop_x1: int, crop_y1: int) -> tuple[int, int, int, int]:
    """
    Convert defect bbox coordinates from crop-space back to
    original full-image space by adding the crop origin offset.
    """
    return (
        crop_x1 + dx1,
        crop_y1 + dy1,
        crop_x1 + dx2,
        crop_y1 + dy2,
    )


# ──────────────────────────────────────────────────────────────
# DRAWING UTILITIES
# ──────────────────────────────────────────────────────────────

def draw_box(img, x1: int, y1: int, x2: int, y2: int,
             label: str, color: tuple, thickness: int,
             font_scale: float, font_thickness: int, label_bg: bool):
    """Draw a labelled bounding box on img in-place."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    tx = x1
    ty = max(y1 - 6, th + 6)

    if label_bg:
        cv2.rectangle(img, (tx, ty - th - baseline), (tx + tw, ty + baseline),
                      color, cv2.FILLED)
        text_color = (255, 255, 255)
    else:
        text_color = color

    cv2.putText(img, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                font_thickness, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ──────────────────────────────────────────────────────────────

def detect_product(model: YOLO, img_path: str,
                   cfg: ProductDetectorConfig) -> Optional[tuple]:
    """
    Run the product detector on img_path.
    Returns (x1, y1, x2, y2, conf) of the highest-confidence box,
    or None if nothing was detected above cfg.conf_threshold.
    """
    results = model.predict(
        source=img_path,
        imgsz=cfg.imgsz,
        conf=cfg.conf_threshold,
        verbose=False,
        device="",
    )
    best_box  = None
    best_conf = -1.0
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box  = box

    if best_box is None:
        return None
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    return x1, y1, x2, y2, best_conf


def detect_defects(model: YOLO, crop_img,
                   cfg: DefectDetectorConfig) -> list[dict]:
    """
    Run the defect detector on a cropped image (numpy array).
    Returns a list of dicts with keys: x1, y1, x2, y2, conf, cls_id, cls_name
    """
    results = model.predict(
        source=crop_img,
        imgsz=cfg.imgsz,
        conf=cfg.conf_threshold,
        verbose=False,
        device="",
    )
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        names = result.names  # {0: 'dentporosity', 1: 'unclearsurface'}
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf":     float(box.conf[0]),
                "cls_id":   cls_id,
                "cls_name": names.get(cls_id, str(cls_id)),
            })
    return detections


# ──────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────

def run_pipeline(cfg: PipelineConfig):
    workspace_dir = Path(__file__).resolve().parent
    product_weights = workspace_dir / cfg.product.weights
    defect_weights  = workspace_dir / cfg.defect.weights
    source_dir      = workspace_dir / cfg.source_dir
    output_dir      = workspace_dir / cfg.output_dir

    print("=" * 60)
    print("   Two-Phase Defect Detection Pipeline")
    print("=" * 60)
    print(f"Source          : {source_dir}")
    print(f"Output          : {output_dir}")
    print(f"Product model   : {product_weights}")
    print(f"Defect model    : {defect_weights}")
    print(f"Product conf    : {cfg.product.conf_threshold}")
    print(f"Defect conf     : {cfg.defect.conf_threshold}")
    print(f"Product buffer  : {cfg.product.buffer_fraction * 100:.0f}%")
    print("-" * 60)

    # Validate paths
    for path, label in [(product_weights, "Product weights"),
                        (defect_weights,  "Defect weights"),
                        (source_dir,      "Source directory")]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    # Load both models
    print("Loading models...")
    product_model = YOLO(str(product_weights))
    defect_model  = YOLO(str(defect_weights))
    print("Both models loaded.\n")

    # Gather source images
    image_files = list(source_dir.rglob("*.jpg")) + list(source_dir.rglob("*.JPG"))
    if not image_files:
        print(f"No JPG images found in {source_dir}")
        sys.exit(0)
    print(f"Found {len(image_files)} images.\n")

    # Stats
    stats = {"no_product": 0, "no_defects": 0, "defects_found": 0, "skipped": 0}
    total_defects = 0

    for i, img_path in enumerate(image_files, 1):
        rel_path    = img_path.relative_to(source_dir)
        output_path = output_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{i:>3}/{len(image_files)}] {rel_path}")

        # Read full original image
        orig = cv2.imread(str(img_path))
        if orig is None:
            print("  → SKIP (could not read)")
            stats["skipped"] += 1
            continue

        # ── Phase 1: Product Detection ──────────────────────────
        product_result = detect_product(product_model, str(img_path), cfg.product)

        if product_result is None:
            print("  → Phase 1: no product detected — saving original")
            cv2.imwrite(str(output_path), orig, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
            stats["no_product"] += 1
            continue

        px1, py1, px2, py2, p_conf = product_result
        h, w = orig.shape[:2]

        # Compute the actual crop coordinates (with buffer)
        cx1, cy1, cx2, cy2 = compute_crop_box(
            h, w, px1, py1, px2, py2, cfg.product.buffer_fraction
        )
        print(f"  → Phase 1: product at ({px1},{py1})-({px2},{py2})  conf={p_conf:.2f}")

        # Crop for defect inference
        crop = orig[cy1:cy2, cx1:cx2].copy()

        # ── Phase 2: Defect Detection (on crop) ─────────────────
        defects = detect_defects(defect_model, crop, cfg.defect)

        # Map defect coords back to original image space
        remapped = []
        for d in defects:
            ox1, oy1, ox2, oy2 = crop_to_original(
                d["x1"], d["y1"], d["x2"], d["y2"], cx1, cy1
            )
            remapped.append({**d, "x1": ox1, "y1": oy1, "x2": ox2, "y2": oy2})

        print(f"  → Phase 2: {len(remapped)} defect(s) detected")
        for d in remapped:
            print(f"       {d['cls_name']}  conf={d['conf']:.2f}  "
                  f"({d['x1']},{d['y1']})-({d['x2']},{d['y2']})")

        # ── Draw on original image ───────────────────────────────
        annotated = orig.copy()
        v = cfg.vis

        # Draw product bbox
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

        # Draw defect bboxes (in original image space)
        for d in remapped:
            color = (v.defect_colors[d["cls_id"]]
                     if d["cls_id"] < len(v.defect_colors)
                     else (255, 255, 0))
            draw_box(
                annotated, d["x1"], d["y1"], d["x2"], d["y2"],
                label=f"{d['cls_name']} {d['conf']:.2f}",
                color=color,
                thickness=v.defect_thickness,
                font_scale=v.font_scale,
                font_thickness=v.font_thickness,
                label_bg=v.label_bg,
            )

        cv2.imwrite(str(output_path), annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])

        if remapped:
            stats["defects_found"] += 1
            total_defects += len(remapped)
        else:
            stats["no_defects"] += 1

        # Free memory
        del orig, crop, annotated
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  Images with defects     : {stats['defects_found']}")
    print(f"  Images clean (no defect): {stats['no_defects']}")
    print(f"  No product detected     : {stats['no_product']}")
    print(f"  Skipped (read errors)   : {stats['skipped']}")
    print(f"  Total defect detections : {total_defects}")
    print(f"  Output directory        : {output_dir}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────
# ENTRY POINT — override config values via CLI args
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Two-Phase Defect Detection Pipeline")
    p.add_argument("--source",          default="all_products",
                   help="Source image directory (default: all_products)")
    p.add_argument("--output",          default="two_phase_output",
                   help="Output directory (default: two_phase_output)")
    p.add_argument("--product-weights", default="output/product_detector/train_results/weights/best.pt")
    p.add_argument("--defect-weights",  default="output/defect_detector/train_results/weights/best.pt")
    p.add_argument("--product-conf",    type=float, default=0.25)
    p.add_argument("--defect-conf",     type=float, default=0.40)
    p.add_argument("--product-imgsz",   type=int,   default=640)
    p.add_argument("--defect-imgsz",    type=int,   default=640)
    p.add_argument("--buffer",          type=float, default=0.05,
                   help="Product crop buffer as fraction (default: 0.05 = 5%%)")
    p.add_argument("--no-product-box",  action="store_true",
                   help="Do not draw the product bounding box on output")
    p.add_argument("--jpeg-quality",    type=int,   default=95)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = PipelineConfig(
        source_dir       = args.source,
        output_dir       = args.output,
        jpeg_quality     = args.jpeg_quality,
        draw_product_box = not args.no_product_box,
        product=ProductDetectorConfig(
            weights          = args.product_weights,
            imgsz            = args.product_imgsz,
            conf_threshold   = args.product_conf,
            buffer_fraction  = args.buffer,
        ),
        defect=DefectDetectorConfig(
            weights          = args.defect_weights,
            imgsz            = args.defect_imgsz,
            conf_threshold   = args.defect_conf,
        ),
    )

    run_pipeline(cfg)
