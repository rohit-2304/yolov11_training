"""
circle_qa/visualise.py
======================
Annotate a BGR image with CircleQA results.
"""

import cv2
import numpy as np

LINE_COLORS = {
    "inner_surface_bottom":   (0,   200,   0),   # green
    "inner_surface_top":      (0,   255, 255),   # cyan
    "light_band_boundary":    (0,   165, 255),   # orange
    "outer_surface_boundary": (255,   0, 255),   # magenta
}


def draw_result(img_bgr: np.ndarray, result: dict,
                panel_size: tuple = (480, 480)) -> np.ndarray:
    """
    Render a zoomed annotated crop showing:
      - Concentric circle arcs (per-zone, colour-coded by boundary)
      - Centre crosshair
      - PASS / FAIL / ERROR badge
      - First defect description (if any)

    Parameters
    ----------
    img_bgr    : source BGR image (the cropped product image)
    result     : dict returned by CircleQA.run()
    panel_size : (width, height) of the returned annotation panel

    Returns
    -------
    A BGR image of shape (panel_h, panel_w, 3).
    """
    lines      = result.get("lines", [])
    defects    = result.get("defects", [])
    status     = result.get("status", "ERROR")
    center     = result.get("center")
    num_zones  = result.get("num_zones", 16)

    # ── Determine crop region ───────────────────────────────────
    max_r = 150
    for ln in lines:
        for r in ln.per_zone_radii:
            if r is not None:
                max_r = max(max_r, r)

    h, w = img_bgr.shape[:2]

    if center is not None:
        cx, cy = center
    else:
        cx, cy = w // 2, h // 2

    crop_half = int(max_r * 1.25)
    x1, y1 = cx - crop_half, cy - crop_half
    x2, y2 = cx + crop_half, cy + crop_half

    crop_x1, crop_y1 = max(0, x1), max(0, y1)
    crop_x2, crop_y2 = min(w, x2), min(h, y2)
    cropped = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    # Pad if the crop extends outside the image
    pad_top    = crop_y1 - y1
    pad_bottom = y2 - crop_y2
    pad_left   = crop_x1 - x1
    pad_right  = x2 - crop_x2
    if any(p > 0 for p in [pad_top, pad_bottom, pad_left, pad_right]):
        cropped = cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(30, 30, 30)
        )

    ccx, ccy = crop_half, crop_half   # centre in crop space

    # ── Draw per-zone arcs ──────────────────────────────────────
    for z in range(num_zones):
        ang_s = z * (360.0 / num_zones)
        ang_e = (z + 1) * (360.0 / num_zones)
        for ln in lines:
            zone_r = ln.per_zone_radii[z]
            if zone_r is None:
                continue
            color = LINE_COLORS.get(ln.label, (200, 200, 200))
            cv2.ellipse(cropped, (ccx, ccy), (zone_r, zone_r),
                        0, ang_s, ang_e, color, 2)

    # ── Centre crosshair ────────────────────────────────────────
    cv2.drawMarker(cropped, (ccx, ccy), (0, 0, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    # ── Resize to panel ─────────────────────────────────────────
    panel_w, panel_h = panel_size
    resized = cv2.resize(cropped, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

    # ── PASS / FAIL / ERROR badge ───────────────────────────────
    badge_color = {
        "PASS":  (0, 140, 0),
        "FAIL":  (0, 0, 180),
        "ERROR": (80, 80, 80),
    }.get(status, (80, 80, 80))

    (bw, bh), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    pad = 8
    cv2.rectangle(resized, (8, 8), (8 + bw + pad * 2, 8 + bh + pad * 2),
                  badge_color, -1)
    cv2.putText(resized, status, (8 + pad, 8 + bh + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # ── First defect line ────────────────────────────────────────
    if defects:
        short = defects[0].split(":")[0].strip()[:60]
        (dt_w, dt_h), _ = cv2.getTextSize(short, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        y_off = 8 + bh + pad * 2 + 8
        cv2.rectangle(resized, (8, y_off), (10 + dt_w, y_off + dt_h + 4),
                      (0, 0, 0), -1)
        cv2.putText(resized, short, (9, y_off + dt_h + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 80, 255), 1, cv2.LINE_AA)

    return resized
