"""
ring_inspection/visualise.py
============================
Rendering helpers for RingInspectionModel.predict() output.

draw_polar_visualization   → annotated Sobel strip (polar space)
draw_cartesian_overlay     → per-zone arcs on original image
"""

import cv2
import numpy as np

# Colour palette: circle_1 … circle_4 (BGR)
CIRCLE_COLORS = {
    "circle_1": (0, 200, 0),  # green
    "circle_2": (0, 255, 255),  # cyan
    "circle_3": (0, 165, 255),  # orange
    "circle_4": (255, 0, 255),  # magenta
}
SYNTH_COLOR = (80, 80, 255)  # blue-ish — marks synthesised C3 arcs

STATUS_COLORS = {
    "PASS": (0, 180, 0),
    "REJECT": (0, 0, 200),
    "ERROR": (80, 80, 80),
}


# ──────────────────────────────────────────────────────────────
# Polar-space visualization
# ──────────────────────────────────────────────────────────────


def draw_polar_visualization(
    sobel: np.ndarray,
    per_zone: list,
    labels: list,
    min_r: int,
    num_zones: int,
    panel_width: int = 800,
) -> np.ndarray:
    """
    Render the Sobel strip with per-zone detected peaks overlaid as
    coloured horizontal tick marks.

    Parameters
    ----------
    sobel      : (360, ring_width) Sobel magnitude map
    per_zone   : list[list[int|None]] — per_zone[line_idx][zone_idx] = radius
    labels     : list of circle label strings matching per_zone order
    min_r      : minimum radius offset (for converting radius → column)
    num_zones  : number of angular zones
    panel_width: output image width in pixels

    Returns BGR image.
    """
    h, w = sobel.shape
    # Normalise sobel to 8-bit for display
    vis8 = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    vis_bgr = cv2.cvtColor(vis8, cv2.COLOR_GRAY2BGR)

    zone_rows = h // num_zones

    for line_idx, (label, zone_radii) in enumerate(zip(labels, per_zone)):
        color = CIRCLE_COLORS.get(label, (200, 200, 200))
        for z, r in enumerate(zone_radii):
            if r is None:
                continue
            col = r - min_r
            row_mid = int((z + 0.5) * zone_rows)
            cv2.line(
                vis_bgr,
                (max(0, col - 4), row_mid),
                (min(w - 1, col + 4), row_mid),
                color,
                2,
            )

    # Scale to panel width, keep aspect ratio
    scale = panel_width / w
    new_h = int(h * scale)
    resized = cv2.resize(vis_bgr, (panel_width, new_h), interpolation=cv2.INTER_LINEAR)
    return resized


# ──────────────────────────────────────────────────────────────
# Cartesian (original image) overlay
# ──────────────────────────────────────────────────────────────


def draw_cartesian_overlay(
    img_bgr: np.ndarray,
    cx: int,
    cy: int,
    per_zone: list,
    labels: list,
    status: str,
    num_zones: int,
    synthesized_mask: list = None,
    panel_size: tuple = (600, 600),
) -> np.ndarray:
    """
    Draw per-zone concentric arcs on a zoomed crop of the original image.

    Parameters
    ----------
    img_bgr        : source BGR image
    cx, cy         : ring centre (image coordinates)
    per_zone       : per_zone[line_idx][zone_idx] = radius | None
    labels         : circle label strings
    status         : "PASS" | "REJECT" | "ERROR"
    num_zones      : number of angular zones
    synthesized_mask : optional list of bool per zone for circle_3
                       (True = synthesized → draw in dashed style / different colour)
    panel_size     : (width, height) of output panel

    Returns BGR image.
    """
    # Determine crop radius from outermost detected radii
    max_r = 150
    for zone_radii in per_zone:
        for r in zone_radii:
            if r is not None:
                max_r = max(max_r, r)

    h, w = img_bgr.shape[:2]
    crop_half = int(max_r * 1.25)

    x1, y1 = cx - crop_half, cy - crop_half
    x2, y2 = cx + crop_half, cy + crop_half
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(w, x2), min(h, y2)
    cropped = img_bgr[cy1:cy2, cx1:cx2].copy()

    # Pad if crop extends outside image
    pt = cy1 - y1
    pb = y2 - cy2
    pl = cx1 - x1
    pr = x2 - cx2
    if any(v > 0 for v in [pt, pb, pl, pr]):
        cropped = cv2.copyMakeBorder(
            cropped, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(30, 30, 30)
        )

    ccx, ccy = crop_half, crop_half  # centre in crop-space

    # Draw per-zone arcs
    zone_deg = 360.0 / num_zones
    for line_idx, (label, zone_radii) in enumerate(zip(labels, per_zone)):
        color = CIRCLE_COLORS.get(label, (200, 200, 200))
        for z, r in enumerate(zone_radii):
            if r is None:
                continue
            ang_s = float(z * zone_deg)
            ang_e = float((z + 1) * zone_deg)

            # Synthesized circle 3 → use distinct colour
            is_synth = (
                label == "circle_3"
                and synthesized_mask is not None
                and synthesized_mask[z]
            )
            arc_color = SYNTH_COLOR if is_synth else color

            cv2.ellipse(cropped, (ccx, ccy), (r, r), 0, ang_s, ang_e, arc_color, 2)

    # Centre crosshair
    cv2.drawMarker(
        cropped,
        (ccx, ccy),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=20,
        thickness=2,
    )

    # Resize to panel
    panel_w, panel_h = panel_size
    resized = cv2.resize(cropped, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

    # Status badge
    badge_color = STATUS_COLORS.get(status, (80, 80, 80))
    (bw, bh), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    pad = 8
    cv2.rectangle(
        resized, (8, 8), (8 + bw + pad * 2, 8 + bh + pad * 2), badge_color, -1
    )
    cv2.putText(
        resized,
        status,
        (8 + pad, 8 + bh + pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Synthesized legend
    if synthesized_mask is not None and any(synthesized_mask):
        legend = "C3: SYNTHESIZED"
        (lw, lh), _ = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        y_off = 8 + bh + pad * 2 + 8
        cv2.rectangle(resized, (8, y_off), (10 + lw, y_off + lh + 4), (0, 0, 0), -1)
        cv2.putText(
            resized,
            legend,
            (9, y_off + lh + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            SYNTH_COLOR,
            1,
            cv2.LINE_AA,
        )

    return resized
