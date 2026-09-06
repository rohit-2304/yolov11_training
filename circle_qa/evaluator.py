"""
circle_qa/evaluator.py
======================
Self-contained CircleQA evaluator extracted from app/pipeline/circle_qc.py.
All algorithmic logic lives here; no app-specific imports.

Usage:
    from circle_qa import CircleQA, DEFAULT_PARAMS

    qa = CircleQA()                          # use defaults
    qa = CircleQA(params={"canny_lo": 15})   # override specific params
    result = qa.run(image_bgr)
"""

import cv2
import numpy as np
import scipy.signal
from dataclasses import dataclass
from typing import Optional, List, Dict
from itertools import combinations

# ──────────────────────────────────────────────────────────────
# TUNABLE PARAMETERS  (baseline / fallback values)
# ──────────────────────────────────────────────────────────────
DEFAULT_PARAMS: dict = dict(
    # ── Pre-processing ──────────────────────────────────────────
    clahe_clip        = 1.5,
    clahe_tile        = 8,
    bilateral_d       = 5,
    bilateral_sigma   = 75,
    canny_lo          = 20,
    canny_hi          = 80,
    dilate_iters      = 1,

    # ── Circle extraction ───────────────────────────────────────
    min_perimeter     = 400,   # px — ignore short contours
    min_circularity   = 0.75,

    # ── Polar strip ─────────────────────────────────────────────
    radial_pad_outer  = 20,    # px added beyond outermost circle
    radial_pad_inner  = 60,    # px subtracted from innermost circle

    # ── Line detection in polar strip ───────────────────────────
    num_zones          = 16,
    peak_height_global = 15,   # Sobel magnitude for global profile
    peak_height_zone   = 15,   # Sobel magnitude per zone
    peak_distance      = 10,   # min column distance between peaks
    peak_prominence    = 8,    # minimum peak prominence
    max_lines          = 4,    # keep only N strongest global peaks

    # ── Shadow suppression ──────────────────────────────────────
    shadow_min_zone_fraction = 0.6,

    # ── Defect thresholds (relative to ring_width = max_r - min_r) ──
    elongation_std_frac   = 0.04,
    offcenter_dist_frac   = 0.03,
    discontinuity_frac    = 0.12,
    band_gap_std_frac     = 0.018,
    inner_band_gap_std_frac = 0.06,
    inner_disc_frac       = 0.12,

    # ── Geometric structure validation ─────────────────────────
    max_inner_lip_gap_frac  = 0.18,
    min_rubber_gap_frac     = 0.06,
    min_metal_gap_frac      = 0.025,
    min_outer_span_frac     = 0.10,

    # ── Missing-boundary recovery ───────────────────────────────
    recover_missing_peaks = True,
    gap_scan_min_height   = 8,
    gap_scan_min_zones    = 8,
)


# ──────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────
@dataclass
class RingLine:
    """One detected concentric boundary."""
    label: str
    global_radius: int
    per_zone_radii: list   # radius per angular zone (or None if missing)
    zone_angles: list      # (start_deg, end_deg) per zone

    @property
    def detected_zones(self):
        return [r for r in self.per_zone_radii if r is not None]

    @property
    def missing_zones(self):
        return sum(1 for r in self.per_zone_radii if r is None)

    @property
    def radius_std(self):
        d = self.detected_zones
        return float(np.std(d)) if len(d) > 1 else 0.0

    @property
    def radius_mean(self):
        d = self.detected_zones
        return float(np.mean(d)) if d else float(self.global_radius)


# ──────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────

def _col_to_radius(col: int, min_r: int) -> int:
    return int(col + min_r)


def _preprocess(img_bgr: np.ndarray, p: dict) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=p['clahe_clip'],
        tileGridSize=(p['clahe_tile'], p['clahe_tile'])
    )
    enhanced = clahe.apply(gray)
    blurred = cv2.bilateralFilter(
        enhanced,
        d=p['bilateral_d'],
        sigmaColor=p['bilateral_sigma'],
        sigmaSpace=p['bilateral_sigma']
    )
    return blurred


def _detect_circle_center(blurred: np.ndarray, bbox, p: dict):
    """
    Detect shared centre of concentric circles from Canny edges + contour fitting.
    Returns (cx, cy, max_r, min_r, all_radii).
    """
    edges = cv2.Canny(blurred, p['canny_lo'], p['canny_hi'])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=p['dilate_iters'])
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centers_x, centers_y, valid_radii = [], [], []
    for c in contours:
        peri = cv2.arcLength(c, closed=True)
        if peri < p['min_perimeter']:
            continue
        area = cv2.contourArea(c)
        if peri == 0:
            continue
        circularity = (4 * np.pi * area) / (peri ** 2)
        if circularity > p['min_circularity']:
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                    continue
            centers_x.append(cx)
            centers_y.append(cy)
            valid_radii.append(r)

    if not valid_radii:
        raise RuntimeError("No circular contours detected in target region.")

    if len(centers_x) >= 3:
        cx = int(np.mean(centers_x))
        cy = int(np.mean(centers_y))
    else:
        i = np.argmax(valid_radii)
        cx, cy = int(centers_x[i]), int(centers_y[i])

    max_r = int(max(valid_radii) + p['radial_pad_outer'])
    min_r = max(0, int(min(valid_radii) - p['radial_pad_inner']))
    return cx, cy, max_r, min_r, valid_radii


def _build_polar_sobel(blurred: np.ndarray, cx: int, cy: int,
                        max_r: int, min_r: int):
    polar_full = cv2.warpPolar(
        src=blurred,
        dsize=(max_r, 360),
        center=(cx, cy),
        maxRadius=max_r,
        flags=cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR
    )
    polar = polar_full[:, min_r:]
    sobel = cv2.Sobel(polar, cv2.CV_64F, 1, 0, ksize=3)
    sobel = cv2.convertScaleAbs(sobel)
    return polar, sobel


def _detect_global_lines(sobel: np.ndarray, min_r: int, ring_width: int, p: dict):
    profile = np.median(sobel, axis=0)
    peaks, _ = scipy.signal.find_peaks(
        profile,
        height=p['peak_height_global'],
        distance=p['peak_distance'],
        prominence=p['peak_prominence'],
    )
    heights = profile[peaks]
    n = p['max_lines']
    if len(peaks) > n:
        top_idx = np.argsort(heights)[-n:]
        peaks = peaks[np.sort(top_idx)]

    recovered_cols: set = set()
    if p.get('recover_missing_peaks', True) and 0 < len(peaks) < n:
        num_zones  = p['num_zones']
        zone_rows  = sobel.shape[0] // num_zones
        min_height = p.get('gap_scan_min_height', 8)
        min_zones  = p.get('gap_scan_min_zones', 8)
        half_dist  = max(3, p['peak_distance'] // 2)
        n_cols     = sobel.shape[1]

        recovered = list(map(int, peaks))
        for _ in range(n - len(peaks)):
            sorted_r = sorted(recovered)
            gaps = []
            for i in range(len(sorted_r) - 1):
                lo, hi = sorted_r[i] + half_dist, sorted_r[i + 1] - half_dist
                if hi > lo:
                    gaps.append((lo, hi, hi - lo))
            last_lo = sorted_r[-1] + half_dist
            if n_cols - 1 > last_lo:
                gaps.append((last_lo, n_cols - 1, n_cols - 1 - last_lo))
            if not gaps:
                break
            gaps.sort(key=lambda x: -x[2])
            placed = False
            for lo, hi, _ in gaps:
                zone_cols = []
                for z in range(num_zones):
                    row_s = z * zone_rows
                    row_e = min((z + 1) * zone_rows, sobel.shape[0])
                    zp = np.mean(sobel[row_s:row_e, :], axis=0)
                    window = zp[lo:hi + 1]
                    if len(window) == 0 or window.max() < min_height:
                        continue
                    zone_cols.append(int(np.argmax(window)) + lo)
                if len(zone_cols) >= min_zones:
                    new_col = int(np.median(zone_cols))
                    if all(abs(new_col - r) >= half_dist for r in recovered):
                        recovered.append(new_col)
                        recovered_cols.add(new_col)
                        placed = True
                        break
            if not placed:
                break
        peaks = np.array(sorted(recovered))

    radii = [_col_to_radius(int(pk), min_r) for pk in peaks]
    return sorted(radii), peaks, recovered_cols


def _validate_peak_structure(global_radii: list, ring_width: int, p: dict,
                              profile=None, min_r: int = 0, all_peak_cols=None):
    min_rubber = p.get('min_rubber_gap_frac', 0.06) * ring_width
    min_metal  = p.get('min_metal_gap_frac',  0.025) * ring_width
    min_span   = p.get('min_outer_span_frac', 0.08)  * ring_width
    max_lip    = p.get('max_inner_lip_gap_frac', 0.18) * ring_width

    def is_valid(r):
        if len(r) < 4:
            return False
        s = sorted(r)
        return (
            s[1] - s[0] <= max_lip   and
            s[2] - s[1] >= min_rubber and
            s[3] - s[2] >= min_metal  and
            s[3] - s[1] >= min_span
        )

    if is_valid(global_radii):
        return sorted(global_radii)

    if profile is None or all_peak_cols is None or len(all_peak_cols) < 4:
        return sorted(global_radii)

    candidates = sorted([_col_to_radius(int(c), min_r) for c in all_peak_cols])
    best_combo, best_score = None, -1.0
    for combo in combinations(candidates, 4):
        if not is_valid(list(combo)):
            continue
        score = sum(float(profile[c - min_r]) if 0 <= (c - min_r) < len(profile) else 0.0
                    for c in combo)
        if score > best_score:
            best_score = score
            best_combo = list(combo)

    return best_combo if best_combo is not None else sorted(global_radii)


def _detect_per_zone_lines(sobel, min_r, global_peaks_cols, p, recovered_cols=None):
    if recovered_cols is None:
        recovered_cols = set()
    num_zones  = p['num_zones']
    zone_rows  = sobel.shape[0] // num_zones
    n_lines    = len(global_peaks_cols)
    sorted_cols = sorted(global_peaks_cols)

    if len(sorted_cols) >= 2:
        min_gap = int(np.min(np.diff(sorted_cols)))
    else:
        min_gap = 30
    default_hg = max(6, min_gap // 2)

    per_anchor_win = []
    for i, col in enumerate(sorted_cols):
        left_room  = (col - sorted_cols[i - 1] - 1) // 2 if i > 0 else 9999
        right_room = (sorted_cols[i + 1] - col - 1) // 2 if i < len(sorted_cols) - 1 else 9999
        if col in recovered_cols:
            wide   = max(default_hg * 2, 20)
            lo_off = min(default_hg, left_room)
            hi_off = min(wide, right_room)
        else:
            sym    = min(default_hg, left_room, right_room)
            lo_off = sym
            hi_off = sym
        per_anchor_win.append((lo_off, hi_off))

    col_to_aidx = {col: i for i, col in enumerate(sorted_cols)}
    anchor_win  = [per_anchor_win[col_to_aidx[c]] for c in global_peaks_cols]
    per_zone    = [[None] * num_zones for _ in range(n_lines)]

    for z in range(num_zones):
        row_s = z * zone_rows
        row_e = min((z + 1) * zone_rows, sobel.shape[0])
        zone_profile = np.mean(sobel[row_s:row_e, :], axis=0)
        zone_peaks, _ = scipy.signal.find_peaks(zone_profile,
                                                 height=p['peak_height_zone'],
                                                 distance=3, prominence=4)
        anchors     = np.array(global_peaks_cols, dtype=float)
        assigned_zp = set()

        if len(zone_peaks) > 0:
            zp     = np.array(zone_peaks, dtype=float)
            signed = zp[None, :] - anchors[:, None]
            order  = np.argsort(np.abs(signed).ravel())
            for flat_idx in order:
                a_idx  = flat_idx // len(zp)
                zp_idx = flat_idx % len(zp)
                lo_off, hi_off = anchor_win[a_idx]
                sd = signed[a_idx, zp_idx]
                if sd < -lo_off or sd > hi_off:
                    continue
                if per_zone[a_idx][z] is not None:
                    continue
                if zp_idx in assigned_zp:
                    continue
                per_zone[a_idx][z] = _col_to_radius(int(zone_peaks[zp_idx]), min_r)
                assigned_zp.add(zp_idx)

        for a_idx, g_col in enumerate(global_peaks_cols):
            if per_zone[a_idx][z] is not None:
                continue
            lo_off, hi_off = anchor_win[a_idx]
            col_lo = max(0, g_col - lo_off)
            col_hi = min(zone_profile.shape[0], g_col + hi_off + 1)
            region = zone_profile[col_lo:col_hi].copy()

            for other_idx, other_r in enumerate(per_zone):
                if other_idx == a_idx:
                    continue
                other_r_z = other_r[z]
                if other_r_z is None:
                    continue
                other_col = other_r_z - min_r
                o_lo, o_hi = anchor_win[other_idx]
                mask_lo = max(col_lo, other_col - o_lo) - col_lo
                mask_hi = min(col_hi, other_col + o_hi + 1) - col_lo
                if mask_hi > mask_lo >= 0:
                    region[mask_lo:mask_hi] = 0

            if region.max() < p['peak_height_zone']:
                continue
            weights   = region.astype(float)
            cols_range = np.arange(col_lo, col_hi)
            com_col   = int(round(np.sum(cols_range * weights) / np.sum(weights)))
            per_zone[a_idx][z] = _col_to_radius(com_col, min_r)

    # Interpolate single missing zones
    for line_idx in range(n_lines):
        radii    = per_zone[line_idx]
        new_radii = list(radii)
        for z in range(num_zones):
            if radii[z] is None:
                left_val  = radii[(z - 1) % num_zones]
                right_val = radii[(z + 1) % num_zones]
                if left_val is not None and right_val is not None:
                    new_radii[z] = int(round((left_val + right_val) / 2.0))
        per_zone[line_idx] = new_radii

    return per_zone


def _label_lines(global_radii: list):
    labels = [
        "inner_surface_bottom",
        "inner_surface_top",
        "light_band_boundary",
        "outer_surface_boundary",
    ]
    return [(labels[i] if i < len(labels) else f"extra_line_{i}", r)
            for i, r in enumerate(sorted(global_radii))]


def _check_defects(lines: list, ring_width: int, cx: int, cy: int, p: dict):
    defects   = []
    light_band = next((ln for ln in lines if ln.label == "light_band_boundary"), None)
    outer_band = next((ln for ln in lines if ln.label == "outer_surface_boundary"), None)
    inner_top  = next((ln for ln in lines if ln.label == "inner_surface_top"), None)

    if light_band is None:
        defects.append("light_band_boundary NOT DETECTED – severe defect or missing band")
        return defects

    n_zones  = len(light_band.per_zone_radii)
    detected = light_band.detected_zones
    missing  = light_band.missing_zones

    def fit_circle_polar(radii, zone_angles_mid):
        A, b = [], []
        for r, ang in zip(radii, zone_angles_mid):
            if r is not None:
                ang_rad = np.deg2rad(ang)
                A.append([1.0, np.cos(ang_rad), np.sin(ang_rad)])
                b.append(r)
        if len(b) < 4:
            return 0.0, 0.0, 0.0
        A, b = np.array(A), np.array(b)
        res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return float(res[0]), float(res[1]), float(res[2])

    zone_angles_mid = [
        (z * 360.0 / n_zones + (z + 1) * 360.0 / n_zones) / 2.0
        for z in range(n_zones)
    ]

    # 1. DISCONTINUITY / OVERLAP CHECK
    overlap_zones = []
    if missing > 0:
        R0, dx, dy = fit_circle_polar(light_band.per_zone_radii, zone_angles_mid)
        discont_zones = []
        for z in range(n_zones):
            if light_band.per_zone_radii[z] is None:
                if outer_band is not None and outer_band.per_zone_radii[z] is not None:
                    ang_rad   = np.deg2rad(zone_angles_mid[z])
                    r_fit     = R0 + dx * np.cos(ang_rad) + dy * np.sin(ang_rad)
                    fit_gap   = outer_band.per_zone_radii[z] - r_fit
                    if (fit_gap / ring_width) < 0.14:
                        overlap_zones.append(z)
                    else:
                        discont_zones.append(z)
                else:
                    discont_zones.append(z)
        if overlap_zones:
            defects.append(
                f"OVERLAP between light-band and outer boundary: "
                f"{len(overlap_zones)}/{n_zones} zones"
            )
        if discont_zones:
            missing_frac = len(discont_zones) / n_zones
            if missing_frac > p['discontinuity_frac']:
                defects.append(
                    f"DISCONTINUOUS light band: {len(discont_zones)}/{n_zones} zones missing "
                    f"({missing_frac*100:.0f}% > threshold {p['discontinuity_frac']*100:.0f}%)"
                )

    if len(detected) < 3:
        return defects

    # 2. ELONGATION CHECK
    std_frac = light_band.radius_std / ring_width
    if std_frac > p['elongation_std_frac']:
        defects.append(
            f"ELONGATED/COMPRESSED light band: radius std={light_band.radius_std:.1f}px "
            f"({std_frac*100:.1f}% of ring width, threshold {p['elongation_std_frac']*100:.1f}%)"
        )

    # 3. OFF-CENTER CHECK
    A, b = [], []
    for z_idx, r in enumerate(light_band.per_zone_radii):
        if r is not None:
            ang_rad = np.deg2rad(zone_angles_mid[z_idx])
            A.append([1.0, np.cos(ang_rad), np.sin(ang_rad)])
            b.append(r)
    if len(b) >= 6:
        res, _, _, _ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
        offset      = np.sqrt(res[1]**2 + res[2]**2)
        offset_frac = offset / ring_width
        if offset_frac > p['offcenter_dist_frac']:
            defects.append(
                f"OFF-CENTER light band: offset={offset:.1f}px "
                f"({offset_frac*100:.1f}% of ring width, threshold {p['offcenter_dist_frac']*100:.1f}%)"
            )

    # 4. OUTER BAND-GAP UNIFORMITY
    if outer_band is not None:
        gaps = []
        for z_idx in range(n_zones):
            r_lbb = light_band.per_zone_radii[z_idx]
            r_osb = outer_band.per_zone_radii[z_idx]
            if r_osb is not None:
                gaps.append(r_osb - r_lbb if r_lbb is not None else 0.0)
        if len(gaps) >= 8:
            gap_arr   = np.array(gaps, dtype=float)
            gap_std   = float(np.std(gap_arr))
            gap_mean  = float(np.mean(gap_arr))
            std_frac2 = gap_std / ring_width
            if std_frac2 > p['band_gap_std_frac']:
                defects.append(
                    f"UNEVEN BAND GAP: gap std={gap_std:.1f}px (mean={gap_mean:.1f}px, "
                    f"std/ring_width={std_frac2*100:.1f}% > threshold {p['band_gap_std_frac']*100:.1f}%)"
                )

    # 5. INNER BAND-GAP UNIFORMITY
    if inner_top is not None:
        inner_gaps = [
            light_band.per_zone_radii[z] - inner_top.per_zone_radii[z]
            for z in range(n_zones)
            if inner_top.per_zone_radii[z] is not None
            and light_band.per_zone_radii[z] is not None
            and light_band.per_zone_radii[z] > inner_top.per_zone_radii[z]
        ]
        if len(inner_gaps) >= 8:
            ig_arr  = np.array(inner_gaps, dtype=float)
            ig_std  = float(np.std(ig_arr))
            ig_mean = float(np.mean(ig_arr))
            ig_frac = ig_std / ring_width
            if ig_frac > p['inner_band_gap_std_frac']:
                defects.append(
                    f"UNEVEN INNER BAND GAP: std={ig_std:.1f}px (mean={ig_mean:.1f}px, "
                    f"std/ring_width={ig_frac*100:.1f}% > threshold {p['inner_band_gap_std_frac']*100:.1f}%)"
                )

    # 6. INNER RING DISCONTINUITY
    if inner_top is not None:
        ist_frac = inner_top.missing_zones / n_zones
        if ist_frac > p['inner_disc_frac']:
            defects.append(
                f"DISCONTINUOUS inner ring: {inner_top.missing_zones}/{n_zones} zones missing "
                f"({ist_frac*100:.0f}% > threshold {p['inner_disc_frac']*100:.0f}%)"
            )

    return defects


# ──────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────

class CircleQA:
    """
    Plug-and-play concentric circle QA evaluator.

    Parameters
    ----------
    params : dict, optional
        Override any key from DEFAULT_PARAMS.  Only the supplied keys are
        replaced; everything else falls back to DEFAULT_PARAMS.

    Example
    -------
    >>> from circle_qa import CircleQA
    >>> qa = CircleQA(params={"canny_lo": 15, "canny_hi": 70})
    >>> result = qa.run(image_bgr)
    >>> print(result["status"], result["defects"])
    """

    def __init__(self, params: Optional[dict] = None):
        self.p = DEFAULT_PARAMS.copy()
        if params:
            for k, v in params.items():
                if k not in self.p:
                    raise ValueError(f"Unknown CircleQA param: '{k}'. "
                                     f"Valid keys: {sorted(self.p.keys())}")
                self.p[k] = v

    # ── run ─────────────────────────────────────────────────────
    def run(self, img_bgr: np.ndarray) -> dict:
        """
        Run full circle QA on a single BGR image.

        The image is assumed to already be cropped to the product region
        (e.g. output of crop_products.py).  No bbox filtering is applied.

        Returns
        -------
        dict with keys:
            status         : "PASS" | "FAIL" | "ERROR"
            defects        : list[str]  — human-readable defect descriptions
            center         : (cx, cy) int tuple or None
            ring_width     : int — max_r - min_r in pixels
            lines          : list[RingLine]
            min_r          : int
            num_zones      : int
        """
        if img_bgr is None or img_bgr.size == 0:
            return self._error("Input image is empty.")

        try:
            blurred = _preprocess(img_bgr, self.p)
            cx, cy, max_r, min_r, _ = _detect_circle_center(blurred, None, self.p)
        except RuntimeError as e:
            return self._error(str(e))
        except Exception as e:
            return self._error(f"Preprocessing / circle detection failed: {e}")

        try:
            ring_width = max_r - min_r
            _, sobel   = _build_polar_sobel(blurred, cx, cy, max_r, min_r)

            global_radii, global_peaks_cols, recovered_cols = \
                _detect_global_lines(sobel, min_r, ring_width, self.p)

            # Structural validation
            profile_1d = np.median(sobel, axis=0)
            all_cands, _ = scipy.signal.find_peaks(
                profile_1d,
                height=max(4, self.p.get('peak_height_global', 15) * 0.4),
                distance=self.p.get('peak_distance', 10),
                prominence=max(3, self.p.get('peak_prominence', 8) * 0.4),
            )
            global_radii = _validate_peak_structure(
                global_radii, ring_width, self.p,
                profile=profile_1d, min_r=min_r, all_peak_cols=all_cands,
            )
            global_peaks_cols = np.array([r - min_r for r in global_radii], dtype=int)
            recovered_cols = set()

            per_zone = _detect_per_zone_lines(
                sobel, min_r, global_peaks_cols, self.p, recovered_cols=recovered_cols
            )

            labeled   = _label_lines(global_radii)
            num_zones = self.p['num_zones']
            zone_size = 360.0 / num_zones

            lines = []
            for line_idx, (lbl, g_r) in enumerate(labeled):
                zone_angles = [(z * zone_size, (z + 1) * zone_size) for z in range(num_zones)]
                zone_radii  = per_zone[line_idx] if line_idx < len(per_zone) else [None] * num_zones
                lines.append(RingLine(
                    label=lbl,
                    global_radius=g_r,
                    per_zone_radii=zone_radii,
                    zone_angles=zone_angles,
                ))

            defects    = _check_defects(lines, ring_width, cx, cy, self.p)
            is_defective = bool(defects)

            return {
                "status":     "FAIL" if is_defective else "PASS",
                "defects":    defects,
                "center":     (cx, cy),
                "ring_width": ring_width,
                "lines":      lines,
                "min_r":      min_r,
                "num_zones":  num_zones,
            }

        except Exception as e:
            return self._error(f"QA analysis failed: {e}")

    # ── helpers ─────────────────────────────────────────────────
    @staticmethod
    def _error(msg: str) -> dict:
        return {
            "status":     "ERROR",
            "defects":    [msg],
            "center":     None,
            "ring_width": 0,
            "lines":      [],
            "min_r":      0,
            "num_zones":  0,
        }
