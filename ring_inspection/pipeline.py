"""
ring_inspection/pipeline.py
===========================
Pure processing functions implementing the AGENTS.md pipeline:

  preprocess          → grayscale + CLAHE + bilateral
  detect_candidates   → Canny + dilate + contour circularity filter
  lock_anchor_circles → select Circle-1 (inner) and Circle-4 (outer) pair
  build_polar_sobel   → warpPolar (region-bounded) + Sobel X
  detect_global_peaks → median profile peak detection with gap-scan recovery
  detect_per_zone_peaks → per-zone peak assignment with CoM fallback + interpolation
  validate_peaks      → structural geometry validation (FP suppression)
  count_consecutive_dropouts → for Circle-3 strict-mode check
"""

from itertools import combinations

import cv2
import numpy as np
import scipy.signal

# ─────────────────────────────────────────────────────────────
# STEP 1 — Preprocessing
# ─────────────────────────────────────────────────────────────


def preprocess(img_bgr: np.ndarray, p: dict) -> np.ndarray:
    """BGR → Grayscale → CLAHE → Bilateral blur."""
    pp = p["preprocessing"]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=pp["clahe_clip"], tileGridSize=(pp["clahe_tile"], pp["clahe_tile"])
    )
    enhanced = clahe.apply(gray)
    blurred = cv2.bilateralFilter(
        enhanced,
        d=pp["bilateral_d"],
        sigmaColor=pp["bilateral_sigma_color"],
        sigmaSpace=pp["bilateral_sigma_space"],
    )
    return blurred


# ─────────────────────────────────────────────────────────────
# STEP 2 — Primary Anchor Locking (Circle 1 & Circle 4)
# ─────────────────────────────────────────────────────────────


def detect_candidates(blurred: np.ndarray, p: dict) -> list:
    """
    Canny → dilate → contour filter → list of (cx, cy, r) candidates.
    Only keeps contours that pass the minimum perimeter + circularity gate.
    """
    ad = p["anchor_detection"]
    edges = cv2.Canny(blurred, ad["canny_lo"], ad["canny_hi"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=ad["dilate_iters"])
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        peri = cv2.arcLength(c, closed=True)
        if peri < ad["min_perimeter"]:
            continue
        area = cv2.contourArea(c)
        if peri == 0:
            continue
        circularity = (4 * np.pi * area) / (peri**2)
        (cx, cy), r = cv2.minEnclosingCircle(c)
        candidates.append((float(cx), float(cy), float(r), float(circularity)))

    return candidates


def lock_anchor_circles(candidates: list, p: dict, img_shape: tuple = None):
    """
    Since images are now cropped to the product, we take the center of the circle
    that has the highest circularity to unwrap directly from the center.
    min_r is 0, and max_r is the distance to the image edge.
    """
    if not candidates:
        raise RuntimeError("No circular contours found — cannot lock center.")

    # Take the 3 largest detected and center of best circularity among them
    candidates.sort(key=lambda x: x[2], reverse=True)
    top_3_largest = candidates[:3]
    top_3_largest.sort(key=lambda x: x[3], reverse=True)
    best_cand = top_3_largest[0]
    cx, cy = best_cand[0], best_cand[1]

    # Unwrap from center, we will crop dynamically based on peaks later
    min_r = 0
    if img_shape:
        h, w = img_shape[:2]
        max_r = int(min(cx, cy, w - cx, h - cy))
        if max_r <= 0:
            max_r = int(min(w, h) / 2)
    else:
        max_r = int(min(cx, cy)) if min(cx, cy) > 0 else 500

    # Return dummy values for r1 and r4 to maintain signature compatibility
    return int(cx), int(cy), 0, 0, min_r, max_r


# ─────────────────────────────────────────────────────────────
# STEP 3 — Region-Bounded Vertical Polar Unwrap + Sobel X
# ─────────────────────────────────────────────────────────────


def build_polar_sobel(blurred: np.ndarray, cx: int, cy: int, min_r: int, max_r: int):
    """
    Warp to polar (360 × max_r), crop to active ring band [min_r:], apply Sobel X.
    Returns (polar_strip, sobel_strip) both shaped (360, ring_width).
    """
    polar_full = cv2.warpPolar(
        src=blurred,
        dsize=(max_r, 360),
        center=(cx, cy),
        maxRadius=max_r,
        flags=cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR,
    )
    polar = polar_full[:, min_r:]  # shape (360, ring_width)
    sobel = cv2.Sobel(polar, cv2.CV_64F, 1, 0, ksize=3)
    sobel = cv2.convertScaleAbs(sobel)
    return polar, sobel


def col_to_radius(col: int, min_r: int) -> int:
    return int(col + min_r)


# ─────────────────────────────────────────────────────────────
# STEP 4A — Global Peak Detection (N-strongest in median profile)
# ─────────────────────────────────────────────────────────────


def find_crop_start(sobel: np.ndarray, p: dict) -> int:
    pa = p["polar_analysis"]
    profile = np.median(sobel, axis=0)
    all_peaks, _ = scipy.signal.find_peaks(
        profile,
        height=max(6, pa["peak_height_global"] * 0.5),
        distance=max(5, pa["peak_distance"] // 2),
        prominence=max(4, pa["peak_prominence"] * 0.5),
    )
    if len(all_peaks) == 0:
        return 0
    return max(0, all_peaks[0] - 40)


def detect_global_peaks(
    sobel: np.ndarray, min_r: int, ring_width: int, p: dict
) -> tuple:
    """
    Dive the polar unwrap into two half vertically: left half (60%) and right half (40%).
    The two highest peaks in left half are circle 1 and 2.
    The largest peak in the right half is the circle 4.
    Circle 3 is ignored.
    """
    pa = p["polar_analysis"]
    profile = np.median(sobel, axis=0)
    n_cols = sobel.shape[1]

    all_peaks, _ = scipy.signal.find_peaks(
        profile,
        height=max(6, pa["peak_height_global"] * 0.5),
        distance=max(5, pa["peak_distance"] // 2),
        prominence=max(4, pa["peak_prominence"] * 0.5),
    )

    if len(all_peaks) == 0:
        return [], np.array([], dtype=int), set()

    all_peaks = np.sort(all_peaks)

    # Since min_r is updated before this function, we can simply split at 60%
    split_idx = int(0.45 * n_cols)
    left_half_peaks = all_peaks[all_peaks < split_idx]
    right_half_peaks = all_peaks[all_peaks >= split_idx]

    c1_col, c2_col = None, None
    if len(left_half_peaks) > 0:
        left_heights = profile[left_half_peaks]
        # The two highest magnitude peaks in the left half are Circle 1 and 2
        top2_left_idx = np.argsort(left_heights)[-2:]
        left_cands = np.sort(left_half_peaks[top2_left_idx])
        if len(left_cands) == 2:
            c1_col, c2_col = left_cands[0], left_cands[1]
        else:
            c1_col = left_cands[0]

    c4_col = None
    if len(right_half_peaks) > 0:
        # The largest peak in the right half is the circle 4
        right_heights = profile[right_half_peaks]
        c4_col = right_half_peaks[np.argmax(right_heights)]

    # "remove all the logic for finding cirlce 3 just find circle 1, 2 and 4"
    selected = [c for c in (c1_col, c2_col, c4_col) if c is not None]
    selected = np.array(selected, dtype=int)

    radii = [col_to_radius(int(pk), min_r) for pk in selected]
    return radii, selected, set()


# ─────────────────────────────────────────────────────────────
# STEP 4B — Per-Zone Peak Detection + Coordinate Remapping
# ─────────────────────────────────────────────────────────────


def detect_per_zone_peaks(
    sobel: np.ndarray, min_r: int, global_peak_cols, p: dict, recovered_cols=None
) -> list:
    """
    Assign each global anchor to its nearest zone-peak per angular zone.
    Fallback to CoM if no explicit peak; interpolate single-zone gaps.

    Returns per_zone[n_lines][num_zones] (absolute radii, None if missing).
    """
    if recovered_cols is None:
        recovered_cols = set()

    pa = p["polar_analysis"]
    num_zones = pa["num_zones"]
    zone_rows = sobel.shape[0] // num_zones
    n_lines = len(global_peak_cols)
    sorted_cols = sorted(global_peak_cols)

    min_gap = int(np.min(np.diff(sorted_cols))) if len(sorted_cols) >= 2 else 30
    default_hg = max(6, min_gap // 2)

    # Build per-anchor search windows
    per_anchor_win = []
    for i, col in enumerate(sorted_cols):
        left_room = (col - sorted_cols[i - 1] - 1) // 2 if i > 0 else 9999
        right_room = (
            (sorted_cols[i + 1] - col - 1) // 2 if i < len(sorted_cols) - 1 else 9999
        )
        if col in recovered_cols:
            wide = max(default_hg * 2, 20)
            lo_off = min(default_hg, left_room)
            hi_off = min(wide, right_room)
        else:
            sym = min(default_hg, left_room, right_room)
            lo_off = sym
            hi_off = sym
        per_anchor_win.append((lo_off, hi_off))

    col_to_aidx = {col: i for i, col in enumerate(sorted_cols)}
    anchor_win = [per_anchor_win[col_to_aidx[c]] for c in global_peak_cols]
    per_zone = [[None] * num_zones for _ in range(n_lines)]

    for z in range(num_zones):
        row_s = z * zone_rows
        row_e = min((z + 1) * zone_rows, sobel.shape[0])
        zp = np.mean(sobel[row_s:row_e, :], axis=0)

        zone_peaks, _ = scipy.signal.find_peaks(
            zp,
            height=pa["peak_height_zone"],
            distance=3,
            prominence=4,
        )
        anchors = np.array(global_peak_cols, dtype=float)
        assigned_zp = set()

        # Greedy nearest-anchor assignment
        if len(zone_peaks) > 0:
            zpf = np.array(zone_peaks, dtype=float)
            signed = zpf[None, :] - anchors[:, None]
            order = np.argsort(np.abs(signed).ravel())
            for flat_idx in order:
                a_idx = flat_idx // len(zpf)
                zp_idx = flat_idx % len(zpf)
                lo_off, hi_off = anchor_win[a_idx]
                sd = signed[a_idx, zp_idx]
                if sd < -lo_off or sd > hi_off:
                    continue
                if per_zone[a_idx][z] is not None or zp_idx in assigned_zp:
                    continue
                per_zone[a_idx][z] = col_to_radius(int(zone_peaks[zp_idx]), min_r)
                assigned_zp.add(zp_idx)

        # CoM fallback for still-unassigned anchors
        for a_idx, g_col in enumerate(global_peak_cols):
            if per_zone[a_idx][z] is not None:
                continue
            lo_off, hi_off = anchor_win[a_idx]
            col_lo = max(0, g_col - lo_off)
            col_hi = min(zp.shape[0], g_col + hi_off + 1)
            region = zp[col_lo:col_hi].copy()

            # Mask out other anchors' territory
            for other_idx, other_r in enumerate(per_zone):
                if other_idx == a_idx or other_r[z] is None:
                    continue
                other_col = other_r[z] - min_r
                o_lo, o_hi = anchor_win[other_idx]
                ml = max(col_lo, other_col - o_lo) - col_lo
                mh = min(col_hi, other_col + o_hi + 1) - col_lo
                if mh > ml >= 0:
                    region[ml:mh] = 0

            if region.max() < pa["peak_height_zone"]:
                continue
            weights = region.astype(float)
            cols_range = np.arange(col_lo, col_hi)
            com_col = int(round(np.sum(cols_range * weights) / np.sum(weights)))
            per_zone[a_idx][z] = col_to_radius(com_col, min_r)

    # Linear interpolation for single-missing zones (spec §4 eq.)
    for line_idx in range(n_lines):
        radii = per_zone[line_idx]
        new_radii = list(radii)
        for z in range(num_zones):
            if radii[z] is None:
                left = radii[(z - 1) % num_zones]
                right = radii[(z + 1) % num_zones]
                if left is not None and right is not None:
                    new_radii[z] = int(round((left + right) / 2.0))
        per_zone[line_idx] = new_radii

    return per_zone


# ─────────────────────────────────────────────────────────────
# STEP 5 — Structural Geometry Validation (FP Suppression)
# ─────────────────────────────────────────────────────────────


def validate_peaks(
    global_radii: list,
    ring_width: int,
    p: dict,
    profile=None,
    min_r: int = 0,
    all_peak_cols=None,
) -> list:
    """
    Enforce structural geometry constraints from §2 to suppress false positives.

    Constraints (fractions of ring_width):
      gap_12 <= gap_12_frac_max       — inner lip must be narrow
      gap_23 >= gap_23_frac_min       — rubber seat gap must be significant
      gap_34 >= gap_34_frac_min       — metal band gap (can be thin)
      gap_24 >= gap_24_frac_min       — total span (inner-top → outer) must be meaningful

    If the supplied set fails, tries all 4-combinations of candidate peaks.
    Returns the best valid radii list (len=4) or original as fallback.
    """
    gc = p["geometric_constraints"]
    max_lip = gc["gap_12_frac_max"] * ring_width
    min_rubber = gc["gap_23_frac_min"] * ring_width
    min_metal = gc["gap_34_frac_min"] * ring_width
    min_span = gc["gap_24_frac_min"] * ring_width

    def is_valid(r):
        if len(r) < 4:
            return False
        s = sorted(r)
        return (
            s[1] - s[0] <= max_lip
            and s[2] - s[1] >= min_rubber
            and s[3] - s[2] >= min_metal
            and s[3] - s[1] >= min_span
        )

    if is_valid(global_radii):
        return sorted(global_radii)

    if profile is None or all_peak_cols is None or len(all_peak_cols) < 4:
        return sorted(global_radii)

    candidates = sorted([col_to_radius(int(c), min_r) for c in all_peak_cols])
    best_combo, best_score = None, -1.0
    for combo in combinations(candidates, 4):
        if not is_valid(list(combo)):
            continue
        score = sum(
            float(profile[c - min_r]) if 0 <= (c - min_r) < len(profile) else 0.0
            for c in combo
        )
        if score > best_score:
            best_score = score
            best_combo = list(combo)

    return best_combo if best_combo is not None else sorted(global_radii)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

CIRCLE_LABELS = [
    "circle_1",  # Bottom Inner Surface
    "circle_2",  # Top Inner Surface
    "circle_3",  # Band Separation
    "circle_4",  # Top Outer Surface
]


def count_consecutive_dropouts(per_zone_radii: list) -> int:
    """Return the maximum run of consecutive None values in a per-zone list."""
    max_run = run = 0
    for r in per_zone_radii:
        if r is None:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def compute_zone_gap_variance(
    per_zone_a: list, per_zone_b: list, ring_width: int
) -> float:
    """
    Variance of (b_zone_i - a_zone_i) / ring_width for all zones where
    both a and b are detected.  Returns 0 if fewer than 4 paired zones.
    """
    gaps = [
        (b - a) / ring_width
        for a, b in zip(per_zone_a, per_zone_b)
        if a is not None and b is not None
    ]
    return float(np.var(gaps)) if len(gaps) >= 4 else 0.0


# ─────────────────────────────────────────────────────────────
# §12 — Circle-3 Spatial Proximity Constraints
# ─────────────────────────────────────────────────────────────


def check_c3_proximity(
    per_zone_c2: list, per_zone_c3: list, per_zone_c4: list, ring_width: int, p: dict
) -> tuple:
    """
    §12: Validate that Circle 3 is not falsely clustered.

    Two independent tests:
      12.1  C2-buffer proximity: if > c3_c2_proximity_tolerance_pct of detected
            C3 zones lie within c3_c2_proximity_buffer_frac of C2, the track is
            an inner-shoulder reflection → False Positive.
      12.2  C4-clustering: if ALL detected C3 zones sit within
            c3_c4_clustering_threshold_frac of C4, the track is an outer-edge
            reflection artifact → False Positive.

    Returns (c2_clustered: bool, c4_clustered: bool).
    """
    c3p = p["circle_3_inspection_parameters"]
    n_zones = len(per_zone_c3)

    buf_frac = c3p.get("c3_c2_proximity_buffer_frac", 0.08)
    tol_pct = c3p.get("c3_c2_proximity_tolerance_pct", 0.60)
    c4_clust_frac = c3p.get("c3_c4_clustering_threshold_frac", 0.05)

    buf_px = buf_frac * ring_width
    c4_buf_px = c4_clust_frac * ring_width

    zones_in_c2_buf = 0
    zones_near_c4 = 0
    total_detected = 0

    for z in range(n_zones):
        r3 = per_zone_c3[z]
        r2 = per_zone_c2[z]
        r4 = per_zone_c4[z]

        if r3 is None:
            continue
        total_detected += 1

        if r2 is not None and (r3 - r2) <= buf_px:
            zones_in_c2_buf += 1

        if r4 is not None and (r4 - r3) <= c4_buf_px:
            zones_near_c4 += 1

    if total_detected == 0:
        return False, False

    c2_clustered = (zones_in_c2_buf / total_detected) > tol_pct
    c4_clustered = zones_near_c4 == total_detected  # 100% near C4 → artifact

    return c2_clustered, c4_clustered


# ─────────────────────────────────────────────────────────────
# §13 — Sinusoidal Defect Signature Analysis
# ─────────────────────────────────────────────────────────────


def detect_sinusoidal_defect(
    per_zone_radii: list, num_zones: int, ring_width: int, p: dict
) -> tuple:
    """
    §13: Detect smooth sinusoidal wave pattern in per-zone C3 radii.

    A structurally defective ring (elliptical / off-centre deformation)
    shows its Circle-3 boundary tracing a smooth sinusoidal curve across
    the polar angular zones:

        R(θ) = D + A·cos(θ) + B·sin(θ)       (fundamental harmonic fit)

    Amplitude = √(A² + B²) captures peak-to-trough excursion.
    R² measures how well the data fits a clean sine (separates smooth
    low-freq defect from erratic high-freq noise).

    Returns (is_sinusoidal, amplitude_frac, r_squared).
      is_sinusoidal : True when amplitude > threshold AND R² > min_r_squared
      amplitude_frac: amplitude / ring_width
      r_squared     : goodness-of-fit of the fundamental sine
    """
    c3p = p["circle_3_inspection_parameters"]
    dt = p["defect_thresholds"]

    if not c3p.get("sinusoidal_detection_enabled", True):
        return False, 0.0, 0.0

    amp_thresh = c3p.get(
        "sinusoidal_amplitude_threshold_frac", dt.get("max_variance_epsilon_frac", 0.04)
    )
    min_r2 = c3p.get("sinusoidal_min_r_squared", 0.55)

    # Zone mid-angles in radians (one full revolution)
    zone_angles_rad = np.array(
        [np.deg2rad((z + 0.5) * 360.0 / num_zones) for z in range(num_zones)]
    )

    valid_angles, valid_radii = [], []
    for z in range(num_zones):
        if per_zone_radii[z] is not None:
            valid_angles.append(zone_angles_rad[z])
            valid_radii.append(float(per_zone_radii[z]))

    if len(valid_radii) < 6:
        return False, 0.0, 0.0

    va = np.array(valid_angles)
    vr = np.array(valid_radii)

    # Fit:  R = D + A_c·cos(θ) + A_s·sin(θ)
    A_mat = np.column_stack([np.ones_like(va), np.cos(va), np.sin(va)])
    try:
        res, _, _, _ = np.linalg.lstsq(A_mat, vr, rcond=None)
    except np.linalg.LinAlgError:
        return False, 0.0, 0.0

    D, A_c, A_s = float(res[0]), float(res[1]), float(res[2])
    amplitude = float(np.sqrt(A_c**2 + A_s**2))
    amplitude_frac = amplitude / ring_width

    # R² — goodness of fit (how smooth / sinusoidal the wave is)
    predicted = D + A_c * np.cos(va) + A_s * np.sin(va)
    ss_res = float(np.sum((vr - predicted) ** 2))
    ss_tot = float(np.sum((vr - np.mean(vr)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else 0.0

    # §13.2: smooth low-frequency sine with amplitude beyond epsilon
    is_sinusoidal = (r_squared >= min_r2) and (amplitude_frac > amp_thresh)

    return is_sinusoidal, amplitude_frac, r_squared
