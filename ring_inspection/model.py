"""
ring_inspection/model.py
========================
RingInspectionModel — ML-like plug-and-play ring QA model.

Public API
----------
    model = RingInspectionModel()               # load default weights
    model = RingInspectionModel("weights.json") # load custom weights

    model.train("golden_images/", allow_circle_3_fallback=False)
    model.save_model_weights("product_sap87.json")

    status, metrics, (polar_vis, cart_overlay) = model.predict(img_bgr)
    # status  : "PASS" | "REJECT"
    # metrics : dict with radii, gap variances, defect descriptions
    # vis     : annotated BGR images for display / saving
"""

import json
from pathlib import Path

import cv2
import numpy as np
import scipy.signal

from . import pipeline as pl
from .visualise import draw_cartesian_overlay, draw_polar_visualization

# Path to the bundled default weights
_DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "default_weights.json"


# ──────────────────────────────────────────────────────────────
# RingInspectionModel
# ──────────────────────────────────────────────────────────────


class RingInspectionModel:
    """
    ML-like ring inspection model.  Parameters are stored in a JSON weight
    file instead of neural-network weights.

    Parameters are grouped into sections matching the AGENTS.md schema:
        preprocessing            CLAHE + bilateral filter knobs
        anchor_detection         Canny + contour circularity filter
        polar_analysis           num_zones, peak thresholds, gap-scan recovery
        geometric_constraints    fixed-gap FP suppression fractions
        defect_thresholds        variance / discontinuity limits
        circle_3_inspection_parameters  fallback logic + learned ratio
    """

    def __init__(self, config_path: str = None):
        # Load defaults first, then optionally overlay custom weights
        with open(_DEFAULT_WEIGHTS_PATH, "r") as f:
            self.params: dict = json.load(f)

        if config_path:
            self.load_model_weights(config_path)

    # ── Serialisation ──────────────────────────────────────────

    def load_model_weights(self, path: str):
        """Merge weights from a JSON file into self.params."""
        with open(path, "r") as f:
            loaded = json.load(f)
        # Deep-merge: top-level sections
        for section, values in loaded.items():
            if isinstance(values, dict) and section in self.params:
                self.params[section].update(values)
            else:
                self.params[section] = values

    def save_model_weights(self, path: str):
        """Persist current params to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.params, f, indent=4)
        print(f"[RingInspectionModel] Weights saved → {path}")

    # ── Training ───────────────────────────────────────────────

    def train(self, golden_images_dir: str, allow_circle_3_fallback: bool = False):
        """
        Auto-tune parameters from a folder of golden (passing) product images.

        Learns:
          - nominal_gap_12_frac  (fixed gap 1-2 / ring_width)
          - nominal_gap_24_frac  (fixed gap 2-4 / ring_width)
          - nominal_ring_width_px (for anchor-pair selection in future images)
          - learned_ratio_3_to_24 (C3 position relative to gap 2-4)
          - max_variance_epsilon_frac (tight variance budget for gap-23, gap-34)
          - peak_height thresholds (from observed Sobel magnitudes)

        Sets allow_circle_3_fallback in circle_3_inspection_parameters.
        """
        self.params["circle_3_inspection_parameters"][
            "allow_circle_3_fallback"
        ] = allow_circle_3_fallback

        golden_dir = Path(golden_images_dir)
        image_files = (
            list(golden_dir.rglob("*.jpg"))
            + list(golden_dir.rglob("*.JPG"))
            + list(golden_dir.rglob("*.png"))
            + list(golden_dir.rglob("*.PNG"))
        )
        if not image_files:
            print(f"[train] No images found in {golden_images_dir}. Aborting.")
            return

        print(f"[train] Processing {len(image_files)} golden images …")

        stats = dict(
            gap_12_fracs=[],
            gap_24_fracs=[],
            ratio_3_to_24=[],
            ring_widths=[],
            var_gap_34=[],
            var_gap_23=[],
            peak_heights=[],
        )

        success = 0
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            result = self._run_detection(img)
            if result is None:
                continue

            circles, ring_width, per_zone, cx, cy, min_r = result
            if len(circles) < 4:
                continue

            r1, r2, r3, r4 = [
                circles[k] for k in ["circle_1", "circle_2", "circle_3", "circle_4"]
            ]

            if None in (r1, r2, r4):
                continue

            # Fixed-gap fractions
            stats["gap_12_fracs"].append((r2 - r1) / ring_width)
            stats["gap_24_fracs"].append((r4 - r2) / ring_width)
            stats["ring_widths"].append(ring_width)

            # Circle-3 position ratio in the 2→4 span
            if r4 > r2 and r3 is not None:
                stats["ratio_3_to_24"].append((r3 - r2) / (r4 - r2))

            # Per-zone gap variances
            if "circle_3" in circles and circles["circle_3"] is not None:
                c3_idx = list(circles.keys()).index("circle_3")
                c2_idx = list(circles.keys()).index("circle_2")
                c4_idx = list(circles.keys()).index("circle_4")
                if (
                    c3_idx < len(per_zone)
                    and c2_idx < len(per_zone)
                    and c4_idx < len(per_zone)
                ):
                    var_34 = pl.compute_zone_gap_variance(
                        per_zone[c3_idx], per_zone[c4_idx], ring_width
                    )
                    var_23 = pl.compute_zone_gap_variance(
                        per_zone[c2_idx], per_zone[c3_idx], ring_width
                    )
                    stats["var_gap_34"].append(var_34)
                    stats["var_gap_23"].append(var_23)

            success += 1

        print(f"[train] Successfully analysed {success}/{len(image_files)} images.")

        if success == 0:
            print("[train] WARNING: No usable samples — keeping default weights.")
            return

        def _mean(lst):
            return float(np.mean(lst)) if lst else None

        gc = self.params["geometric_constraints"]
        dt = self.params["defect_thresholds"]
        c3p = self.params["circle_3_inspection_parameters"]

        # Learned geometric constants
        if stats["gap_12_fracs"]:
            gc["nominal_gap_12_frac"] = _mean(stats["gap_12_fracs"])
            # Update the FP suppression ceiling (spec §2 — Gap 1-2 is fixed)
            gc["gap_12_frac_max"] = _mean(stats["gap_12_fracs"]) * 1.15

        if stats["gap_24_fracs"]:
            gc["nominal_gap_24_frac"] = _mean(stats["gap_24_fracs"])

        if stats["ring_widths"]:
            gc["nominal_ring_width_px"] = _mean(stats["ring_widths"])

        # Circle-3 learned ratio
        if stats["ratio_3_to_24"]:
            c3p["learned_ratio_3_to_24"] = _mean(stats["ratio_3_to_24"])

        # Variance epsilon: maximum observed + 20 % margin (spec §5.1)
        all_vars = stats["var_gap_34"] + stats["var_gap_23"]
        if all_vars:
            dt["max_variance_epsilon_frac"] = float(max(all_vars)) * 1.20

        self.params["trained"] = True
        print(
            f"[train] Weights updated. "
            f"gap_12_frac={gc.get('nominal_gap_12_frac', '?'):.3f}  "
            f"gap_24_frac={gc.get('nominal_gap_24_frac', '?'):.3f}  "
            f"ratio_3_to_24={c3p['learned_ratio_3_to_24']:.3f}  "
            f"epsilon_frac={dt['max_variance_epsilon_frac']:.4f}"
        )

    # ── Inference ──────────────────────────────────────────────

    def predict(self, image: np.ndarray) -> tuple:
        """
        Run full ring inspection on a BGR image.

        Returns
        -------
        status  : "PASS" | "REJECT"
        metrics : dict — radii, gap fractions, variance values, defect list
        (polar_vis, cart_overlay) : annotated BGR images
        """
        if image is None or image.size == 0:
            return "REJECT", {"defects": ["Empty input image"]}, (None, None)

        # ── Core detection ──────────────────────────────────────
        try:
            result = self._run_detection(image)
        except Exception as e:
            return "REJECT", {"defects": [f"Detection error: {e}"]}, (None, None)

        if result is None:
            return "REJECT", {"defects": ["No circular contours found"]}, (None, None)

        circles, ring_width, per_zone, cx, cy, min_r = result
        labels = list(circles.keys())  # circle_1 … circle_4
        n_zones = self.params["polar_analysis"]["num_zones"]

        # ── Circle-3 fallback / synthesis ───────────────────────
        synthesized_mask = [False] * n_zones  # tracks which C3 zones are synthetic
        c3p = self.params["circle_3_inspection_parameters"]

        if "circle_3" in circles and circles["circle_3"] is None:
            pass  # C3 is intentionally ignored for now

        # ── Defect checks ────────────────────────────────────────
        defects = self._check_defects(circles, per_zone, ring_width, cx, cy)

        # ── Build metrics dict ────────────────────────────────────
        r1 = circles.get("circle_1")
        r2 = circles.get("circle_2")
        r3 = circles.get("circle_3")
        r4 = circles.get("circle_4")

        c3_idx = labels.index("circle_3") if "circle_3" in labels else None
        c2_idx = labels.index("circle_2") if "circle_2" in labels else None
        c4_idx = labels.index("circle_4") if "circle_4" in labels else None

        def safe_pz(idx):
            if idx is None or idx >= len(per_zone):
                return [None] * n_zones
            return per_zone[idx]

        var_gap_34 = (
            pl.compute_zone_gap_variance(safe_pz(c3_idx), safe_pz(c4_idx), ring_width)
            if c3_idx is not None and c4_idx is not None
            else None
        )
        var_gap_23 = (
            pl.compute_zone_gap_variance(safe_pz(c2_idx), safe_pz(c3_idx), ring_width)
            if c2_idx is not None and c3_idx is not None
            else None
        )

        metrics = {
            "radii": {k: v for k, v in circles.items()},
            "ring_width": ring_width,
            "center": (cx, cy),
            "gap_12_px": (r2 - r1) if (r2 and r1) else None,
            "gap_23_px": (r3 - r2) if (r3 and r2) else None,
            "gap_34_px": (r4 - r3) if (r4 and r3) else None,
            "gap_24_px": (r4 - r2) if (r4 and r2) else None,
            "gap_12_frac": (r2 - r1) / ring_width if (r2 and r1) else None,
            "gap_24_frac": (r4 - r2) / ring_width if (r4 and r2) else None,
            "ratio_3_to_24": (
                (r3 - r2) / (r4 - r2) if (r3 and r2 and r4 and r4 > r2) else None
            ),
            "var_gap_34": var_gap_34,
            "var_gap_23": var_gap_23,
            "epsilon_threshold": self.params["defect_thresholds"][
                "max_variance_epsilon_frac"
            ],
            "circle_3_synthesized": any(synthesized_mask),
            "defects": defects,
        }

        status = "PASS" if not defects else "REJECT"

        # ── Visualizations ────────────────────────────────────────
        try:
            # Rebuild sobel for polar vis
            blurred = pl.preprocess(image, self.params)
            _, sobel = pl.build_polar_sobel(
                blurred,
                cx,
                cy,
                min_r,
                min_r
                + ring_width
                + self.params["anchor_detection"]["radial_pad_outer"],
            )
            polar_vis = draw_polar_visualization(
                sobel, per_zone, labels, min_r, n_zones
            )
        except Exception:
            polar_vis = None

        cart_overlay = draw_cartesian_overlay(
            image,
            cx,
            cy,
            per_zone,
            labels,
            status,
            n_zones,
            synthesized_mask=synthesized_mask,
        )

        return status, metrics, (polar_vis, cart_overlay)

    # ── Internal helpers ───────────────────────────────────────

    def _run_detection(self, img: np.ndarray):
        """
        Execute steps 1-5 of the pipeline and return:
            (circles_dict, ring_width, per_zone, cx, cy, min_r)
        Returns None on hard failure.
        """
        p = self.params
        pa = p["polar_analysis"]

        # Step 1 — Preprocess
        blurred = pl.preprocess(img, p)

        # Step 2 — Anchor locking (Center from highest circularity contour)
        candidates = pl.detect_candidates(blurred, p)
        if len(candidates) < 1:
            return None

        try:
            cx, cy, r1_anchor, r4_anchor, min_r, max_r = pl.lock_anchor_circles(
                candidates, p, img_shape=img.shape
            )
        except RuntimeError:
            return None

        ring_width = max_r - min_r

        # Step 3 — Polar unwrap + Sobel X
        polar_strip, sobel = pl.build_polar_sobel(blurred, cx, cy, min_r, max_r)

        # Crop out empty inner space based on first peak
        crop_start = pl.find_crop_start(sobel, p)
        if crop_start > 0:
            sobel = sobel[:, crop_start:]
            polar_strip = polar_strip[:, crop_start:]
            min_r += crop_start
            ring_width = max_r - min_r

        # Step 4A — Global peak detection
        global_radii, peak_cols, recovered_cols = pl.detect_global_peaks(
            sobel, min_r, ring_width, p
        )

        if len(global_radii) < 2:
            return None

        # Structural validation (FP suppression) bypassed

        # Step 4B — Per-zone peak detection + interpolation
        per_zone = pl.detect_per_zone_peaks(
            sobel, min_r, peak_cols, p, recovered_cols=recovered_cols
        )

        # Label circles by radial position
        sorted_radii = sorted(global_radii)
        circles = {}
        if len(sorted_radii) == 3:
            circles["circle_1"] = sorted_radii[0]
            circles["circle_2"] = sorted_radii[1]
            circles["circle_4"] = sorted_radii[2]
        else:
            labels = pl.CIRCLE_LABELS[: len(sorted_radii)]
            circles = {lbl: r for lbl, r in zip(labels, sorted_radii)}

        # Pad missing circles with None (fewer than 4 detected)
        for lbl in pl.CIRCLE_LABELS:
            if lbl not in circles:
                circles[lbl] = None

        return circles, ring_width, per_zone, cx, cy, min_r

    def _handle_c3_fallback(
        self, circles: dict, per_zone: list, ring_width: int, cx: int, cy: int
    ):
        """
        Handle Circle-3 dropout according to spec §8.

        Strict mode  (allow_circle_3_fallback=False):
            Count consecutive dropout zones.
            If > max_consecutive_dropouts_allowed → leave as None
            (defect check will flag it).

        Relaxed mode (allow_circle_3_fallback=True):
            Synthesise a virtual Circle 3 using the learned ratio.
            Mark all synthesized zones in synthesized_mask.
        """
        c3p = self.params["circle_3_inspection_parameters"]
        n_zones = self.params["polar_analysis"]["num_zones"]
        synthesized_mask = [False] * n_zones

        labels = list(circles.keys())
        c3_idx = labels.index("circle_3") if "circle_3" in labels else None

        if c3_idx is None:
            return circles, per_zone, synthesized_mask

        # Determine how many C3 per-zone slots are missing
        if c3_idx < len(per_zone):
            c3_zones = per_zone[c3_idx]
        else:
            c3_zones = [None] * n_zones

        missing_count = sum(1 for r in c3_zones if r is None)

        if not c3p["allow_circle_3_fallback"]:
            # Strict mode — just return as-is; check_defects will flag it
            return circles, per_zone, synthesized_mask

        # Relaxed mode — synthesise
        if missing_count == 0:
            return circles, per_zone, synthesized_mask  # C3 fine, nothing to do

        r2 = circles.get("circle_2")
        r4 = circles.get("circle_4")
        if r2 is None or r4 is None:
            return circles, per_zone, synthesized_mask

        ratio = c3p["learned_ratio_3_to_24"]
        r3_synth = int(round(r2 + (r4 - r2) * ratio))

        # Find per-zone C2 and C4 radii for zone-level synthesis
        c2_idx = labels.index("circle_2") if "circle_2" in labels else None
        c4_idx = labels.index("circle_4") if "circle_4" in labels else None

        new_c3_zones = list(c3_zones)
        for z in range(n_zones):
            if c3_zones[z] is None:
                # Try zone-level synthesis if C2/C4 per-zone data available
                r2_z = per_zone[c2_idx][z] if c2_idx is not None else r2
                r4_z = per_zone[c4_idx][z] if c4_idx is not None else r4
                if r2_z is not None and r4_z is not None:
                    new_c3_zones[z] = int(round(r2_z + (r4_z - r2_z) * ratio))
                else:
                    new_c3_zones[z] = r3_synth
                synthesized_mask[z] = True

        per_zone[c3_idx] = new_c3_zones
        circles["circle_3"] = r3_synth

        print(
            f"[predict] Circle-3 fallback fired: "
            f"{missing_count}/{n_zones} zones synthesised "
            f"(ratio={ratio:.3f}, R3_synth={r3_synth}px)"
        )

        return circles, per_zone, synthesized_mask

    def _check_defects(
        self, circles: dict, per_zone: list, ring_width: int, cx: int, cy: int
    ) -> list:
        """
        All defect checks are removed as per user request to only detect circles.
        Returns an empty list (always PASS).
        """
        return []
