Here is a comprehensive, production-grade Markdown technical specification document tailored exactly for your coding agent. It perfectly frames the problem, details the architecture, outlines the optimization logic for Circle 3, and defines the JSON parameter schema.

---

# Technical Specification: Project Antigravity ML-Like Parameter Auto-Tuning Framework

## 1. Overview & Objectives

Project Antigravity is a high-precision computer vision metrology pipeline designed to identify, segment, and inspect 4 concentric features on industrial ring components. Due to product variations and hardware illumination issues (e.g., edge dropouts), a rigid geometric algorithm fails across different product families.

The goal of this module is to wrap our core image processing pipeline into an **ML-like class interface (`RingInspectionModel`)**. Instead of standard neural weights, this model utilizes an optimized set of **geometric and structural parameters** stored in a JSON file.

The model must support:

* **`.train(image_folder)`:** Auto-tunes thresholds and learns nominal geometric constraints from a directory of golden (passing) samples of a specific product type.
* **`.predict(image)`:** Runs high-speed inference to locate the 4 primary boundaries, dynamically interpolates missing data, and flags defective anomalies.

---

## 2. Product Geometry Baseline

Each physical component features exactly 4 target boundaries, as illustrated below:

1. **Circle 1:** Bottom Inner Surface
2. **Circle 2:** Top Inner Surface
3. **Circle 3:** Band Separation
4. **Circle 4:** Top Outer Surface

### Geometric Constraints & False Positive Suppression

The relationships between the target circles vary predictably across different product categories:

| Target Gap | Type | Behavioral Logic & Operational Constraint |
| --- | --- | --- |
| **Gap 1-2** | **Fixed** | Constant value. **All detected features within this zone are explicitly suppressed as False Positives.** |
| **Gap 1-4** | **Fixed** | Maximum structural distance boundary constraint. |
| **Gap 2-4** | **Fixed** | Total structural band budget. **Any feature classified as a "Circle 4 candidate" found *inside* this zone is suppressed as a False Positive.** |
| **Gap 2-3** | **Variable** | Dynamic distance tracked to evaluate product defects. |
| **Gap 3-4** | **Variable** | Dynamic distance tracked to evaluate product defects. |

---

## 3. Comprehensive Pipeline Architecture

```
[Input Image] ➔ [Grayscale] ➔ [CLAHE] ➔ [Bilateral Blur]
       ➔ [Canny Edge] ➔ [Coarse Centers & Circle 1 & 4 Anchor Lock]
       ➔ [Region-Bounded Vertical Polar Unwrap] ➔ [Horizontal Sobel X]
       ➔ [N-Zone Section Slicing] ➔ [Peak Profiling & Line Fitting]
       ➔ [Dynamic Geometry Verification] ➔ [Pass / Fail Inference]

```
take inspiration from ../app/pipeline/circle_qa.py

### Step 3.1: Coarse Preprocessing

1. **Grayscale Transformation:** Convert native BGR input to single-channel space.
2. **CLAHE Enhancement:** Apply Contrast Limited Adaptive Histogram Equalization (`clipLimit`, `tileGridSize`) to normalize light falloff.
3. **Bilateral Filtering:** Apply edge-preserving smoothing (`d`, `sigmaColor`, `sigmaSpace`) to suppress surface metallic grain textures while maintaining boundary sharpness.

### Step 3.2: Primary Anchor Locking (Circle 1 & 4)

1. Execute Canny Edge Detection (`threshold1`, `threshold2`).
2. Dilate the resulting edge map to bridge microscopic pixel splits.
3. Find all valid contours, filtering out background noise based on a minimum perimeter threshold.
4. Filter for circularity:

$$\text{Circularity} = \frac{4\pi \times \text{Area}}{\text{Perimeter}^2} > \text{min\_circularity\_thresh}$$


5. Extract potential candidates using `cv2.minEnclosingCircle`.
6. Locate two dominant circles whose radial delta satisfies the product's **Fixed Gap 1-4** parameter constraint. The average of their coordinates locks the global tracking center `(exact_cx, exact_cy)`.

### Step 3.3: Region-Bounded Vertical Polar Unwrap

To prevent waste of processing overhead and crop out internal whitespace zones, unroll the active band using `cv2.warpPolar`.

* **Unwrap Domain:** Set `minRadius = Circle_1 - buffer` and `maxRadius = Circle_4 + buffer`.
* **Image Matrix Structure:** The resulting array has a height of 360 (representing the $360^\circ$ angular sweep as vertical rows) and a width matching the `active_radial_depth` ($maxRadius - minRadius$).
* **Gradient Extraction:** Compute the Horizontal Sobel Filter (`dx=1, dy=0`) to isolate concentric changes running parallel down the vertical rows.

---

## 4. Multi-Zone Section Slicing & Mathematical Line Fitting

To catch local defects and structural out-of-roundness, do not average the entire polar strip at once.

1. **Segmentation:** Divide the 360 vertical rows of the Sobel map into $N$ equal sections (typically 16 or 32 vertical segments, corresponding to $22.5^\circ$ or $11.25^\circ$ physical wedges).
2. **Peak Detection:** For each section independently, compress the rows along `axis=0` to create a 1D horizontal profile curve. Run peak detection (`height`, `distance`) to find the column index ($X$) of all features.
3. **Coordinate Remapping:** Transform the local horizontal columns back into absolute Cartesian radial space:

$$\text{Radius} = X \times \left(\frac{\text{max\_radius} - \text{min\_radius}}{\text{polar\_width}}\right) + \text{min\_radius}$$



### Missing Peak Interpolation

If a tracking zone fails to return a peak due to severe contrast dropout, but the adjacent zones before and after successfully record the feature, compute a linear interpolation to bridge the gap:


$$R_{\text{missing}} = R_{\text{zone}-1} + \left( \frac{R_{\text{zone}+1} - R_{\text{zone}-1}}{2} \right)$$


This maintains continuous boundary paths across noisy sections.

---

## 5. Circle 3 Inspection Strategy & Defect Optimization

Circle 3 (the Band Separation) represents the dynamic interface layer. Its defect signature is modeled as shown below:

### 5.1: Mathematical Modeling of Pass/Fail States

* **Normal Product (Pass):** The detected peaks for Circle 3 form a flat, horizontal row line in the polar strip. When translated to Cartesian space, it maintains a highly uniform radial distance to Circle 2 and Circle 4 across all $N$ sections:

$$\text{Variance}(R_{4} - R_{3}) \le \epsilon \quad \text{AND} \quad \text{Variance}(R_{3} - R_{2}) \le \epsilon$$



Where $\epsilon$ is a trained tolerance threshold parameter.
* **Defective Product (Fail):** The piece is automatically flagged as defective if it exhibits either of the following characteristics:
1. **Severe Structural Dropout:** The peak for Circle 3 is completely missing across multiple consecutive tracking zones and cannot be resolved by interpolation.
2. **Out-of-Roundness / Ellipticity:** The line snakes horizontally across the polar sectors, causing the tracking distances ($R_{4} - R_{3}$ and $R_{3} - R_{2}$) to oscillate significantly beyond the trained $\epsilon$ parameter value.



---

## 6. The ML-Like JSON Weight Parameter Schema

The behavior of the preprocessing, Canny tracking, and inspection rules must be entirely configured via a JSON parameter file acting as the model weights.

```

---

## 7. Execution Interface Design

Your coding agent must implement the code structure using the following class interface format:

```python
import json
import numpy as np
import cv2

class RingInspectionModel:
    def __init__(self, config_path=None):
        self.params = {}
        if config_path:
            self.load_model_weights(config_path)

    def load_model_weights(self, path):
        with open(path, 'r') as f:
            self.params = json.load(f)

    def save_model_weights(self, path):
        with open(path, 'w') as f:
            json.dump(self.params, f, indent=4)

    def train(self, golden_images_dir):
        """
        Loops through clean product samples to auto-tune parameters.
        Calculates nominal values for fixed gaps and establishes baseline
        variances for circle 3 peak thresholds. Updates self.params dynamically.
        """
        print(f"Training on clean samples inside: {golden_images_dir}")
        # IMPLEMENTATION: Tune peak heights, verify fixed geometric constants,
        # set max_allowed_variance_epsilon slightly above observed maximums.
        pass

    def predict(self, image):
        """
        Runs full runtime inference on an input image matrix using the active weights configuration.
        Returns:
            status (str): "PASS" or "REJECT"
            metrics (dict): Extracted true radii values, out-of-roundness tracking, anomaly descriptions.
            visualization_tuple (tuple): (polar_visualization_img, cartesian_arc_overlay_img)
        """
        # IMPLEMENTATION: Run pipeline step-by-step using structural rules
        # to isolate the 4 target boundaries.
        
        # Arc segment rendering tracking format:
        # cv2.ellipse(output, center, (r, r), 0, start_angle, end_angle, color, 1)
        
        status = "PASS"
        metrics = {}
        return status, metrics, (None, None)

```

## 8. Circle 3 Visibility Fallback & Dynamic Parameter Tuning

### 8.1 Fallback Operational Logic
For specific product families, the physical "Circle 3 (Band Separation)" boundary exhibits extremely low contrast or reflection dropout, making it structurally impossible to guarantee clean peak tracking via the Sobel Y/X profiling layer. 

To prevent false rejections on these specific parts, the model interface must accommodate an explicit training flag: `allow_circle_3_fallback`.

* **When `allow_circle_3_fallback == False` (Strict Mode):**
  The model enforces mandatory four-circle detection. If Circle 3 is missing across consecutive sectors beyond `max_consecutive_dropouts_allowed`, the part is instantly marked as **REJECT**.
* **When `allow_circle_3_fallback == True` (Relaxed Mode):**
  If the peak-detection algorithm fails to securely locate a continuous boundary line for Circle 3 across the $N$ horizontal sectors, the model will deploy a geometric fallback. It will mathematically estimate a "Virtual Circle 3" based on nominal training data to maintain pipeline continuity and **PASS the product by default** (provided Circles 1, 2, and 4 pass strict concentricity specifications).

  [Inference Pipeline] ➔ Check allow_circle_3_fallback flag
├── False ➔ Enforce all 4 circles strictly ➔ Fail if Circle 3 drops out
└── True  ➔ Check Peak Count ➔ If < 4, synthetically construct Circle 3 ➔ PASS by default


### 8.2 Architectural Implementation Rules
1. **Virtual Synthesis:** If the fallback fires, the model calculates the local radius for Circle 3 using a historical scalar ratio derived during training:
   $$R_{3\_synthetic} = R_{2} + (R_{4} - R_{2}) \times \text{nominal\_ratio\_3\_to\_24}$$
2. **Visual Mapping Modification:** In the final output visualization matrix, synthesized segments must be rendered as a **dashed or distinct color arc** to explicitly flag to the supervisor that the layer boundary was mathematically generated rather than raw sensor-tracked.

---

## 9. Updated JSON Weight Parameter Schema
The JSON profile payload must be extended under `circle_3_inspection_parameters` to store this behavioral rule and its learned ratio constants:


10. API Interface Enhancements
The signature of the training method must be updated to accept the boolean configuration toggle:

Python
    def train(self, golden_images_dir, allow_circle_3_fallback=False):
        """
        Loops through clean product samples to auto-tune parameters.
        
        Args:
            golden_images_dir (str): Path to folder containing target clean images.
            allow_circle_3_fallback (bool): If true, configures the JSON weights 
                                            to synthetically bypass Circle 3 dropouts 
                                            and pass them by default.
        """
        self.params["circle_3_inspection_parameters"]["allow_circle_3_fallback"] = allow_circle_3_fallback
        
        # IMPLEMENTATION LOGIC:
        # If allow_circle_3_fallback is True, measure the mean spatial distance where 
        # Circle 3 occasionally appears during training relative to the total 
        # Circle 2-4 span. Save this value into self.params["circle_3_inspection_parameters"]["learned_ratio_3_to_24"]
        pass

## 11. Advanced Peak Filtering and Structural Contours Validation

### 11.1 Anchor Resolution via Dominant Peak Extraction (Circles 1 & 2)
To robustly handle optical artifacts or internal machining scratches that appear within the ring's core, the definition of Circle 1 and Circle 2 must rely on peak energy rather than raw spatial ordering:

1. **Peak Sorting:** When evaluating the radial profile segments, the algorithm must sort all discovered peaks by their magnitude (intensity value in the Sobel map) in descending order.
2. **Anchor Identification:** The two peaks containing the absolute highest energy (greatest magnitudes) are explicitly designated as the structural boundaries for **Circle 1** and **Circle 2**.
3. **Internal False Positive Suppression:** Any alternative peak detected spatially *between* Circle 1 and Circle 2 must be intentionally stripped from the pipeline and ignored.
4. **Training Behavior:** During execution of the `.train()` routine, the nominal parameters, base radii, and the `gap_1_2_nominal_px` constraint must be computed exclusively with respect to these two highest-magnitude master anchor peaks.

---

## 12. Circle 3 Spatial Distribution & Proximity Constraints

To eliminate false positive matches generated by surface glare or edge bleeding directly adjacent to the main shoulders, Circle 3 must satisfy strict distribution thresholds relative to the active channel boundaries.

[Circle 2 Boundary] ─── (Max 50-60% of Arcs Allowed in Buffer Zone) ───
│
[VALID MIDDLE ZONE] ◄── Circle 3 must cross into here
│
[Circle 4 Boundary] ─── (Cannot have 100% of Arcs clustered here) ───

A tracked Circle 3 candidate line will be classified as a **False Positive** and rejected if it fails either of the following proximity tests:

### 12.1 Circle 2 Buffer Proximity Constraint
* **The Rule:** The entire track of Circle 3 cannot be bunched up against Circle 2. 
* **The Threshold:** A maximum of **50% to 60%** (configured via `c3_c2_proximity_tolerance_pct`) of the 16 or 32 zone sector arcs are allowed to reside within a tight pixel buffer range of Circle 2. The remaining arc segments must explicitly cross out of the buffer zone and occupy the open channel space towards Circle 4.

### 12.2 Circle 4 Clustering Suppression Constraint
* **The Rule:** Circle 3 cannot be completely clustered right up against the edge of Circle 4. 
* **The Threshold:** A portion of the tracking points must map to the middle region between Circle 2 and Circle 4. If 100% of the detected points for the candidate line sit immediately adjacent to Circle 4, the track is dropped as an edge-reflection artifact.

---

## 13. Sinusoidal Defect Signature Analysis

### 13.1 Mathematical Modeling of Asymmetric Deformations
When a component is structurally defective due to an eccentric shift or an elliptical out-of-round deformation, its boundary behavior changes characteristically:

* **In the Polar Domain:** The line tracking the defect will not manifest as random erratic noise or an irregular broken path. Instead, it traces a highly continuous, smooth **sinusoidal wave** ($A \sin(B\theta + C) + D$) undulating horizontally across the vertical angular row zones.

[Normal Part]  Polar Strip Column:  |   |   | (Perfectly straight vertical tracking line)
[Defective]    Polar Strip Column:  ~ ~ ~ ~ ~ (Smooth sinusoidal wave across rows)


### 13.2 Algorithmic Handling
1. **Differentiating Noise from Defect:** While random high-frequency pixel noise is handled via interpolation or neighbor-smoothing, the system must actively monitor for low-frequency periodic oscillations across the 16 or 32 sectors.
2. **Defect Characterization:** If a smooth sine-like wave is identified with a total peak-to-trough amplitude exceeding `max_allowed_variance_epsilon`, the system must deliberately log the defect as an structural anomaly (e.g., "Component Ovality / Ellipticity Exceeded") rather than classifying it as a generic tracking failure.

---

## 14. JSON Weight Parameter Schema Updates
Extend the `circle_3_inspection_parameters` and the `coarse_segmentation` nodes in your parameter configurations to enforce these behavioral rules: