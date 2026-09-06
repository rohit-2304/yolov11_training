"""
ring_inspection — ML-like plug-and-play ring QA module.

Quick start
-----------
    from ring_inspection import RingInspectionModel

    # Load with defaults (no training required)
    model = RingInspectionModel()

    # Or load pre-trained product weights
    model = RingInspectionModel("weights/sap87c2.json")

    # Train on a golden-sample directory
    model.train("cropped_all_products/SAP87c2/", allow_circle_3_fallback=False)
    model.save_model_weights("weights/sap87c2.json")

    # Inference
    status, metrics, (polar_vis, cart_overlay) = model.predict(img_bgr)
    # status  : "PASS" | "REJECT"
    # metrics : radii, gap fractions, variance values, defect descriptions
    # vis     : annotated BGR images
"""

from .model import RingInspectionModel

__all__ = ["RingInspectionModel"]
