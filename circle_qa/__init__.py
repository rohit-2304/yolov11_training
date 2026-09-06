"""
circle_qa — plug-and-play concentric circle QA module.

Public API:
    CircleQA(params)          — evaluator with tunable params
    DEFAULT_PARAMS            — baseline parameter dict
    draw_result(img, result)  — annotate an image with QA result
"""
from .evaluator import CircleQA, DEFAULT_PARAMS
from .visualise import draw_result

__all__ = ["CircleQA", "DEFAULT_PARAMS", "draw_result"]
