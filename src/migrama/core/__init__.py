"""Migrama core: Shared utilities and interfaces for migrama modules."""

from .network import CellGrapher
from .pattern import (
    BoundingBox,
    CellCropper,
    DetectorParameters,
    PatternDetector,
    PatternRecord,
    load_bboxes_csv,
)
from .segmentation import CellposeCounter, CellposeSegmenter
from .tracking import CellTracker

__all__ = [
    # Pattern detection (pattern.nd2 -> CSV)
    "PatternDetector",
    "DetectorParameters",
    "PatternRecord",
    # Cell cropping (cells.nd2 + CSV -> crops)
    "CellCropper",
    "BoundingBox",
    "load_bboxes_csv",
    # Segmentation
    "CellposeCounter",
    "CellposeSegmenter",
    # Tracking
    "CellTracker",
    # Graph
    "CellGrapher",
]
