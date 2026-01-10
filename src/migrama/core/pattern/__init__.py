"""
Pattern detection and cropping functionality.

PatternDetector: pattern.nd2 -> CSV (cell,fov,x,y,w,h)
CellCropper: cells.nd2 + CSV -> cropped regions
"""

from .cropper import BoundingBox, CellCropper, load_bboxes_csv
from .detector import DetectorParameters, PatternDetector, PatternRecord

__all__ = [
    # Pattern detection (pattern.nd2 -> CSV)
    "PatternDetector",
    "DetectorParameters",
    "PatternRecord",
    # Cell cropping (cells.nd2 + CSV -> crops)
    "CellCropper",
    "BoundingBox",
    "load_bboxes_csv",
]
