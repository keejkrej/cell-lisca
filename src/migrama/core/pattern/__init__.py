"""Pattern detection and cropping functionality.

PatternDetector: pattern.nd2 -> CSV (cell,fov,x,y,w,h)
PatternAverager: cells.nd2 -> averaged TIFFs (for pattern detection from phase contrast)
CellCropper: cells.nd2 + CSV -> cropped regions
PatternFovSource: Abstract base for lazy FOV pattern sources
Nd2PatternFovSource: ND2-based pattern source
TiffPatternFovSource: TIFF-folder-based pattern source
"""

from .averaging import PatternAverager
from .cropper import BoundingBox, CellCropper, load_bboxes_csv
from .detector import DetectorParameters, PatternDetector, PatternRecord
from .source import Nd2PatternFovSource, PatternFovSource, TiffPatternFovSource

__all__ = [
    # Pattern detection (pattern.nd2 -> CSV)
    "PatternDetector",
    "DetectorParameters",
    "PatternRecord",
    # Pattern averaging (cells.nd2 -> averaged TIFFs)
    "PatternAverager",
    # Cell cropping (cells.nd2 + CSV -> crops)
    "CellCropper",
    "BoundingBox",
    "load_bboxes_csv",
    # FOV sources (lazy pattern image providers)
    "PatternFovSource",
    "Nd2PatternFovSource",
    "TiffPatternFovSource",
]
