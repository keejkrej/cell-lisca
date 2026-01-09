"""
Simple data models for inter-module communication.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


@dataclass
class BoundingBox:
    """Bounding box coordinates."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def area(self) -> int:
        """Calculate bounding box area."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center coordinates."""
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class Pattern:
    """Pattern data with bounding box."""
    pattern_id: int
    bbox: BoundingBox
    center: Tuple[int, int]
    area: int


@dataclass
class FOVData:
    """Data for a field of view."""
    fov_index: int
    patterns: List[Pattern] = field(default_factory=list)
    patterns_path: str = ""
    cells_path: str = ""
    image_shape: Optional[Tuple[int, int]] = None
    extraction_timestamp: Optional[str] = None
    
    @property
    def n_patterns(self) -> int:
        """Number of patterns in this FOV."""
        return len(self.patterns)


@dataclass
class AnalysisRecord:
    """Cell count analysis record."""
    fov_index: int
    pattern_id: int
    frame_index: int
    cell_count: int


@dataclass
class ExtractedSequence:
    """Extracted timelapse sequence data."""
    fov_idx: int
    pattern_idx: int
    seq_idx: int
    start_frame: int
    end_frame: int
    n_frames: int
    channels: List[str]
    bbox: Tuple[int, int, int, int]
    data_path: Optional[str] = None
    data: Optional[Any] = None  # numpy.ndarray


@dataclass
class PipelineConfig:
    """Pipeline configuration parameters."""
    # Pattern detection
    patterns_path: str = ""
    cells_path: str = ""
    nuclei_channel: int = 1
    
    # Analysis
    min_size: int = 15
    
    # Extraction
    n_cells: int = 4
    tolerance_gap: int = 6
    min_frames: int = 20
    
    # Tracking
    search_radius: float = 100.0
    adjacency_method: str = "boundary_length"
    
    # Output
    output_dir: str = "./output"


@dataclass
class PipelineState:
    """Pipeline execution state."""
    stage: str = "initialized"
    h5_path: Optional[str] = None
    completed_fovs: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(f"{datetime.now()}: {error}")
    
    def is_complete(self) -> bool:
        """Check if pipeline is complete."""
        return self.stage == "complete" and len(self.errors) == 0
