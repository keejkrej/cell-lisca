"""
Data models and schemas for inter-module communication.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x: int = Field(..., description="X coordinate of top-left corner")
    y: int = Field(..., description="Y coordinate of top-left corner")
    width: int = Field(..., description="Width of bounding box")
    height: int = Field(..., description="Height of bounding box")
    
    @property
    def area(self) -> int:
        """Calculate bounding box area."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center coordinates."""
        return (self.x + self.width // 2, self.y + self.height // 2)


class Pattern(BaseModel):
    """Pattern data with bounding box."""
    pattern_id: int = Field(..., description="Unique pattern identifier")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    center: Tuple[int, int] = Field(..., description="Center coordinates")
    area: int = Field(..., description="Pattern area in pixels")


class FOVData(BaseModel):
    """Data for a field of view."""
    fov_index: int = Field(..., description="FOV index")
    patterns: List[Pattern] = Field(default_factory=list, description="List of patterns")
    patterns_path: str = Field(..., description="Path to patterns image file")
    cells_path: str = Field(..., description="Path to cells image file")
    image_shape: Optional[Tuple[int, int]] = Field(None, description="Image dimensions (height, width)")
    extraction_timestamp: Optional[str] = Field(None, description="When data was extracted")
    
    @property
    def n_patterns(self) -> int:
        """Number of patterns in this FOV."""
        return len(self.patterns)


class AnalysisRecord(BaseModel):
    """Cell count analysis record."""
    fov_index: int = Field(..., description="FOV index")
    pattern_id: int = Field(..., description="Pattern identifier")
    frame_index: int = Field(..., description="Frame index in timelapse")
    cell_count: int = Field(..., description="Number of cells detected")


class ExtractedSequence(BaseModel):
    """Extracted timelapse sequence data."""
    fov_idx: int = Field(..., description="FOV index")
    pattern_idx: int = Field(..., description="Pattern index")
    seq_idx: int = Field(..., description="Sequence index")
    start_frame: int = Field(..., description="First frame in sequence")
    end_frame: int = Field(..., description="Last frame in sequence")
    n_frames: int = Field(..., description="Number of frames")
    channels: List[str] = Field(..., description="Channel names")
    bbox: Tuple[int, int, int, int] = Field(..., description="Bounding box (x, y, w, h)")
    
    # Optional data reference
    data_path: Optional[str] = Field(None, description="Path to data file")
    data: Optional[np.ndarray] = Field(None, description="Actual image data")


class PipelineConfig(BaseModel):
    """Pipeline configuration parameters."""
    # Pattern detection
    patterns_path: str = Field(..., description="Path to patterns image file")
    cells_path: str = Field(..., description="Path to cells image file")
    nuclei_channel: int = Field(1, description="Channel index for nuclei")
    
    # Analysis
    min_size: int = Field(15, description="Minimum object size for nuclei detection")
    
    # Extraction
    n_cells: int = Field(4, description="Target number of cells per pattern")
    tolerance_gap: int = Field(6, description="Max consecutive frames with wrong cell count")
    min_frames: int = Field(20, description="Minimum frames for valid sequences")
    
    # Tracking
    search_radius: float = Field(100.0, description="Maximum search radius for cell tracking")
    adjacency_method: str = Field("boundary_length", description="Method for building adjacency graphs")
    
    # Output
    output_dir: str = Field("./output", description="Output directory")


@dataclass
class PipelineState:
    """Pipeline execution state."""
    stage: str = Field(..., description="Current pipeline stage")
    h5_path: Optional[str] = Field(None, description="Path to cumulative H5 file")
    completed_fovs: List[int] = Field(default_factory=list, description="Completed FOV indices")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(f"{datetime.now()}: {error}")
    
    def is_complete(self) -> bool:
        """Check if pipeline is complete."""
        return self.stage == "complete" and len(self.errors) == 0


# Data validation schemas
class PatternDetectionSchema(BaseModel):
    """Schema for pattern detection output."""
    total_fovs: int
    total_patterns: int
    processed_fovs: List[int]
    creation_time: str
    cell_pattern_version: str = "1.0"


class AnalysisSchema(BaseModel):
    """Schema for analysis output."""
    cells_path: str
    nuclei_channel: int
    min_size: int
    creation_time: str
    total_records: int
    processed_fovs: List[int]


class ExtractionSchema(BaseModel):
    """Schema for extraction output."""
    n_cells: int
    tolerance_gap: int
    min_frames: int
    creation_time: str
    cells_path: str
    total_sequences: int
