"""
Pipeline interfaces and base classes for cell-lisca modules.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from ..models.data_models import (
    FOVData, AnalysisRecord, ExtractedSequence, 
    PipelineConfig, PipelineState
)

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """Base class for pipeline stages."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize stage with configuration."""
        self.config = config
        self.state = PipelineState(stage=self.__class__.__name__)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def validate_inputs(self) -> bool:
        """Validate required inputs exist."""
        pass
    
    @abstractmethod
    def process(self) -> Dict[str, Any]:
        """Execute the pipeline stage."""
        pass
    
    def cleanup(self) -> None:
        """Clean up resources."""
        pass


class PatternDetectionInterface(PipelineStage):
    """Interface for pattern detection modules."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.state.stage = "pattern_detection"
    
    @abstractmethod
    def detect_patterns(self, fov_idx: int) -> FOVData:
        """Detect patterns in a specific FOV."""
        pass
    
    @abstractmethod
    def extract_bounding_boxes(self, fov_idx: int) -> Dict[str, Any]:
        """Extract bounding boxes for a specific FOV."""
        pass
    
    def validate_inputs(self) -> bool:
        """Check that input files exist."""
        patterns_path = Path(self.config.patterns_path)
        cells_path = Path(self.config.cells_path)
        
        if not patterns_path.exists():
            self.state.add_error(f"Patterns file not found: {patterns_path}")
            return False
        
        if not cells_path.exists():
            self.state.add_error(f"Cells file not found: {cells_path}")
            return False
        
        return True


class AnalysisInterface(PipelineStage):
    """Interface for cell counting analysis modules."""
    
    def __init__(self, config: PipelineConfig, h5_path: str):
        super().__init__(config)
        self.h5_path = h5_path
        self.state.stage = "analysis"
    
    @abstractmethod
    def count_cells(self, fov_idx: int, pattern_idx: int, frame_idx: int) -> int:
        """Count cells in a specific pattern at a specific frame."""
        pass
    
    @abstractmethod
    def analyze_fov(self, fov_idx: int) -> List[AnalysisRecord]:
        """Analyze all patterns in an FOV across all frames."""
        pass
    
    def validate_inputs(self) -> bool:
        """Check that H5 file exists and has bounding boxes."""
        h5_path = Path(self.h5_path)
        
        if not h5_path.exists():
            self.state.add_error(f"H5 file not found: {h5_path}")
            return False
        
        # Additional validation could be added here
        return True


class ExtractionInterface(PipelineStage):
    """Interface for data extraction modules."""
    
    def __init__(self, config: PipelineConfig, h5_path: str):
        super().__init__(config)
        self.h5_path = h5_path
        self.state.stage = "extraction"
    
    @abstractmethod
    def find_sequences(self, analysis_data: List[AnalysisRecord]) -> List[ExtractedSequence]:
        """Find valid sequences based on cell count criteria."""
        pass
    
    @abstractmethod
    def extract_sequence(self, sequence: ExtractedSequence) -> Any:
        """Extract image data for a sequence."""
        pass
    
    def validate_inputs(self) -> bool:
        """Check that H5 file exists and has analysis data."""
        h5_path = Path(self.h5_path)
        
        if not h5_path.exists():
            self.state.add_error(f"H5 file not found: {h5_path}")
            return False
        
        # Additional validation could be added here
        return True


class TrackingInterface(PipelineStage):
    """Interface for cell tracking modules."""
    
    def __init__(self, config: PipelineConfig, input_path: str):
        super().__init__(config)
        self.input_path = input_path
        self.state.stage = "tracking"
    
    @abstractmethod
    def track_cells(self, data: Any) -> Any:
        """Track cells across frames."""
        pass
    
    @abstractmethod
    def build_graph(self, tracked_data: Any) -> Any:
        """Build region adjacency graph."""
        pass
    
    @abstractmethod
    def detect_transitions(self, graph: Any) -> List[Dict[str, Any]]:
        """Detect topological transitions."""
        pass
    
    def validate_inputs(self) -> bool:
        """Check that input file exists."""
        input_path = Path(self.input_path)
        
        if not input_path.exists():
            self.state.add_error(f"Input file not found: {input_path}")
            return False
        
        return True


class Pipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.stages: Dict[str, PipelineStage] = {}
        self.state = PipelineState(stage="initialized")
        self.logger = logging.getLogger(__name__)
    
    def add_stage(self, name: str, stage: PipelineStage) -> None:
        """Add a pipeline stage."""
        self.stages[name] = stage
        self.logger.info(f"Added stage: {name}")
    
    def run(self, stage_name: Optional[str] = None) -> Dict[str, Any]:
        """Run pipeline or specific stage."""
        results = {}
        
        if stage_name:
            # Run specific stage
            if stage_name not in self.stages:
                raise ValueError(f"Stage not found: {stage_name}")
            
            stage = self.stages[stage_name]
            if not stage.validate_inputs():
                raise ValueError(f"Input validation failed for stage: {stage_name}")
            
            results[stage_name] = stage.process()
            self.state.stage = f"completed_{stage_name}"
        else:
            # Run all stages
            for name, stage in self.stages.items():
                self.logger.info(f"Running stage: {name}")
                
                if not stage.validate_inputs():
                    raise ValueError(f"Input validation failed for stage: {name}")
                
                results[name] = stage.process()
                self.logger.info(f"Completed stage: {name}")
            
            self.state.stage = "complete"
        
        return results
    
    def get_state(self) -> PipelineState:
        """Get current pipeline state."""
        return self.state
