# Cell Tracker Technical Documentation

## Overview

The Cell Tracker is a comprehensive Python package designed for analyzing T1 transitions in four-cell clusters from microscopy time-lapse data. T1 transitions are fundamental cell rearrangement events where cells exchange neighbors, playing a crucial role in tissue morphogenesis and development.

## System Architecture

### Core Components

1. **CellposeSegmenter** (`segmentation.py`)
   - Handles cell segmentation using Cellpose 4.x models
   - Processes multi-channel microscopy images
   - Supports both single images and time-lapse sequences

2. **CellTracker** (`tracking.py`)
   - Maintains consistent cell identities across frames using IoU overlap
   - Handles cell division, death, and tracking failures
   - Assigns global IDs for visualization consistency

3. **AdjacencyGraphBuilder** (`adjacency.py`)
   - Constructs weighted graphs representing cell-cell adjacencies
   - Calculates boundary contact lengths between neighboring cells
   - Supports multiple weighting methods (boundary_length, overlap_area)

4. **T1TransitionAnalyzer** (`analysis.py`)
   - Identifies T1 transition edges in four-cell clusters
   - Analyzes edge weight changes over time
   - Detects transition events based on configurable thresholds

5. **TopologyAnalyzer** (`analysis.py`)
   - Analyzes cluster topology and connectivity patterns
   - Classifies four-cell configurations
   - Tracks topological changes across time

6. **CellTrackingPipeline** (`pipeline.py`)
   - Integrates all components into a unified analysis workflow
   - Handles complete time-lapse processing
   - Generates comprehensive visualizations and reports

## Algorithm Details

### Cell Segmentation

The segmentation pipeline uses Cellpose's neural network models:

```python
# Initialize Cellpose model
model = models.CellposeModel(gpu=True)

# Segment multi-channel image
masks, flows, styles = model.eval(image)
```

Key features:
- GPU acceleration for fast processing
- Multi-channel input support (nuclei + cytoplasm)
- Robust performance on various cell types

### Cell Tracking Algorithm

The tracking system uses Intersection over Union (IoU) for frame-to-frame cell matching:

1. **IoU Calculation**: For each cell pair between consecutive frames:
   ```
   IoU = |intersection| / |union|
   ```

2. **Matching Strategy**:
   - Find best IoU match for each current cell
   - Apply threshold (default: 0.3) to accept matches
   - Assign new IDs to unmatched cells
   - Handle one-to-one matching to prevent conflicts

3. **Global ID Management**:
   - Maintains consistent IDs across the entire sequence
   - Tracks cell lineages and handles disappearances

### Adjacency Graph Construction

The system builds weighted graphs where:
- **Nodes**: Individual cells (labeled regions)
- **Edges**: Adjacent cell pairs
- **Weights**: Contact boundary lengths

Two methods are supported:

1. **Boundary Length Method** (recommended):
   ```python
   # Count pixel adjacencies in horizontal and vertical directions
   horizontal_adj = logical_and(mask1[:, :-1], mask2[:, 1:])
   vertical_adj = logical_and(mask1[:-1, :], mask2[1:, :])
   boundary_length = sum(horizontal_adj) + sum(vertical_adj)
   ```

2. **Overlap Area Method** (faster approximation):
   ```python
   # Use morphological dilation for contact detection
   dilated1 = binary_dilation(mask1)
   dilated2 = binary_dilation(mask2)
   overlap = logical_and(dilated1, dilated2)
   ```

### T1 Transition Detection

T1 transitions are identified by analyzing the graph topology:

1. **T1 Edge Identification**:
   - Find nodes with exactly 3 neighbors (degree = 3)
   - The T1 edge connects these two nodes
   - This edge represents the critical contact that will be lost/gained

2. **Transition Monitoring**:
   - Track edge weights over time
   - Detect significant weight changes (threshold-based)
   - Classify events as weight drops or increases

3. **Configuration Analysis**:
   ```python
   degree_sequence = sorted([graph.degree(node) for node in nodes])
   
   # Common configurations:
   # [2, 2, 3, 3] or [2, 3, 3, 4] → T1-ready
   # [1, 1, 2, 2] → Linear chain
   # [2, 2, 2, 2] → Square/diamond
   ```

## Data Flow

```
Input Image Sequence
        ↓
   Cellpose Segmentation
        ↓
   Cell Tracking (IoU)
        ↓
   Adjacency Graph Building
        ↓
   T1 Analysis & Topology
        ↓
   Visualization & Export
```

### Frame Processing Pipeline

For each frame, the system performs:

1. **Segmentation**: Extract cell masks from multi-channel image
2. **Tracking**: Match cells to previous frame, update global IDs
3. **Graph Building**: Construct weighted adjacency graph
4. **Analysis**: Identify T1 edges and analyze topology
5. **Visualization**: Generate multi-panel analysis figure
6. **Storage**: Accumulate results for time-series analysis

## Visualization System

The visualization manager creates comprehensive multi-panel figures:

1. **Raw Image Panels**: Original nuclei and cytoplasm channels
2. **Segmentation Overlay**: Colored masks with cell boundaries
3. **Tracked Visualization**: Consistent colors based on global IDs
4. **Graph Visualization**: Network representation with weighted edges
5. **Analysis Annotations**: T1 edges highlighted, topology metrics

## Configuration and Parameters

### Segmentation Parameters
- `gpu`: Enable GPU acceleration (default: True)
- Model selection: Uses default Cellpose model

### Tracking Parameters
- `iou_threshold`: Minimum IoU for cell matching (default: 0.3)
- Global ID management: Automatic increment system

### Adjacency Parameters
- `method`: 'boundary_length' or 'overlap_area' (default: 'boundary_length')
- Edge weight calculation: Pixel-accurate contact measurement

### Analysis Parameters
- `threshold_change`: Minimum weight change for T1 event detection (default: 5.0)
- Topology classification: Based on degree sequence patterns

## Performance Considerations

### Computational Complexity
- **Segmentation**: O(n × pixels) where n = number of frames
- **Tracking**: O(k²) where k = number of cells per frame
- **Graph Building**: O(k × pixels) for boundary length method
- **Analysis**: O(k + e) where e = number of edges

### Memory Usage
- Stores complete frame history for time-series analysis
- Masks and graphs accumulated in memory
- Visualization objects cached for consistent rendering

### Optimization Strategies
1. **GPU Acceleration**: Cellpose segmentation benefits significantly from GPU
2. **Method Selection**: Overlap area method trades accuracy for speed
3. **Frame Range Limiting**: Process specific frame ranges to reduce memory
4. **Output Streaming**: Save intermediate results to disk

## File Formats and I/O

### Input Formats
- **Numpy Arrays**: Primary format (.npy files)
  - Shape: `(timeframes, channels, height, width)`
  - Data type: uint16 or float32
- **TIFF Files**: Multi-channel time-lapse microscopy (future support)

### Output Formats
- **PNG Images**: Frame-by-frame analysis visualizations
- **CSV Files**: Quantitative T1 transition data
- **Numpy Arrays**: Processed masks and results (optional)

### Data Structure
```python
# Analysis results structure
results = {
    'total_frames': int,
    'frame_range': (start, end),
    't1_weight_range': (min, max),
    't1_events_detected': int,
    't1_events': [event_dict, ...],
    'topology_tracking': tracking_dict,
    'output_files': file_paths_dict
}
```

## Integration and Extensibility

### Plugin Architecture
The modular design allows easy extension:
- Custom segmentation models can replace CellposeSegmenter
- Alternative tracking algorithms can implement the same interface
- New analysis metrics can be added to the analyzer classes

### API Design
```python
# High-level API
pipeline = CellTrackingPipeline()
results = pipeline.process_timelapse(data, output_dir)

# Component-level API
segmenter = CellposeSegmenter()
tracker = CellTracker()
graph_builder = AdjacencyGraphBuilder()
analyzer = T1TransitionAnalyzer()
```

### Error Handling
- Graceful degradation when segmentation fails
- Tracking continuity maintained despite individual frame failures
- Comprehensive logging and progress reporting

## Future Development

### Planned Enhancements
1. **Multi-cluster Analysis**: Simultaneous tracking of multiple four-cell clusters
2. **3D Support**: Extension to volumetric time-lapse data
3. **Machine Learning**: Automated T1 event classification
4. **Real-time Processing**: Live analysis during image acquisition
5. **Interactive Visualization**: Web-based exploration interface

### Research Applications
- Developmental biology: Gastrulation and organogenesis studies
- Cancer research: Tumor cell migration and invasion
- Tissue engineering: Scaffold colonization analysis
- Drug screening: Compound effects on cell behavior

## Dependencies and Requirements

### Core Dependencies
- **numpy**: Numerical computations and array operations
- **matplotlib**: Visualization and figure generation
- **cellpose**: Deep learning-based cell segmentation
- **networkx**: Graph analysis and manipulation
- **scikit-image**: Image processing utilities
- **scipy**: Scientific computing functions
- **tifffile**: TIFF image format support

### System Requirements
- **Python**: ≥ 3.11 (for modern typing support)
- **Memory**: ≥ 8GB RAM (for typical datasets)
- **GPU**: CUDA-compatible for Cellpose acceleration
- **Storage**: Sufficient space for output visualizations

This technical documentation provides a comprehensive overview of the Cell Tracker system architecture, algorithms, and implementation details. The modular design ensures maintainability while the integrated pipeline provides ease of use for researchers analyzing T1 transitions in microscopy data.