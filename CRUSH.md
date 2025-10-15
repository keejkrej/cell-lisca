# CRUSH.md - Development Guidelines for cell-lisca Repository

## Build/Lint/Test Commands

This is a multi-package repository with three Python packages. Run commands from individual package directories.

### General Commands (run from package directories)
- **Install**: `uv sync`
- **Run tests**: `python -m pytest tests/ -v`
- **Run single test**: `python -m pytest tests/test_specific.py::test_function -v`
- **Run cell-filter counter test**: `python -m cell_filter.tests.test_counter`
- **Run cell-grapher test**: `python cell_grapher/test.py`
- **Lint**: No explicit lint command configured
- **Type check**: No explicit typecheck command configured

### Module Entry Points
- **cell-viewer**: `python -m cell_viewer` or `python src/cell_viewer/main.py`
- **cell-filter pattern**: `python -m cell_filter.pattern`
- **cell-filter filter**: `python -m cell_filter.filter`
- **cell-filter extract**: `python -m cell_filter.extract`
- **cell-grapher**: `python -m cell_grapher --input xxx.npy --output ./output`

## Code Style Guidelines

### Python Version & Formatting
- **Python**: >=3.11 required
- **Type hints**: Use modern union syntax (`int | str` instead of `Union[int, str]`)
- **Docstrings**: Google-style with sections (Parameters, Returns, etc.)
- **Line length**: Not strictly enforced, aim for readability

### Import Organization
```python
# Standard library
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models

# Local imports
from cell_filter.core import Cropper, CropperParameters
```

### Class Structure
- Use section comments: `# Constructor`, `# Private Methods`, `# Public Methods`
- Exception handling with try/except blocks and logging
- Use f-strings for formatting
- Prefer `raise ValueError("message")` over generic exceptions

### Naming Conventions
- **Classes**: PascalCase (e.g., `CellposeCounter`, `Patterner`)
- **Functions/variables**: snake_case
- **Constants**: UPPER_SNAKE_CASE
- **Private methods**: prefix with underscore (`_method_name`)

### Dependencies
- **cell-viewer**: PySide6, numpy, matplotlib, pyyaml
- **cell-filter**: numpy, torch, opencv-python, cellpose>4, nd2, scikit-image
- **cell-grapher**: numpy, matplotlib, scikit-image, networkx, scipy, pyyaml

### Testing
- Tests located in `tests/` directories or individual test files
- Integration tests may reference external data paths
- Use assertions for validation, not specific numeric results for data-dependent tests

## Cell-Grapher Updated Usage

### Overview
Cell-grapher has been updated to use pre-computed segmentation masks from cell-filter output, eliminating the need for Cellpose dependency. It now focuses solely on cell tracking, region adjacency graph construction, and T1 transition analysis.

### Key Changes
- **No Cellpose dependency**: Uses segmentation channel from cell-filter NPY files
- **Updated API**: New `analyze_cell_filter_data()` function for easy usage
- **Pre-computed masks**: Directly uses cell-filter segmentation results
- **Renamed**: Changed from cell-tracker to cell-grapher to reflect focus on graph construction

### CLI Usage Example
```bash
# Basic usage
cell-grapher --input fov_000_pattern_000_seq_000.npy --output ./tracking_output

# With custom parameters
cell-grapher \
  --input fov_000_pattern_000_seq_000.npy \
  --output ./tracking_output \
  --yaml fov_000_pattern_000_seq_000.yaml \
  --start-frame 10 \
  --end-frame 50 \
  --iou-threshold 0.25 \
  --adjacency-method centroid_distance

# Validate input format only
cell-grapher --input fov_000_pattern_000_seq_000.npy --validate-only
```

### CLI Arguments
- `--input, -i`: Path to cell-filter NPY file (required)
- `--output, -o`: Output directory (default: tracking_output)
- `--yaml, -y`: Path to YAML metadata file (optional, auto-detected)
- `--start-frame, -s`: Starting frame number (default: 0)
- `--end-frame, -e`: Ending frame number (exclusive, optional)
- `--iou-threshold`: IoU threshold for tracking (default: 0.3)
- `--adjacency-method`: Graph building method (default: boundary_length)
- `--validate-only`: Only validate input format

### API Usage Example
```python
from cell_grapher.pipeline import analyze_cell_filter_data

results = analyze_cell_filter_data(
    npy_path="fov_000_pattern_000_seq_000.npy",
    yaml_path="fov_000_pattern_000_seq_000.yaml",  # optional
    output_dir="tracking_analysis",
    start_frame=0
)
```

### Input Requirements
- Cell-filter NPY file with segmentation channel (last channel)
- Optional YAML metadata file for channel information
- Segmentation masks should have local cell IDs (background=0, cells=1,2,3...)

### Output
- Tracked cell masks with global IDs
- T1 transition analysis plots and CSV data
- Frame-by-frame visualizations
- Topology tracking over time

## Cell-Filter Output Formats

### Overview
The cell-filter project processes micropatterned timelapse microscopy images through three main modules: `pattern`, `filter`, and `extract`.

### Pattern Module Output
- **PNG Images**: `fov_{idx:03d}.png` - Annotated pattern images with green bounding boxes and indices
- **Dimensions**: 15x8 inch figures with tight layout

### Filter Module Output
- **YAML Filter Results**: `fov_{idx:03d}/fov_{idx:03d}_filter.yaml`
  ```yaml
  filter_results:
    pattern_idx: [frame1, frame2, frame3, ...]  # Valid frames per pattern
  ```
- **Tracking File**: `processed_views.yaml` with FOV and datetime metadata

### Extract Module Output
- **NPY Data Files**: `fov_{idx:03d}_pattern_{pat:03d}_seq_{seq:03d}.npy`
  - **Shape**: `(n_frames, n_channels+2, height, width)`
  - **Channels**: `[pattern, cell_channels..., segmentation]`
  - **Data Type**: Preserves original ND2 data type
- **YAML Metadata**: `fov_{idx:03d}_pattern_{pat:03d}_seq_{seq:03d}.yaml`
  ```yaml
  start_frame: int
  end_frame: int
  channels: ["pattern", "channel1", "channel2", ..., "segmentation"]
  pattern_bbox: [x, y, w, h]
  ```

### Processing Features
- **Sequence Splitting**: Splits at gaps > `--max-gap` frames
- **Minimum Frames**: Filters sequences shorter than `--min-frames`
- **Head/Tail Extension**: Adds `n_frames` before/after each sequence
- **Segmentation**: Cellpose-based with local cell IDs (background=0, cells=1,2,3,...)