# AGENTS.md - Development Guidelines for migrama Repository

This document contains development guidelines for contributors to the migrama repository. For usage instructions, see the documentation in the docs/ directory.

## Build/Lint/Test Commands

This is a monolithic Python package. Run commands from the repository root.

### General Commands
- **Install**: `uv sync` or `pip install -e .`
- **Lint**: `ruff check --fix .`
- **Type check**: No explicit typecheck command configured

### Testing
- **Current Status**: No test files are currently implemented in the repository
- **Future Testing**: When tests are added, they should be placed in `tests/` directory in the repository root
- **Test Commands** (for future use):
  - `python -m pytest tests/ -v`
  - `python -m pytest tests/test_specific.py::test_function -v`

### Module Entry Points
- **migrama pattern**: `migrama pattern detect` or `migrama pattern extract`
- **migrama analyze**: `migrama analyze run --cells xxx.nd2 --h5 bounding_boxes.h5 --range 0:10`
- **migrama extract**: `migrama extract run --cells xxx.nd2 --h5 bounding_boxes.h5 --n-cells 4`
- **migrama graph**: `migrama graph analyze --input xxx.h5 --fov 0 --pattern 0 --sequence 0 --output ./output` or `migrama graph list-sequences --input xxx.h5`
- **migrama tension**: `migrama tension run --mask xxx.npy`
- **migrama viewer**: `migrama viewer launch`

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
from migrama.core import Cropper, CropperParameters
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

### Documentation
- **Location**: All documentation files must be placed in the `docs/` directory
- **Formats**: Use Markdown (.md) for documentation
- **Structure**: Follow the existing documentation structure in docs/

### Dependencies
- **migrama**: numpy, h5py, pydantic>=2.0.0, typer, cellpose>4, torch, torchvision, matplotlib, opencv-python, scikit-image, scipy, tifffile, networkx, btrack, nd2, xarray, dask, pyyaml>=6.0.2, PySide6

### Testing
- **Current Status**: No test files are currently implemented
- **Future Implementation**: When tests are added, they should be placed in `tests/` directories within each package
- **Test Guidelines**: Use assertions for validation, not specific numeric results for data-dependent tests
- **Integration Tests**: May reference external data paths when implemented

## Data Format Specifications

### Pipeline H5 File Structure
The pipeline uses a single cumulative H5 file that grows through three stages:

**Stage 1: migrama pattern extract** - Bounding boxes
```
/bounding_boxes/
  fov_index, pattern_id, bbox_x, bbox_y, bbox_width, bbox_height,
  center_x, center_y, area, patterns_path, cells_path, image_height, image_width
/metadata/
  total_fovs, total_patterns, processed_fovs, creation_time
  fov_000/, fov_001/... (per-FOV attrs)
```

**Stage 2: migrama analyze run** - Cell counts
```
/analysis/
  fov_index, pattern_id, frame_index, cell_count (all int32 arrays)
/analysis/metadata/
  cells_path, nuclei_channel, min_size, processed_fovs, creation_time
```

**Stage 3: migrama extract run** - Cropped sequences
```
/extracted/
  fov_{idx}/pattern_{idx}/seq_{idx}/
    data (n_frames, n_channels+2, h, w)
    channels (list of channel names)
    start_frame, end_frame, bbox_x, bbox_y, bbox_width, bbox_height (attrs)
/extracted attrs: n_cells, tolerance_gap, min_frames, cells_path, creation_time
```

### Cell-Grapher Input Requirements
- Migrama H5 file with extracted sequences containing segmentation channel (last channel)
- Optional YAML metadata file for channel information
- Segmentation masks should have local cell IDs (background=0, cells=1,2,3...)
