# AGENTS.md - Development Guidelines for migrama Repository

This document contains development guidelines for contributors to the migrama repository. For usage instructions, see the documentation in the docs/ directory.

## Build/Lint/Test Commands

This is a monolithic Python package. Run commands from the repository root.

### General Commands
- **Install**: `uv sync` or `uv pip install -e .`
- **Run CLI**: `uv run migrama <command> ...`
- **Lint**: `ruff check --fix .`
- **Type check**: No explicit typecheck command configured

### Testing
- **Current Status**: No test files are currently implemented in the repository
- **Future Testing**: When tests are added, they should be placed in `tests/` directory in the repository root
- **Test Commands** (for future use):
  - `python -m pytest tests/ -v`
  - `python -m pytest tests/test_specific.py::test_function -v`

### Module Entry Points
- **migrama pattern**: `migrama pattern -p patterns.nd2 -o patterns.csv`
- **migrama analyze**: `migrama analyze -c cells.nd2 --csv patterns.csv -o analysis.csv --n-cells 4`
- **migrama extract**: `migrama extract -c cells.nd2 --csv analysis.csv -o extracted.h5`
- **migrama convert**: `migrama convert -i tiff_folder/ -o converted.h5 --nuclei-channel 0 --cell-channel 1`
- **migrama info**: `migrama info -i extracted.h5`
- **migrama graph**: `migrama graph -i extracted.h5 --fov 0 --pattern 0 --sequence 0 -o ./output`
- **migrama tension**: `migrama tension --mask xxx.npy`
- **migrama viewer**: `migrama viewer`

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
from migrama.core import PatternDetector, CellCropper
```

### Class Structure
- Use section comments: `# Constructor`, `# Private Methods`, `# Public Methods`
- Exception handling with try/except blocks and logging
- Use f-strings for formatting
- Prefer `raise ValueError("message")` over generic exceptions

### Naming Conventions
- **Classes**: PascalCase (e.g., `CellposeCounter`, `PatternDetector`)
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

## Pipeline Architecture

The migrama pipeline processes micropatterned timelapse microscopy data through four stages:

### Stage 1: Pattern Detection (`migrama pattern`)
- **Input**: `patterns.nd2` (single-frame, single-channel pattern image)
- **Output**: `patterns.csv` with columns: `cell,fov,x,y,w,h`
  - `cell`: pattern index within FOV
  - `fov`: field of view index
  - `x,y,w,h`: bounding box coordinates

### Stage 2: Cell Analysis (`migrama analyze`)
- **Input**: `cells.nd2` + `patterns.csv`
- **Output**: `analysis.csv` with columns: `cell,fov,x,y,w,h,t0,t1`
  - `t0,t1`: longest contiguous frame range where target cell count is maintained

### Stage 3: Sequence Extraction (`migrama extract`)
- **Input**: `cells.nd2` + `analysis.csv`
- **Output**: `extracted.h5` containing:
  - Cropped timelapse sequences
  - Tracked nuclei masks
  - Tracked cell masks aligned to nuclei IDs

### Stage 4: Graph Analysis (`migrama graph`)
- **Input**: `extracted.h5` (tracked segmentation layer)
- **Output**: Region adjacency networks, T1 transition analysis

## Core Classes

### Pattern Detection
- `PatternDetector`: Detects patterns from ND2 files, outputs CSV
- `DetectorParameters`: Configuration for detection algorithm

### Cell Cropping
- `CellCropper`: Loads cells.nd2 + CSV, provides cropping methods
- `BoundingBox`: Dataclass for bounding box coordinates
- `load_bboxes_csv()`: Utility to load CSV into dict[fov, list[BoundingBox]]

### Segmentation & Tracking
- `CellposeCounter`: Counts cells in images
- `CellposeSegmenter`: Segments cells using Cellpose
- `CellTracker`: Tracks cells across frames

### Graph Analysis
- `CellGrapher`: Builds and analyzes region adjacency graphs
