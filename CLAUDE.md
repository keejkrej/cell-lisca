# CLAUDE.md - Development Guidelines for cell-lisca Repository

This document contains development guidelines for contributors to the cell-lisca repository. For usage instructions, see the individual package USAGE.md files.

## Build/Lint/Test Commands

This is a multi-package repository with three Python packages. Run commands from individual package directories.

### General Commands (run from package directories)
- **Install**: `uv sync`
- **Lint**: `ruff check --fix .`
- **Type check**: No explicit typecheck command configured

### Testing
- **Current Status**: No test files are currently implemented in the repository
- **Future Testing**: When tests are added, they should be placed in `tests/` directories within each package
- **Test Commands** (for future use):
  - `python -m pytest tests/ -v`
  - `python -m pytest tests/test_specific.py::test_function -v`

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
- **cell-viewer**: PySide6, numpy, matplotlib, pyyaml>=6.0.2
- **cell-filter**: numpy, torch, matplotlib, opencv-python, cellpose>4, nd2, xarray, dask, scikit-image, scipy, tifffile, networkx, pyyaml
- **cell-grapher**: numpy, matplotlib, scikit-image, networkx, scipy, pyyaml, opencv-python, btrack

### Testing
- **Current Status**: No test files are currently implemented
- **Future Implementation**: When tests are added, they should be placed in `tests/` directories within each package
- **Test Guidelines**: Use assertions for validation, not specific numeric results for data-dependent tests
- **Integration Tests**: May reference external data paths when implemented

## Data Format Specifications

### Cell-Filter Output Format
The cell-filter project outputs NPY files with the following structure:
- **Shape**: `(n_frames, n_channels+2, height, width)`
- **Channels**: `[pattern, cell_channels..., segmentation]`
- **Data Type**: Preserves original ND2 data type
- **Segmentation**: Cellpose-based with local cell IDs (background=0, cells=1,2,3,...)

### Cell-Grapher Input Requirements
- Cell-filter NPY file with segmentation channel (last channel)
- Optional YAML metadata file for channel information
- Segmentation masks should have local cell IDs (background=0, cells=1,2,3...)