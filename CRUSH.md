# CRUSH.md - Development Guidelines for cell-lisca Repository

## Build/Lint/Test Commands

This is a multi-package repository with three Python packages. Run commands from individual package directories.

### General Commands (run from package directories)
- **Install**: `uv sync`
- **Run tests**: `python -m pytest tests/ -v`
- **Run single test**: `python -m pytest tests/test_specific.py::test_function -v`
- **Run cell-filter counter test**: `python -m cell_filter.tests.test_counter`
- **Run cell-tracker test**: `python cell_tracker/test.py`
- **Lint**: No explicit lint command configured
- **Type check**: No explicit typecheck command configured

### Module Entry Points
- **cell-viewer**: `python -m cell_viewer` or `python src/cell_viewer/main.py`
- **cell-filter pattern**: `python -m cell_filter.pattern`
- **cell-filter filter**: `python -m cell_filter.filter`
- **cell-filter extract**: `python -m cell_filter.extract`

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
- **cell-tracker**: numpy, matplotlib, scikit-image, cellpose>=4, networkx

### Testing
- Tests located in `tests/` directories or individual test files
- Integration tests may reference external data paths
- Use assertions for validation, not specific numeric results for data-dependent tests