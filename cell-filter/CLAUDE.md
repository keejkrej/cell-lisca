# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Installation and Setup:**
```bash
# Recommended: Use uv for dependency management
uv sync

# Alternative: Use pip
pip install -e .
```

**Running the Application:**
```bash
# Pattern inspection - view micropatterns with bounding boxes
python -m cell_filter.pattern --patterns data/patterns.nd2 --cells data/cells.nd2 --nuclei-channel 1 --fov 0

# Cell counting and filtering - analyze cell counts per micropattern
python -m cell_filter.filter --patterns data/patterns.nd2 --cells data/cells.nd2 --n-cells 4 --output data/analysis/ --all

# Time-series extraction - extract image sequences for qualifying patterns
python -m cell_filter.extract --patterns data/patterns.nd2 --cells data/cells.nd2 --filter-results data/analysis/ --output data/analysis/ --min-frames 20
```

**Testing:**
- Tests are located in `tests/test.ipynb` (Jupyter notebook format)
- No automated test runner configured - manual testing via notebook

**GPU Requirements:**
- Check CUDA availability: `python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"`
- GPU acceleration significantly improves Cellpose segmentation performance

## Architecture Overview

Cell Filter is a Python package for analyzing micropatterned timelapse microscopy images. The codebase follows a modular architecture with three main workflow stages:

**Core Components (`src/cell_filter/core/`):**
- `crop.py` - `Cropper` class handles ND2/TIFF file loading, pattern detection, and image cropping
- `segmentation.py` - `CellposeSegmenter` and `CellposeCounter` classes for cell segmentation and counting
- `count.py` - Cell counting utilities and validation logic

**Workflow Modules:**
- `pattern/` - Pattern inspection and visualization (`Patterner` class)
- `filter/` - Cell counting and pattern filtering (`Filterer` class with `Patterns` state tracking)
- `extract/` - Time-series extraction for qualifying patterns (`Extractor` class)

**Key Design Patterns:**
- Each workflow stage has its own module with `core.py`, `main.py`, and `__main__.py`
- `CropperParameters` dataclass centralizes configuration
- State tracking via `Patterns` class in filter module
- YAML-based output for metadata and results persistence
- GPU validation and fallback to CPU processing

**Data Flow:**
1. **Pattern Detection**: Load ND2/TIFF files → detect micropatterns → generate bounding boxes
2. **Cell Filtering**: Segment nuclei with Cellpose → count cells per pattern → filter by target cell count
3. **Time-Series Extraction**: Load filter results → extract qualifying pattern sequences → save as numpy arrays

**File I/O:**
- Supports ND2 and TIFF formats via `cell_filter.utils`
- Uses xarray for efficient data handling
- Output structure: `fov_XXX/fov_XXX_filter.yaml` for metadata, `.npy` files for image stacks

**Dependencies:**
- `cellpose>4` for deep learning-based cell segmentation
- `torch` for GPU acceleration
- `nd2` for ND2 file reading
- Standard scientific Python stack (numpy, scipy, scikit-image, opencv-python)