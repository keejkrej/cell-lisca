# Cell-LISCA: Live-cell Imaging of Self-organizing Cellular Arrays

A comprehensive toolkit for analyzing micropatterned timelapse microscopy images, focusing on cell tracking, pattern recognition, and topological transitions in cellular arrays.

## Overview

cell-lisca is a multi-package Python workspace that provides specialized tools for processing and analyzing microscopy data of cells grown on micropatterns. The suite consists of four integrated applications that work together to provide a complete analysis pipeline:

1. **cell-pattern**: Pattern detection and annotation for micropatterned microscopy images
2. **cell-filter**: Cell counting, filtering, and data extraction with segmentation
3. **cell-grapher**: Cell tracking, region adjacency graph construction, and T1 transition analysis
4. **cell-tensionmap**: Integration with TensionMap for stress tensor inference
5. **cell-viewer**: Interactive visualization and frame selection for microscopy data

## Features

- **Pattern Detection**: Automatically identify and annotate micropatterns in microscopy images (cell-pattern)
- **Cell Segmentation**: Advanced cell segmentation using Cellpose (cell-filter)
- **Cell Tracking**: Track cells across timeframes with IoU-based alignment (cell-grapher)
- **Topological Analysis**: Analyze T1 transitions and cellular rearrangements (cell-grapher)
- **Tension Inference**: Calculate cellular stress tensors using VMSI (cell-tensionmap)
- **Interactive Visualization**: User-friendly interface for data exploration (cell-viewer)
- **Data Export**: Multiple output formats for downstream analysis

## Installation

### Prerequisites

- Python 3.11 or higher
- Either [uv](https://github.com/astral-sh/uv) package manager (recommended) or conda

### Option 1: Installation with uv (Recommended)

1. Clone the repository:

```bash
git clone <repository-url>
cd cell-lisca
```

2. Install all packages using uv:

```bash
uv sync
```

3. Install individual packages (if needed):

```bash
# Install cell-pattern
cd cell-pattern && uv sync

# Install cell-filter
cd cell-filter && uv sync

# Install cell-grapher
cd cell-grapher && uv sync

# Install cell-tensionmap
cd cell-tensionmap && uv sync

# Install cell-viewer
cd cell-viewer && uv sync
```

### Option 2: Installation with conda + pip

1. Clone the repository:

```bash
git clone <repository-url>
cd cell-lisca
```

2. Create a new conda environment:

```bash
conda create -n cell-lisca python=3.11
conda activate cell-lisca
```

3. Install each package:

**Cell-filter:**

```bash
cd cell-filter
pip install -e .
```

**Cell-grapher:**

```bash
cd ../cell-grapher
pip install -e .
```

**Cell-viewer:**

```bash
cd ../cell-viewer
pip install -e .
```

### Verification

To verify installation, run:

```bash
# Test cell-pattern module
cell-pattern --help

# Test cell-filter modules
cell-filter-filter --help
cell-filter-extract --help

# Test cell-grapher
cell-grapher --help

# Test cell-viewer
cell-viewer
# (should launch the GUI application)
```

**Note**: You can also use the `python -m` syntax if preferred:
- `python -m cell_pattern` instead of `cell-pattern`
- `python -m cell_filter.filter` instead of `cell-filter-filter`
- `python -m cell_filter.extract` instead of `cell-filter-extract`
- `python -m cell_grapher` instead of `cell-grapher`
- `python -m cell_tensionmap` instead of `cell-tensionmap`
- `python -m cell_viewer` instead of `cell-viewer`

## Documentation

Full documentation is available at: https://cell-lisca.readthedocs.io/

The documentation includes:
- **Installation Guide**: Detailed setup instructions
- **Quick Start**: Step-by-step tutorial
- **Module Documentation**: Detailed API reference for all modules
- **Examples**: Practical examples and workflows
- **Contributing Guide**: How to contribute to the project

To build the documentation locally:

```bash
cd docs
pip install -r requirements.txt
make html
# Open _build/html/index.html in your browser
```

### Basic Workflow

1. **Pattern Detection**: Show/save a plot of the patterns marked with bounding boxes

```bash
cell-pattern \
  --patterns /path/to/patterns.nd2 \
  --cells /path/to/cells.nd2 \
  --nuclei-channel 1 \
  --fov 0 \
  --output ./output
```
You will find `fov_000.png` in the output folder.

2. **Cell Filtering**: Filter the timelapse based on number of cells

```bash
cell-filter-filter \
  --patterns /path/to/patterns.nd2 \
  --cells /path/to/cells.nd2 \
  --nuclei-channel 1 \
  --n-cells 4 \
  --output ./output/ \
  --range 0:1 \
  --min-size 30
```

3. **Data Extraction**: Extract the timelapse of filtered cells

```bash
cell-filter-extract \
  --patterns /path/to/patterns.nd2 \
  --cells /path/to/cells.nd2 \
  --nuclei-channel 1 \
  --filter-results ./output/ \
  --output ./output/ \
  --min-frames 20 \
  --max-gap 6
```

4. **Cell Tracking & Analysis**: Track cells and analyze topological transitions

For NPY files (legacy):
```bash
cell-grapher analyze \
  --input ./output/fov_000_pattern_000_seq_000.npy \
  --output ./analysis
```

For H5 files (recommended):
```bash
# List available sequences
cell-grapher list-sequences --input ./output/bounding_boxes_all_fovs.h5

# Analyze a specific sequence
cell-grapher analyze \
  --input ./output/bounding_boxes_all_fovs.h5 \
  --fov 0 \
  --pattern 0 \
  --sequence 0 \
  --output ./analysis
```

5. **Visualization**: Interactively view and select frames

```bash
cell-viewer
# Then open files through the application's file menu
```

## Project Structure

```
cell-lisca/
├── cell-filter/          # Pattern detection and cell segmentation
│   ├── src/cell_filter/
│   │   ├── core/         # Core processing modules
│   │   ├── extract/      # Data extraction with segmentation
│   │   ├── filter/       # Cell counting and filtering
│   │   ├── pattern/      # Pattern detection
│   │   └── utils/        # Utility functions
│   └── tests/            # Test files
├── cell-grapher/         # Cell tracking and graph analysis
│   ├── src/cell_grapher/
│   │   ├── core/         # Graph processing modules
│   │   └── ...           # Analysis and visualization modules
│   └── tests/            # Test files
└── cell-viewer/          # Interactive data viewer
    ├── src/cell_viewer/
    │   └── ui/           # User interface components
    └── tests/            # Test files
```

## Dependencies

### Core Dependencies

- Python >=3.11
- numpy
- matplotlib
- pyyaml

### Package-Specific Dependencies

**cell-filter**:

- torch
- matplotlib
- opencv-python
- cellpose>4
- nd2
- xarray
- dask
- scikit-image
- scipy
- tifffile
- networkx
- pyyaml

**cell-grapher**:

- numpy
- matplotlib
- scikit-image
- networkx
- scipy
- pyyaml
- opencv-python
- btrack

**cell-viewer**:

- PySide6
- numpy
- matplotlib
- pyyaml>=6.0.2

## Development

### Running Tests

```bash
# From individual package directories
cd cell-filter && python -m pytest tests/ -v
cd cell-grapher && python -m pytest tests/ -v
cd cell-viewer && python -m pytest tests/ -v
```

### Code Style

This project follows the development guidelines outlined in [CLAUDE.md](CLAUDE.md), including:

- Google-style docstrings
- Type hints with modern union syntax
- Organized imports (standard, third-party, local)
- Consistent naming conventions

## License

MIT License - see individual package LICENSE files for details.

## Citation

If you use this software in your research, please cite:

```
cell-lisca: Cell Migration Studies with Live Cell Imaging of Single Cell Arrays
Tianyi Cao
```

## Support

For questions, issues, or contributions, please refer to the individual package documentation or contact the maintainer at ctyjackcao@outlook.com.
