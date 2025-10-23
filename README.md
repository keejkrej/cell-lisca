# Cell-LISCA: Live-cell Imaging of Self-organizing Cellular Arrays

A comprehensive toolkit for analyzing micropatterned timelapse microscopy images, focusing on cell tracking, pattern recognition, and topological transitions in cellular arrays.

## Overview

Cell-LISCA is a multi-package Python workspace that provides specialized tools for processing and analyzing microscopy data of cells grown on micropatterns. The suite consists of three integrated applications that work together to provide a complete analysis pipeline:

1. **cell-filter**: Pattern detection, cell counting, and data extraction with segmentation
2. **cell-grapher**: Cell tracking, region adjacency graph construction, and T1 transition analysis
3. **cell-viewer**: Interactive visualization and frame selection for microscopy data

## Features

- **Pattern Detection**: Automatically identify and annotate micropatterns in microscopy images
- **Cell Segmentation**: Advanced cell segmentation using Cellpose
- **Cell Tracking**: Track cells across timeframes with IoU-based alignment
- **Topological Analysis**: Analyze T1 transitions and cellular rearrangements
- **Interactive Visualization**: User-friendly interface for data exploration
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
# Install cell-filter
cd cell-filter && uv sync

# Install cell-grapher
cd cell-grapher && uv sync

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
# Test cell-filter modules
cell-filter-pattern --help
cell-filter-filter --help
cell-filter-extract --help

# Test cell-grapher
cell-grapher --help

# Test cell-viewer
cell-viewer
# (should launch the GUI application)
```

**Note**: You can also use the `python -m` syntax if preferred:
- `python -m cell_filter.pattern` instead of `cell-filter-pattern`
- `python -m cell_filter.filter` instead of `cell-filter-filter`
- `python -m cell_filter.extract` instead of `cell-filter-extract`
- `python -m cell_grapher` instead of `cell-grapher`
- `python -m cell_viewer` instead of `cell-viewer`

## Quick Start

### Basic Workflow

1. **Pattern Detection**: Show/save a plot of the patterns marked with bounding boxes

```bash
cell-filter-pattern \
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

```bash
cell-grapher \
  --input ./output/fov_000_pattern_000_seq_000.npy \
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
Cell-LISCA: Live-cell Imaging of Self-organizing Cellular Arrays
Tianyi Cao
```

## Support

For questions, issues, or contributions, please refer to the individual package documentation or contact the maintainer at ctyjackcao@outlook.com.
