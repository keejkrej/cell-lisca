# Cell-Pattern: Pattern Detection for Micropatterned Microscopy Images

A specialized tool for detecting and annotating micropatterns in timelapse microscopy images, extracted from the cell-lisca suite.

## Overview

Cell-pattern provides functionality for:
- Micropattern detection in microscopy images
- Pattern annotation with bounding boxes and indices
- Visualization of detected patterns

## Features

- **Pattern Detection**: Automatic detection of micropatterns using OpenCV contour analysis
- **Visualization**: Annotated output showing pattern boundaries and indices
- **Multi-FOV Support**: Process multiple fields of view from ND2 files
- **Flexible Export**: Save annotated images to specified output paths

## Installation

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Installation with uv

```bash
# In the cell-lisca repository root
uv sync

# For cell-pattern specifically
cd cell-pattern && uv sync
```

## Usage

### Command Line Interface

```bash
# Detect patterns in a single FOV
cell-pattern --patterns patterns.nd2 --cells cells.nd2 --nuclei-channel 1 --fov 0

# Process all FOVs and save output
cell-pattern --patterns patterns.nd2 --cells cells.nd2 --nuclei-channel 1 --fov-all --output ./output/

# Enable debug logging
cell-pattern --patterns patterns.nd2 --cells cells.nd2 --debug
```

### Arguments

- `--patterns`: Path to patterns ND2 file (single channel, single frame)
- `--cells`: Path to cells ND2 file (multi-channel timelapse)
- `--nuclei-channel`: Channel index for nuclei (default: 1)
- `--fov`: Specific FOV to process (default: 0)
- `--fov-all`: Process all FOVs in the dataset
- `--output`: Output directory for saving annotated images
- `--debug`: Enable debug logging

### Python API

```python
from cell_pattern.core import Patterner

# Initialize patterner
patterner = Patterner(
    patterns_path="patterns.nd2",
    cells_path="cells.nd2",
    nuclei_channel=1
)

# Process single FOV
patterner.plot_view(fov_idx=0, output_path="./output/")

# Process all FOVs
for fov_idx in range(patterner.n_fovs):
    patterner.plot_view(fov_idx, "./output/")

# Clean up
patterner.close()
```

## Input Requirements

### Patterns File
- Format: ND2 file
- Channels: Exactly 1 channel
- Frames: Exactly 1 frame
- Content: Micropattern binary masks or contrast patterns

### Cells File
- Format: ND2 file
- Channels: Minimum 2 channels
- Frames: Variable (timelapse data)
- Content: Cell fluorescence/microscopy data

## Integration with Cell-LISCA

Cell-pattern is part of the cell-lisca workspace and provides the first step in the analysis pipeline:

1. **Pattern Detection** (cell-pattern): Identify and annotate micropatterns
2. **Filtering & Counting** (cell-filter): Filter patterns based on cell count
3. **Cell Tracking** (cell-grapher): Track cells and construct graphs
4. **Tension Analysis** (cell-tensionmap): Infer cellular tensions

## Output

Annotated PNG images showing:
- Detected patterns with green bounding boxes
- Pattern indices (0, 1, 2, ...) displayed above each pattern
- Pattern boundaries extracted from input microscopy data

## Dependencies

- numpy
- opencv-python
- matplotlib
- nd2
- xarray
- dask
- scikit-image
- scipy
- typer

## License

MIT License
