# Cell Tracker

A specialized tool for analyzing T1 transitions in four-cell clusters using microscopy time-lapse data. This package performs automated cell segmentation, tracking, and topological analysis to identify and quantify T1 transitions - critical cell rearrangement events in tissue development.

## Features

- **Automated Cell Segmentation**: Uses Cellpose 4.x for robust cell identification in multi-channel microscopy images
- **Cell Tracking**: IoU-based tracking maintains consistent cell identities across time frames
- **T1 Transition Analysis**: Identifies and quantifies T1 transitions by analyzing adjacency relationships
- **Topology Analysis**: Comprehensive analysis of cell cluster topology and connectivity patterns
- **Visualization**: Multi-panel visualizations showing segmentation, tracking, and graph analysis
- **Export Capabilities**: Saves analysis results as CSV files and publication-ready figures

## Installation

```bash
# Install using pip (recommended)
pip install -e .

# Or using uv (faster)
uv pip install -e .
```

## Quick Start

```python
from cell_tracker.pipeline import analyze_timelapse_data

# Analyze your time-lapse data
results = analyze_timelapse_data(
    data_path='path/to/your/data.npy',
    output_dir='analysis_results',
    start_frame=0
)
```

## Data Format

Input data should be a numpy array with shape `(timeframes, channels, height, width)`:

- Channel 0: Pattern channel (optional)
- Channel 1: Nuclei (used for segmentation)
- Channel 2: Cytoplasm fluorescence (used for segmentation)
- Channel 3: Phase contrast (optional)

## Output

The analysis generates:

- Frame-by-frame segmentation visualizations
- T1 transition analysis plots
- CSV files with quantitative data
- Summary statistics and event detection

## Example Usage

See `test.py` for a complete example using the included sample data.

## Requirements

- Python ≥ 3.11
- numpy, matplotlib, scikit-image
- cellpose ≥ 4.0
- networkx, scipy, tifffile
