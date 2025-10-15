# Cell Grapher

A tool for creating region adjacency graphs and analyzing T1 transitions in microscopy images using pre-computed segmentations from cell-filter.

## Features

- Cell tracking using IoU-based alignment
- Region adjacency graph construction
- T1 transition analysis
- Visualization tools

## Dependencies

- numpy
- matplotlib
- scikit-image
- networkx
- scipy
- pyyaml

## Usage

```bash
cell-grapher --input data.npy --output ./output
```

Or via Python module:
```bash
python -m cell_tracker --input data.npy --output ./output
```

Or via API:
```python
from cell_tracker.pipeline import analyze_cell_filter_data

results = analyze_cell_filter_data(
    npy_path="path/to/cell_filter_output.npy",
    yaml_path="path/to/cell_filter_output.yaml",
    output_dir="analysis_output"
)
```