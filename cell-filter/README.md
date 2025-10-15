# Cell Filter

Filtering micropatterned timelapse microscopy images based on number of cells.

## Features

- Pattern detection and visualization
- Cell counting and filtering
- Data extraction with segmentation

## Modules

- `cell-filter-pattern`: Display patterns with bounding boxes
- `cell-filter-filter`: Filter frames based on nuclei count
- `cell-filter-extract`: Extract final data with segmentation masks

## Dependencies

- numpy
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

## Usage

```bash
cell-filter-pattern [options]
cell-filter-filter [options]  
cell-filter-extract [options]
```

Or via Python modules:
```bash
python -m cell_filter.pattern [options]
python -m cell_filter.filter [options]
python -m cell_filter.extract [options]
```