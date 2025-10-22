# Cell Filter Usage Guide

Cell Filter is a comprehensive tool for processing micropatterned timelapse microscopy images. It consists of three main modules that work together to detect patterns, filter frames based on cell count, and extract data with segmentation masks.

## Overview

Cell Filter processes microscopy data through a three-stage pipeline:

1. **Pattern Module**: Detects and visualizes micropatterns in images
2. **Filter Module**: Filters frames based on nuclei count criteria
3. **Extract Module**: Extracts data with segmentation masks using Cellpose

## Module 1: Pattern Detection

The pattern module identifies micropatterns in your microscopy images and creates visualizations with bounding boxes.

### Command Line Usage

```bash
cell-filter-pattern [options]
```

Or via Python module:

```bash
python -m cell_filter.pattern [options]
```

### Options

| Option             | Description                    | Default                                         |
| ------------------ | ------------------------------ | ----------------------------------------------- |
| `--patterns`       | Path to patterns ND2 file      | `data/20250806_patterns_after.nd2`              |
| `--cells`          | Path to cells ND2 file         | `data/20250806_MDCK_timelapse_crop_fov0004.nd2` |
| `--nuclei-channel` | Nuclei channel index           | `1`                                             |
| `--fov`            | Field of view index to process | `0`                                             |
| `--fov-all`        | Process all fields of view     | `False`                                         |
| `--output`         | Output directory               | `None` (current directory)                      |
| `--debug`          | Enable debug logging           | `False`                                         |

### Examples

```bash
# Process specific FOV
cell-filter-pattern \
  --patterns patterns.nd2 \
  --cells cells.nd2 \
  --nuclei-channel 1 \
  --fov 0 \
  --output ./pattern_output

# Process all FOVs
cell-filter-pattern \
  --patterns patterns.nd2 \
  --cells cells.nd2 \
  --fov-all \
  --output ./pattern_output
```

### Output

- **PNG Images**: `fov_{idx:03d}.png` - Annotated pattern images with green bounding boxes
- **Dimensions**: 15x8 inch figures with tight layout

## Module 2: Cell Filtering

The filter module analyzes frames and filters them based on nuclei count criteria.

### Command Line Usage

```bash
cell-filter-filter [options]
```

Or via Python module:

```bash
python -m cell_filter.filter [options]
```

### Options

| Option             | Description                              | Default                                         |
| ------------------ | ---------------------------------------- | ----------------------------------------------- |
| `--patterns`       | Path to patterns ND2 file                | `data/20250806_patterns_after.nd2`              |
| `--cells`          | Path to cells ND2 file                   | `data/20250806_MDCK_timelapse_crop_fov0004.nd2` |
| `--nuclei-channel` | Nuclei channel index                     | `1`                                             |
| `--output`         | Output directory                         | `data/analysis/`                                |
| `--n-cells`        | Target number of cells per pattern       | `4`                                             |
| `--debug`          | Enable debug logging                     | `False`                                         |
| `--all`            | Process all fields of view               | `False`                                         |
| `--range`          | FOV range to process (format: start:end) | `0:1`                                           |
| `--min-size`       | Minimum cell size                        | `15`                                            |

### Examples

```bash
# Process specific FOV range
cell-filter-filter \
  --patterns patterns.nd2 \
  --cells cells.nd2 \
  --nuclei-channel 1 \
  --output ./filtered_output \
  --n-cells 4 \
  --range 0:5

# Process all FOVs
cell-filter-filter \
  --patterns patterns.nd2 \
  --cells cells.nd2 \
  --nuclei-channel 1 \
  --output ./filtered_output \
  --all \
  --n-cells 4
```

### Output

- **YAML Filter Results**: `fov_{idx:03d}/fov_{idx:03d}_filter.yaml`
  ```yaml
  filter_results:
    pattern_idx: [frame1, frame2, frame3, ...] # Valid frames per pattern
  ```
- **Tracking File**: `processed_views.yaml` with FOV and datetime metadata

## Module 3: Data Extraction

The extract module processes the filtered data and extracts it with segmentation masks using Cellpose.

### Command Line Usage

```bash
cell-filter-extract [options]
```

Or via Python module:

```bash
python -m cell_filter.extract [options]
```

### Options

| Option             | Description                                  | Default                                         |
| ------------------ | -------------------------------------------- | ----------------------------------------------- |
| `--patterns`       | Path to patterns ND2 file                    | `data/20250806_patterns_after.nd2`              |
| `--cells`          | Path to cells ND2 file                       | `data/20250806_MDCK_timelapse_crop_fov0004.nd2` |
| `--nuclei-channel` | Nuclei channel index                         | `1`                                             |
| `--filter-results` | Directory with filter results                | `data/analysis/`                                |
| `--output`         | Output directory                             | `data/analysis/`                                |
| `--min-frames`     | Minimum frames per sequence                  | `20`                                            |
| `--max-gap`        | Maximum frame gap before splitting sequences | `6`                                             |
| `--debug`          | Enable debug logging                         | `False`                                         |

### Example

```bash
cell-filter-extract \
  --patterns patterns.nd2 \
  --cells cells.nd2 \
  --nuclei-channel 1 \
  --filter-results ./filtered_output \
  --output ./extracted_output \
  --min-frames 20 \
  --max-gap 6
```

### Output

- **NPY Data Files**: `fov_{idx:03d}_pattern_{pat:03d}_seq_{seq:03d}.npy`
  - **Shape**: `(n_frames, n_channels+2, height, width)`
  - **Channels**: `[pattern, cell_channels..., segmentation]`
  - **Data Type**: Preserves original ND2 data type
- **YAML Metadata**: `fov_{idx:03d}_pattern_{pat:03d}_seq_{seq:03d}.yaml`
  ```yaml
  start_frame: int
  end_frame: int
  channels: ["pattern", "channel1", "channel2", ..., "segmentation"]
  pattern_bbox: [x, y, w, h]
  ```

## Complete Workflow Example

Here's a complete example of processing ND2 microscopy data:

```bash
# Step 1: Detect patterns
cell-filter-pattern \
  --patterns experiment_patterns.nd2 \
  --cells experiment_cells.nd2 \
  --nuclei-channel 1 \
  --fov-all \
  --output ./patterns

# Step 2: Filter based on cell count
cell-filter-filter \
  --patterns experiment_patterns.nd2 \
  --cells experiment_cells.nd2 \
  --nuclei-channel 1 \
  --output ./filtered \
  --n-cells 4 \
  --all

# Step 3: Extract with segmentation
cell-filter-extract \
  --patterns experiment_patterns.nd2 \
  --cells experiment_cells.nd2 \
  --nuclei-channel 1 \
  --filter-results ./filtered \
  --output ./extracted \
  --min-frames 20 \
  --max-gap 6
```

## API Usage

You can also use the cell-filter modules programmatically:

```python
from cell_filter.pattern.main import process_patterns
from cell_filter.filter.main import filter_frames
from cell_filter.extract.main import extract_data

# Process patterns
pattern_results = process_patterns(
    input_path="data.nd2",
    output_dir="./patterns",
    pattern_size=100,
    min_distance=50
)

# Filter frames
filter_results = filter_frames(
    input_dir="./patterns",
    output_dir="./filtered",
    min_cells=2,
    max_cells=6
)

# Extract data with segmentation
extract_results = extract_data(
    input_dir="./filtered",
    output_dir="./extracted",
    model_type="cyto",
    diameter=30
)
```

## Advanced Features

### Sequence Splitting

The extract module automatically splits sequences at gaps larger than `--max-gap` frames and filters sequences shorter than `--min-frames`.

### Head/Tail Extension

Add context frames before and after each valid sequence using `--head-tail` parameter.

### Custom Cellpose Models

Use custom Cellpose models by specifying the model path:

```bash
cell-filter-extract \
  --input ./filtered \
  --output ./extracted \
  --model-path /path/to/custom/model
```

## Troubleshooting

### Common Issues

1. **Pattern Detection Fails**: Adjust `--pattern-size` and `--min-distance` parameters
2. **Too Few Cells Detected**: Check channel selection and cell count parameters
3. **Segmentation Issues**: Adjust `--diameter` parameter or try different `--model-type`

### Performance Tips

- Use GPU acceleration for Cellpose by installing `torch` with CUDA support
- Process large datasets in chunks to manage memory usage
- Adjust `--max-gap` and `--min-frames` to balance data quality vs. quantity

## Output Data Format

The final NPY files contain:

- Channel 0: Pattern information
- Channels 1-N: Original fluorescence channels
- Last channel: Segmentation masks (background=0, cells=1,2,3,...)

These files are ready for analysis with cell-grapher or visualization with cell-viewer.
