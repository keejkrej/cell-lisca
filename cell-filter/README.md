# Cell Filter

Filter micropatterned timelapse microscopy image datasets by the number of cells per pattern.

## Overview

Cell Filter is a Python package designed to analyze micropatterned timelapse microscopy images. It segments nuclei/cytoplasm, counts cells per micropattern, and extracts time-series data for patterns that match desired cell count criteria.

**Key Features:**

- Automated cell counting using Cellpose segmentation
- Micropattern detection and analysis
- Time-series extraction for qualifying patterns
- Support for ND2 and TIFF image formats
- GPU-accelerated processing
- Command-line interface and Python API

## Installation

### Prerequisites

- Python ≥3.11
- GPU support recommended (CUDA-compatible GPU for Cellpose)

### Install from Source

```bash
git clone https://github.com/keejkrej/cell-filter.git
# or
git clone https://gitlab.physik.uni-muenchen.de/LDAP_ls-raedler/cell-filter.git

cd cell-filter
# Recommended
uv sync

# Alternative
pip install -e .
```

### Dependencies

The package automatically installs the following key dependencies:

- `cellpose>4` - Cell segmentation
- `torch` - Deep learning framework
- `nd2` - ND2 file reading
- `numpy`, `scipy`, `scikit-image` - Image processing
- `matplotlib` - Visualization
- `opencv-python` - Computer vision utilities

## Quick Start

### 1. Prepare Your Data

Organize your microscopy data:

```
data/
├── patterns.nd2          # Pattern reference images
├── cells_timelapse.nd2   # Timelapse cell images
└── analysis/             # Output directory (will be created)
```

### 2. Basic Workflow

The typical workflow consists of three steps:

#### Step 1: Inspect Patterns and Channels

```bash
python -m cell_filter.pattern --patterns data/patterns.nd2 --cells data/cells_timelapse.nd2 --nuclei-channel 1 --fov-all
```

#### Step 2: Filter Patterns by Cell Count

```bash
python -m cell_filter.filter --patterns data/patterns.nd2 --cells data/cells_timelapse.nd2 --nuclei-channel 1 --n-cells 4 --output data/analysis/ --all
python -m cell_filter.filter --patterns data/patterns.nd2 --cells data/cells_timelapse.nd2 --nuclei-channel 1 --n-cells 4 --output data/analysis/ --range 0:5
```

#### Step 3: Extract Time-Series for Qualifying Patterns

```bash
python -m cell_filter.extract --patterns data/patterns.nd2 --cells data/cells_timelapse.nd2 --filter-results data/analysis/ --output data/analysis/ --min-frames 20 --max-gap 6
```

## Usage

### Command-Line Interface

#### Pattern Inspection

View micropatterns with bounding boxes and indices:

```bash
# View specific FOV
python -m cell_filter.pattern --patterns data/patterns.nd2 --cells data/cells.nd2 --fov 0

# View all FOVs
python -m cell_filter.pattern --patterns data/patterns.nd2 --cells data/cells.nd2 --fov-all

# Specify nuclei channel (default: 1)
python -m cell_filter.pattern --patterns data/patterns.nd2 --cells data/cells.nd2 --nuclei-channel 2 --fov 0
```

#### Cell Counting and Filtering

Analyze cell counts per micropattern:

```bash
# Process all FOVs, looking for patterns with 4 cells
python -m cell_filter.filter --patterns data/patterns.nd2 --cells data/cells.nd2 --n-cells 4 --output data/analysis/ --all

# Process specific range of FOVs (0 to 5, exclusive)
python -m cell_filter.filter --patterns data/patterns.nd2 --cells data/cells.nd2 --n-cells 4 --output data/analysis/ --range "0:5"

# Enable debug logging
python -m cell_filter.filter --patterns data/patterns.nd2 --cells data/cells.nd2 --n-cells 4 --output data/analysis/ --debug --all
```

#### Time-Series Extraction

Extract image sequences for patterns meeting criteria:

```bash
# Extract sequences with minimum 20 frames
python -m cell_filter.extract --patterns data/patterns.nd2 --cells data/cells.nd2 --filter-results data/analysis/ --output data/analysis/ --min-frames 20

# Allow maximum 6 frame gaps before splitting sequences
python -m cell_filter.extract --patterns data/patterns.nd2 --cells data/cells.nd2 --filter-results data/analysis/ --output data/analysis/ --min-frames 20 --max-gap 6
```

#### Command Options

**Common Parameters:**

- `--patterns`: Path to pattern reference file (ND2/TIFF)
- `--cells`: Path to timelapse cell images (ND2/TIFF)
- `--nuclei-channel`: Channel index for nuclei (default: 1)
- `--output`: Output directory for results
- `--debug`: Enable detailed logging

**Filtering-specific:**

- `--n-cells`: Target number of cells per pattern (default: 4)
- `--all`: Process all FOVs
- `--range`: Process specific FOV range (e.g., "0:10")

**Extraction-specific:**

- `--filter-results`: Directory containing filter results
- `--min-frames`: Minimum frames required for extraction (default: 20)
- `--max-gap`: Maximum frame gap before splitting sequences (default: 6)

### Using Scripts Directly

Alternatively, you can copy and modify the example scripts in `scripts/`:

```bash
# Copy scripts to your working directory
cp scripts/pattern.py my_pattern_analysis.py
cp scripts/filter.py my_filtering.py
cp scripts/extract.py my_extraction.py

# Edit parameters in the script files, then run:
python my_pattern_analysis.py
python my_filtering.py
python my_extraction.py
```

### Python API

You can also use the package programmatically:

```python
from cell_filter.pattern import Patterner
from cell_filter.filter import Filterer
from cell_filter.extract import Extractor

# Pattern inspection
patterner = Patterner(
    patterns_path="data/patterns.nd2",
    cells_path="data/cells.nd2",
    nuclei_channel=1
)
patterner.plot_view(0)  # FOV 0
patterner.close()

# Filtering
filterer = Filterer(
    patterns_path="data/patterns.nd2",
    cells_path="data/cells.nd2",
    output_folder="data/analysis/",
    n_cells=4,
    nuclei_channel=1
)
filterer.process_fovs(0, 5)  # Process FOVs 0-4

# Extraction
extractor = Extractor(
    patterns_path="data/patterns.nd2",
    cells_path="data/cells.nd2",
    output_folder="data/analysis/",
    nuclei_channel=1
)
extractor.extract(
    filter_results_dir="data/analysis/",
    min_frames=20,
    max_gap=6
)
```

## Output Structure

The analysis creates the following output structure:

```
data/analysis/
├── processed_views.yaml             # Tracking of processed FOVs (append-only log)
├── fov_000/
│   ├── fov_000_filter.yaml          # Filter results (pattern -> frames)
│   ├── fov_000_pattern_003_seq_001.npy   # Extracted stack (pattern, cells, segmentation)
│   ├── fov_000_pattern_003_seq_001.yaml  # Metadata for the stack
│   └── ...
├── fov_001/
│   └── ...
└── ...
```

## GPU Requirements

Cell Filter uses Cellpose for cell segmentation, which benefits significantly from GPU acceleration:

- **Recommended**: CUDA-compatible GPU with ≥4GB VRAM
- **Minimum**: CPU-only processing (significantly slower)
- **GPU Detection**: Automatic GPU validation on startup

If GPU is unavailable, the package will fall back to CPU processing with a performance warning.

## Troubleshooting

### Common Issues

**GPU not detected:**

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If CUDA unavailable, processing will use CPU (slower)
```

**File format issues:**

- Ensure ND2 files are readable: `python -c "import nd2; print(nd2.imread('your_file.nd2').shape)"`
- For TIFF files, verify they're readable by tifffile: `python -c "import tifffile; print(tifffile.imread('your_file.tiff').shape)"`

**Memory issues:**

- Reduce processing range: use `--range "0:2"` instead of `--all`
- Close other applications to free GPU/system memory
- Consider processing smaller image crops

**Channel indexing:**

- ND2 files use 0-based channel indexing
- Use pattern inspection (`--view-all`) to verify correct nuclei channel
- Common nuclei channels: DAPI (often channel 0 or 1), Hoechst (varies)

### Debug Mode

Enable detailed logging to troubleshoot issues:

```bash
python -m cell_filter.filter --debug --patterns data/patterns.nd2 --cells data/cells.nd2 --range "0:1"
```

### Performance Tips

- **Use GPU**: Ensure CUDA is properly installed for ~10x speed improvement
- **Batch processing**: Process views in ranges rather than individually
- **Memory management**: Close applications and clear GPU memory between runs
- **File optimization**: Use compressed TIFF or optimized ND2 files when possible

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## License

MIT License - see `pyproject.toml` for details.

## Citation

If you use Cell Filter in your research, please cite:

```
Cao, T. (2025). Cell Filter: Filtering micropatterned timelapse microscopy images based on number of cells.
GitHub repository: https://github.com/keejkrej/cell-filter
```
