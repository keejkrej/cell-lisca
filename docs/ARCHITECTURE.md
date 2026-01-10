# Migrama Architecture

## Overview

Migrama is a monolithic Python package for analyzing micropatterned timelapse microscopy images. The codebase is organized into modular packages under `src/migrama/`.

## Package Structure

### Core Package (`src/migrama/core/`)

Shared utilities and interfaces used across all modules:

- `io/`: HDF5 I/O utilities, ND2 file handling
- `models/`: Pydantic models for data validation
- `interfaces/`: Abstract base classes for pipeline stages
- `segmentation/`: Cell segmentation utilities (Cellpose wrapper)
- `tracking/`: Cell tracking utilities
- `network/`: Graph operations for region adjacency analysis
- `pattern/`: Pattern detection utilities

### Pipeline Stages

The analysis pipeline consists of four stages:

```
Stage 1: Pattern Detection (migrama.pattern)
  Input: patterns.nd2 (single-frame, single-channel)
  Output: patterns.csv with columns: cell,fov,x,y,w,h

Stage 2: Cell Analysis (migrama.analyze)
  Input: cells.nd2 + patterns.csv
  Output: analysis.csv with columns: cell,fov,x,y,w,h,t0,t1

Stage 3: Sequence Extraction (migrama.extract)
  Input: cells.nd2 + analysis.csv
  Output: extracted.h5 with cropped timelapse sequences and tracked masks

Stage 4: Graph Analysis (migrama.graph)
  Input: extracted.h5
  Output: Region adjacency graphs, T1 transition analysis
```

### Feature Packages

- **pattern**: Pattern detection and bounding box generation
- **analyze**: Cell counting and valid frame range detection
- **extract**: Cropped sequence extraction with segmentation and tracking
- **graph**: Region adjacency graph construction and T1 transition analysis
- **tension**: TensionMap VMSI integration for stress tensor inference
- **viewer**: PySide6-based interactive visualization

## CLI Interface

All modules use Typer for consistent CLI interfaces. Entry point:

```bash
migrama <command> [options]
```

Available commands: `pattern`, `analyze`, `extract`, `graph`, `tension`, `viewer`

## Data Flow

1. `migrama pattern --patterns patterns.nd2 --output patterns.csv`
2. `migrama analyze --cells cells.nd2 --csv patterns.csv --output analysis.csv`
3. `migrama extract --cells cells.nd2 --csv analysis.csv --output extracted.h5`
4. `migrama graph --input extracted.h5 --output ./analysis --fov 0 --pattern 0 --sequence 0`

## Dependencies

All dependencies are listed in `pyproject.toml`. Key dependencies:
- **numpy**, **scipy**, **scikit-image**: Core scientific computing
- **cellpose**: Cell segmentation
- **nd2**: Nikon ND2 file format support
- **h5py**: HDF5 file I/O
- **networkx**, **btrack**: Graph operations and cell tracking
- **PySide6**: GUI for viewer module
- **typer**: CLI framework
