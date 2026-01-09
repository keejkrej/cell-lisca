# Cell-LISCA Architecture Redesign

## Overview

The cell-lisca project has been redesigned with a new modular architecture to improve maintainability, reduce coupling between modules, and provide clearer interfaces for data flow.

## Key Changes

### 1. New cell-core Package

A new shared package `cell-core` has been created to provide:
- **Shared I/O utilities**: HDF5 read/write functions used across modules
- **Data models**: Pydantic models defining data structures for inter-module communication
- **Pipeline interfaces**: Abstract base classes defining contracts for pipeline stages

### 2. Elimination of sys.path Hacks

Previously, cell-filter used sys.path manipulation to import from cell-pattern:
```python
# OLD - Hacky approach
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'cell-pattern', 'src'))
from cell_pattern.utils.h5_io import ...
```

Now all modules import from cell-core:
```python
# NEW - Clean approach
from cell_core.io.h5_io import ...
```

### 3. Standardized CLI Interfaces

All modules now use Typer for consistent CLI interfaces:
- cell-pattern: Already used Typer
- cell-filter: Already used Typer  
- cell-grapher: Migrated from argparse to Typer
- cell-viewer: GUI application (no CLI)

### 4. Improved Data Flow

The data flow between modules is now more explicit:

```
cell-pattern → HDF5 file (bounding boxes)
     ↓
cell-filter → HDF5 file (bounding boxes + analysis + extracted sequences)
     ↓
cell-grapher → Analysis results (graphs, transitions)
```

## Module Structure

### cell-core
- `io/h5_io.py`: HDF5 I/O utilities for all data stages
- `models/data_models.py`: Pydantic models for data validation
- `interfaces/pipeline.py`: Abstract base classes for pipeline stages

### cell-pattern
- Detects and annotates micropatterns
- Outputs bounding boxes to HDF5
- Depends on: cell-core (for I/O)

### cell-filter
- Analyzes cell counts and extracts sequences
- Reads/writes HDF5 files
- Depends on: cell-core (for I/O and models)

### cell-grapher
- Tracks cells and analyzes T1 transitions
- Reads extracted sequences
- Depends on: cell-core (for models)

### cell-viewer
- Interactive visualization GUI
- Standalone application

## Benefits

1. **Reduced Coupling**: Modules no longer need to know about each other's internal structure
2. **Clear Contracts**: Data models explicitly define what flows between stages
3. **Shared Infrastructure**: Common utilities are centralized in cell-core
4. **Better Testing**: Interfaces can be mocked for unit testing
5. **Easier Maintenance**: Changes to I/O or data structures happen in one place

## Future Improvements

1. **Unified Data Format**: Consider having cell-grapher read directly from HDF5 instead of NPY
2. **Pipeline Orchestrator**: Implement the Pipeline class from cell-core.interfaces.pipeline
3. **Configuration Management**: Centralize configuration using cell-core models
4. **Validation**: Add runtime validation using Pydantic models at module boundaries

## Migration Guide

To update existing code:

1. Replace imports from `cell_pattern.utils.h5_io` with `cell_core.io.h5_io`
2. Remove any sys.path manipulation for cross-module imports
3. Use data models from `cell_core.models` for type hints and validation
4. Follow the interface contracts in `cell_core.interfaces` for new modules
