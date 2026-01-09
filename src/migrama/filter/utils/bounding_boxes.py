"""
Utility functions for loading and working with bounding box data from cell-pattern.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

from ..core.io.h5_io import (
    load_bounding_boxes_hdf5,
    get_fov_bounding_boxes,
    get_available_fovs,
)

# Configure logging
logger = logging.getLogger(__name__)


def load_bounding_boxes(h5_path: str | Path, fov_idx: int) -> Dict[str, Any]:
    """
    Load bounding box data for a specific FOV from an H5 file.
    
    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 file containing bounding box data
    fov_idx : int
        Field of view index
        
    Returns
    -------
    dict
        Dictionary containing:
        - "patterns_path": path to patterns file
        - "cells_path": path to cells file  
        - "fov_index": FOV index
        - "n_patterns": number of patterns
        - "image_shape": [height, width]
        - "patterns": list of pattern dictionaries
        
    Raises
    ------
    FileNotFoundError
        If the file doesn't exist
    ValueError
        If no data found for the requested FOV
    """
    h5_path = Path(h5_path)
    
    if not h5_path.exists():
        raise FileNotFoundError(f"Bounding box file not found: {h5_path}")
    
    if h5_path.suffix.lower() not in ['.h5', '.hdf5']:
        raise ValueError(f"Unsupported file format: {h5_path.suffix}. Use .h5 or .hdf5 files.")
    
    return get_fov_bounding_boxes(h5_path, fov_idx)


def get_bounding_box_list(h5_path: str | Path, fov_idx: int) -> List[Tuple[int, int, int, int]]:
    """
    Extract just the bounding box coordinates as a list of tuples.
    
    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 file containing bounding box data
    fov_idx : int
        FOV index
        
    Returns
    -------
    list of tuple
        List of (x, y, width, height) tuples ordered by pattern_id
    """
    data = load_bounding_boxes(h5_path, fov_idx)
    
    # Sort patterns by pattern_id to maintain consistent order
    patterns = sorted(data["patterns"], key=lambda p: p["pattern_id"])
    
    bounding_boxes = []
    for pattern in patterns:
        bbox = pattern["bbox"]
        bounding_boxes.append((bbox["x"], bbox["y"], bbox["width"], bbox["height"]))
    
    return bounding_boxes


def find_available_fovs(h5_path: str | Path) -> List[int]:
    """
    Get list of FOVs available in the H5 file.
    
    Parameters
    ----------
    h5_path : str or Path
        Path to HDF5 file
        
    Returns
    -------
    list of int
        List of FOV indices that have bounding box data
    """
    return get_available_fovs(h5_path)


def load_h5_metadata(h5_path: str | Path) -> Dict[str, Any]:
    """
    Load metadata from an H5 bounding box file.
    
    Parameters
    ----------
    h5_path : str or Path
        Path to HDF5 file
        
    Returns
    -------
    dict
        Metadata including total_fovs, total_patterns, processed_fovs
    """
    return load_bounding_boxes_hdf5(h5_path)
