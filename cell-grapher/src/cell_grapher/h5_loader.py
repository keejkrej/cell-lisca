"""
H5 data loader for cell-grapher to read directly from cell-filter H5 files.
"""

import numpy as np
import yaml
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from cell_core.io.h5_io import (
    load_extracted_sequence,
    list_extracted_sequences,
)

logger = logging.getLogger(__name__)


class H5SegmentationLoader:
    """
    A class for loading pre-computed segmentation masks from cell-filter H5 files.
    
    This class loads segmentation data directly from H5 files that contain
    extracted sequences with segmentation masks, eliminating the need for NPY files.
    """
    
    def __init__(self):
        """Initialize the H5SegmentationLoader."""
        pass
        
    def load_cell_filter_data(
        self,
        h5_path: str,
        fov_idx: int,
        pattern_idx: int,
        seq_idx: int,
        yaml_path: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Load cell-filter output data including segmentation masks from H5 file.
        
        Parameters
        ----------
        h5_path : str
            Path to the H5 file containing extracted sequences
        fov_idx : int
            Field of view index
        pattern_idx : int
            Pattern index
        seq_idx : int
            Sequence index
        yaml_path : str, optional
            Path to the corresponding YAML metadata file.
            If None, will try to find it automatically.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'data': Full timelapse data (frames, channels, height, width)
            - 'segmentation_masks': List of segmentation masks
            - 'metadata': YAML metadata if available
            - 'channels': List of channel names
        """
        # Load the sequence data from H5
        data, metadata = load_extracted_sequence(h5_path, fov_idx, pattern_idx, seq_idx)
        
        # Extract segmentation masks (last channel)
        segmentation_masks = data[:, -1, :, :]  # Last channel is segmentation
        
        # Extract image data (all channels except segmentation)
        image_data = data[:, :-1, :, :]
        
        # Try to find YAML metadata if not provided
        if yaml_path is None:
            h5_path_obj = Path(h5_path)
            yaml_path = str(h5_path_obj.with_suffix('.yaml'))
        
        # Load YAML metadata if exists
        yaml_metadata = None
        if Path(yaml_path).exists():
            with open(yaml_path, 'r') as f:
                yaml_metadata = yaml.safe_load(f)
        
        # Get channel names from metadata or use defaults
        channels = metadata.get('channels', [f'channel_{i}' for i in range(image_data.shape[1])])
        
        return {
            'data': image_data,
            'segmentation_masks': segmentation_masks,
            'metadata': yaml_metadata,
            'channels': channels,
            'sequence_metadata': metadata
        }
    
    def list_sequences(self, h5_path: str) -> List[Dict[str, int]]:
        """
        List all available sequences in the H5 file.
        
        Parameters
        ----------
        h5_path : str
            Path to the H5 file
            
        Returns
        -------
        list
            List of dictionaries with fov_idx, pattern_idx, seq_idx
        """
        return list_extracted_sequences(h5_path)
    
    def validate_cell_filter_output(
        self,
        h5_path: str,
        yaml_path: Optional[str] = None
    ) -> bool:
        """
        Validate that the H5 file contains valid cell-filter output.
        
        Parameters
        ----------
        h5_path : str
            Path to the H5 file
        yaml_path : str, optional
            Path to the corresponding YAML metadata file
            
        Returns
        -------
        bool
            True if valid, False otherwise
        """
        try:
            # Check if H5 file exists
            if not Path(h5_path).exists():
                logger.error(f"H5 file not found: {h5_path}")
                return False
            
            # Try to list sequences
            sequences = self.list_sequences(h5_path)
            
            if not sequences:
                logger.error("No sequences found in H5 file")
                return False
            
            # Try to load the first sequence
            first_seq = sequences[0]
            data, metadata = load_extracted_sequence(
                h5_path, 
                first_seq['fov_idx'],
                first_seq['pattern_idx'], 
                first_seq['seq_idx']
            )
            
            # Check data format
            if len(data.shape) != 4:
                logger.error(f"Expected 4D data (frames, channels, height, width), got shape {data.shape}")
                return False
            
            if data.shape[1] < 2:  # At least one image channel + segmentation
                logger.error(f"Expected at least 2 channels (image + segmentation), got {data.shape[1]}")
                return False
            
            logger.info("H5 file validation passed")
            return True
            
        except Exception as e:
            logger.error(f"H5 file validation failed: {e}")
            return False
