import numpy as np
import yaml
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class SegmentationLoader:
    """
    A class for loading pre-computed segmentation masks from cell-filter output files.
    
    This class loads segmentation data from cell-filter NPY files that contain
    a dedicated segmentation channel, eliminating the need for running Cellpose.
    """
    
    def __init__(self):
        """Initialize the SegmentationLoader."""
        pass
        
    def load_cell_filter_data(
        self,
        npy_path: str,
        yaml_path: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Load cell-filter output data including segmentation masks.
        
        Parameters
        ----------
        npy_path : str
            Path to the NPY file containing timelapse data
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
        # Load the main data
        data = np.load(npy_path)
        
        # Try to find YAML metadata if not provided
        if yaml_path is None:
            npy_path_obj = Path(npy_path)
            yaml_path = str(npy_path_obj.with_suffix('.yaml'))
        
        metadata = {}
        channels = None
        if Path(yaml_path).exists():
            with open(yaml_path, 'r') as f:
                metadata = yaml.safe_load(f)
                channels = metadata.get('channels', [])
        
        # Extract segmentation masks (last channel)
        segmentation_masks = []
        for frame_idx in range(data.shape[0]):
            mask = data[frame_idx, -1]  # Last channel is segmentation
            segmentation_masks.append(mask)
        
        return {
            'data': data,
            'segmentation_masks': segmentation_masks,
            'metadata': metadata,
            'channels': channels,
            'segmentation_channel_idx': -1
        }
    
    def get_frame_mask(
        self,
        cell_filter_data: Dict[str, any],
        frame_idx: int
    ) -> np.ndarray:
        """
        Get segmentation mask for a specific frame.
        
        Parameters
        ----------
        cell_filter_data : dict
            Data from load_cell_filter_data
        frame_idx : int
            Frame index
            
        Returns
        -------
        np.ndarray
            Segmentation mask for the specified frame
        """
        if frame_idx >= len(cell_filter_data['segmentation_masks']):
            raise ValueError(f"Frame index {frame_idx} out of range")
        
        return cell_filter_data['segmentation_masks'][frame_idx]
    
    def get_cell_channels(
        self,
        cell_filter_data: Dict[str, any],
        exclude_channels: List[str] = ['pattern', 'segmentation']
    ) -> np.ndarray:
        """
        Extract cell image channels, excluding pattern and segmentation.
        
        Parameters
        ----------
        cell_filter_data : dict
            Data from load_cell_filter_data
        exclude_channels : List[str]
            Channel names to exclude from cell imagery
            
        Returns
        -------
        np.ndarray
            Cell channels data (frames, cell_channels, height, width)
        """
        data = cell_filter_data['data']
        channels = cell_filter_data['channels']
        
        if not channels:
            # If no channel names, assume last channel is segmentation
            cell_data = data[:, :-1]
        else:
            # Find indices of channels to keep
            keep_indices = []
            for i, channel_name in enumerate(channels):
                if channel_name not in exclude_channels:
                    keep_indices.append(i)
            
            if keep_indices:
                cell_data = data[:, keep_indices]
            else:
                # If all channels excluded, return empty array
                cell_data = np.array([])
        
        return cell_data
    
    def validate_cell_filter_output(
        self,
        npy_path: str,
        yaml_path: Optional[str] = None
    ) -> bool:
        """
        Validate that the cell-filter output has expected format.
        
        Parameters
        ----------
        npy_path : str
            Path to NPY file
        yaml_path : str, optional
            Path to YAML file
            
        Returns
        -------
        bool
            True if format is valid
        """
        try:
            data = np.load(npy_path)
            
            # Check data dimensions
            if data.ndim != 4:
                print(f"Expected 4D data, got {data.ndim}D")
                return False
            
            # Check if we have at least pattern + cell channels + segmentation
            if data.shape[1] < 2:
                print(f"Expected at least 2 channels, got {data.shape[1]}")
                return False
            
            # Check YAML metadata if provided
            if yaml_path and Path(yaml_path).exists():
                with open(yaml_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                    
                if 'channels' not in metadata:
                    print("Warning: 'channels' not found in YAML metadata")
                else:
                    channels = metadata['channels']
                    if 'segmentation' not in channels:
                        print("Warning: 'segmentation' channel not found in metadata")
            
            return True
            
        except Exception as e:
            print(f"Error validating cell-filter output: {e}")
            return False