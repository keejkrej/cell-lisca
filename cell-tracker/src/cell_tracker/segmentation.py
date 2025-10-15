import numpy as np
from typing import List, Dict, Optional, Union
import tifffile
from cellpose import models


class CellposeSegmenter:
    """
    A class for segmenting cells in microscopy images using Cellpose models.
    
    This class handles multi-channel timelapse microscopy data and applies
    Cellpose segmentation to identify individual cells.
    """
    
    def __init__(
        self,
        gpu: bool = True,
    ):
        """
        Initialize the CellposeSegmenter.
        
        Parameters
        ----------
        gpu : bool
            Whether to use GPU for computation
        """
        self.gpu = gpu
        # In Cellpose 4.x, use the default model
        self.model = models.CellposeModel(gpu=self.gpu)
        
    def segment_image(
        self,
        image: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Segment a single image using Cellpose.
        
        Parameters
        ----------
        image : np.ndarray
            Input image with shape (height, width) or (height, width, channels)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'masks': Segmentation masks (2D array)
            - 'flows': Flow fields from Cellpose
            - 'styles': Style vectors
        """
        # In Cellpose 4.x, eval returns (masks, flows, styles)
        
        # In Cellpose 4.x, eval returns (masks, flows, styles)
        result = self.model.eval(
            image,
        )
        
        masks, flows, styles = result
        
        return {
            'masks': masks,
            'flows': flows,
            'styles': styles,
        }
    
    def segment_timelapse(
        self,
        timelapse_path: str,
        frames: Optional[Union[int, List[int]]] = None
    ) -> List[Dict[str, np.ndarray]]:
        """
        Segment a timelapse microscopy file.
        
        Parameters
        ----------
        timelapse_path : str
            Path to the timelapse TIFF file
        frames : int or list of int, optional
            Specific frames to process. If None, processes all frames
            
        Returns
        -------
        list of dict
            List of segmentation results for each frame
        """
        with tifffile.TiffFile(timelapse_path) as tif:
            data = tif.asarray()
        
        if data.ndim == 3:
            data = data[np.newaxis, ...]
        elif data.ndim == 4:
            pass
        else:
            raise ValueError(f"Expected 3D or 4D data, got shape {data.shape}")
        
        n_frames = data.shape[0]
        
        if frames is None:
            frames = list(range(n_frames))
        elif isinstance(frames, int):
            frames = [frames]
            
        results = []
        for frame_idx in frames:
            frame_data = data[frame_idx]
            
            if frame_data.ndim == 3 and frame_data.shape[0] <= 3:
                frame_data = np.transpose(frame_data, (1, 2, 0))
            
            result = self.segment_image(frame_data)
            result['frame'] = frame_idx
            results.append(result)
            
        return results
    
    def save_segmentation(
        self,
        segmentation_result: Dict[str, np.ndarray],
        output_path: str
    ):
        """
        Save segmentation results to a numpy file.
        
        Parameters
        ----------
        segmentation_result : dict
            Segmentation result from segment_image
        output_path : str
            Path to save the .npy file
        """
        np.save(output_path, segmentation_result)
    
    def load_segmentation(self, path: str) -> Dict[str, np.ndarray]:
        """
        Load segmentation results from a numpy file.
        
        Parameters
        ----------
        path : str
            Path to the .npy file
            
        Returns
        -------
        dict
            Segmentation data
        """
        return np.load(path, allow_pickle=True).item()