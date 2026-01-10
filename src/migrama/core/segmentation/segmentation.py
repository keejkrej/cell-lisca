import logging

import numpy as np
import tifffile
from cellpose import models

# Configure logging
logger = logging.getLogger(__name__)


class CellposeSegmenter:
    """
    A class for segmenting cells in microscopy images using Cellpose models.

    This class handles multi-channel timelapse microscopy data and applies
    Cellpose segmentation to identify individual cells.
    """

    def __init__(self):
        """
        Initialize the CellposeSegmenter.

        Note: GPU validation should be performed at application entrypoints
        before creating this instance. GPU is always enabled.
        """
        # In Cellpose 4.x, use the default model
        # GPU availability should have been validated at entrypoint
        self.model = models.CellposeModel(gpu=True)
        logger.debug("Initialized CellposeSegmenter with GPU enabled")

    def segment_image(
        self,
        image: np.ndarray,
    ) -> dict[str, np.ndarray]:
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
        logger.debug(f"Segmenting image with shape {image.shape}")

        # In Cellpose 4.x, eval returns (masks, flows, styles)
        result = self.model.eval(
            image,
        )

        masks, flows, styles = result

        n_cells = len(np.unique(masks)) - 1  # Subtract 1 for background
        logger.debug(f"Segmentation complete: found {n_cells} cells")

        return {
            'masks': masks,
            'flows': flows,
            'styles': styles,
        }

    def segment_timelapse(
        self,
        timelapse_path: str,
        frames: int | list[int] | None = None
    ) -> list[dict[str, np.ndarray]]:
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
        logger.info(f"Starting timelapse segmentation for {timelapse_path}")

        with tifffile.TiffFile(timelapse_path) as tif:
            data = tif.asarray()

        if data.ndim == 3:
            data = data[np.newaxis, ...]
        elif data.ndim == 4:
            pass
        else:
            raise ValueError(f"Expected 3D or 4D data, got shape {data.shape}")

        n_frames = data.shape[0]
        logger.debug(f"Timelapse data shape: {data.shape}, processing {n_frames} frames")

        if frames is None:
            frames = list(range(n_frames))
        elif isinstance(frames, int):
            frames = [frames]

        logger.debug(f"Processing frames: {frames}")

        results = []
        for frame_idx in frames:
            logger.debug(f"Processing frame {frame_idx}")
            frame_data = data[frame_idx]

            if frame_data.ndim == 3 and frame_data.shape[0] <= 3:
                frame_data = np.transpose(frame_data, (1, 2, 0))

            result = self.segment_image(frame_data)
            result['frame'] = frame_idx
            results.append(result)

        logger.info(f"Timelapse segmentation complete: processed {len(results)} frames")
        return results
