"""
Core pattern displayer functionality for cell-pattern.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
from pathlib import Path
from cell_pattern.core import Cropper, CropperParameters
from cell_pattern.utils.h5_io import save_bounding_boxes_hdf5

# Configure logging
logger = logging.getLogger(__name__)


class Patterner:
    """Display patterns with bounding boxes and indices."""

    # Constructor

    def __init__(
        self,
        patterns_path: str,
        cells_path: str,
        nuclei_channel: int,
    ) -> None:
        """Initialize Patterner and set paths."""
        try:
            self.cropper = Cropper(
                patterns_path,
                cells_path,
                CropperParameters(nuclei_channel=nuclei_channel),
            )
            self.n_fovs = self.cropper.pattern_n_fovs
            logger.info(
                f"Successfully initialized Patterner with patterns: {patterns_path} and cells: {cells_path}"
            )
        except Exception as e:
            logger.error(f"Error initializing Patterner: {e}")
            raise ValueError(f"Error initializing Patterner: {e}")

    # Private Methods

    def _draw_boxes(self, image: np.ndarray) -> np.ndarray:
        """Draw bounding boxes and indices for all patterns."""
        # Convert to RGB for colored annotations
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.cropper.bounding_boxes is None:
            return image

        for pattern_idx in range(self.cropper.n_patterns):
            # Get bounding box coordinates
            bbox = self.cropper.bounding_boxes[pattern_idx]
            if bbox is not None:
                x, y, w, h = bbox

                # Draw bounding box
                cv2.rectangle(
                    image,
                    (int(x), int(y)),
                    (int(x + w), int(y + h)),
                    (0, 255, 0),  # Green color
                    2,
                )  # Line thickness

                # Add pattern index
                cv2.putText(
                    image,
                    f"{pattern_idx}",
                    (int(x), int(y) - 10),  # Position above the box
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # Font scale
                    (0, 255, 0),  # Green color
                    2,
                )  # Line thickness

        return image

    # Public Methods

    def plot_view(self, fov_idx: int, output_path: str | None = None) -> None:
        """Plot patterns image for a specific view with bounding boxes and indices."""
        try:
            # Load view and patterns
            self.cropper.load_fov(fov_idx)
            self.cropper.load_patterns()
            self.cropper.process_patterns()

            # Create figure
            plt.figure(figsize=(15, 8))
            ax = plt.gca()

            # Get patterns image and draw boxes
            if self.cropper.thresh is None:
                raise ValueError("Threshold image not available")
            patterns_image = np.copy(self.cropper.thresh)
            annotated_image = self._draw_boxes(patterns_image)

            # Plot annotated image
            ax.imshow(annotated_image)

            # Set title
            ax.set_title(f"FOV {fov_idx} - Patterns with Bounding Boxes")

            # Adjust layout
            plt.tight_layout()

            # Save or show plot
            if output_path:
                plt.savefig(Path(output_path) / f"fov_{fov_idx:03d}.png")
                logger.info(f"Saved plot to {output_path}")
            else:
                plt.show()

        except Exception as e:
            logger.error(f"Error plotting fov {fov_idx}: {e}")
            raise ValueError(f"Error plotting fov {fov_idx}: {e}")
        finally:
            plt.close()

    def extract_bounding_boxes(self, fov_idx: int) -> dict:
        """Extract bounding box data for a specific FOV.
        
        Returns
        -------
        dict
            Dictionary containing bounding box data and metadata:
            {
                "patterns_path": str,
                "cells_path": str,
                "fov_index": int,
                "n_patterns": int,
                "image_shape": [height, width],
                "patterns": [
                    {
                        "pattern_id": int,
                        "bbox": {"x": int, "y": int, "width": int, "height": int},
                        "center": {"x": int, "y": int},
                        "area": int
                    },
                    ...
                ]
            }
        """
        try:
            # Load and process patterns
            self.cropper.load_fov(fov_idx)
            self.cropper.load_patterns()
            self.cropper.process_patterns()

            if self.cropper.bounding_boxes is None:
                raise ValueError("No bounding boxes available")
                
            if self.cropper.contours is None:
                raise ValueError("No contours available")

            # Extract pattern data
            patterns_data = []
            for pattern_idx in range(self.cropper.n_patterns):
                bbox = self.cropper.bounding_boxes[pattern_idx]
                center = self.cropper.centers[pattern_idx]
                contour = self.cropper.contours[pattern_idx]
                area = int(cv2.contourArea(contour))
                
                x, y, w, h = bbox
                
                pattern_data = {
                    "pattern_id": pattern_idx,
                    "bbox": {
                        "x": int(x),
                        "y": int(y), 
                        "width": int(w),
                        "height": int(h)
                    },
                    "center": {
                        "x": int(center[1]),  # center_x
                        "y": int(center[0])   # center_y
                    },
                    "area": area
                }
                patterns_data.append(pattern_data)

            # Create metadata
            data = {
                "patterns_path": str(self.cropper.patterns_path),
                "cells_path": str(self.cropper.cells_path),
                "fov_index": fov_idx,
                "n_patterns": self.cropper.n_patterns,
                "image_shape": list(self.cropper.thresh.shape) if self.cropper.thresh is not None else None,
                "patterns": patterns_data,
                "extraction_timestamp": logger.handlers[0].formatter.formatTime(logger.makeRecord(
                    '', 0, '', 0, '', (), None
                )) if logger.handlers else None
            }

            return data

        except Exception as e:
            logger.error(f"Error extracting bounding boxes for FOV {fov_idx}: {e}")
            raise ValueError(f"Error extracting bounding boxes for FOV {fov_idx}: {e}")

    def save_bounding_boxes(self, fov_idx: int, output_path: str | Path) -> None:
        """Extract and save bounding box data for a specific FOV to HDF5.
        
        Parameters
        ----------
        fov_idx : int
            Field of view index
        output_path : str or Path
            Output directory path
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract bounding boxes
        data = self.extract_bounding_boxes(fov_idx)
        all_fovs_data = {fov_idx: data}
        
        # Save to HDF5
        output_file = output_path / f"fov_{fov_idx:03d}_patterns.h5"
        save_bounding_boxes_hdf5(all_fovs_data, output_file)
        logger.info(f"Saved bounding box data for FOV {fov_idx} to {output_file}")

    def save_all_bounding_boxes(self, output_path: str | Path) -> None:
        """Extract and save bounding box data for all FOVs to HDF5 files.
        
        Parameters
        ----------
        output_path : str or Path
            Output directory path
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for fov_idx in range(self.n_fovs):
            try:
                self.save_bounding_boxes(fov_idx, output_path)
            except Exception as e:
                logger.error(f"Failed to save bounding boxes for FOV {fov_idx}: {e}")
                continue
        
        logger.info(f"Saved bounding box data for all FOVs to {output_path}")

    def process_all_fovs_bounding_boxes(self) -> dict:
        """Extract bounding boxes for all FOVs and save to HDF5.
        
        Returns
        -------
        dict
            Summary of processing results
        """
        all_fovs_data = {}
        summary = {
            "total_fovs": self.n_fovs,
            "total_patterns": 0,
            "processed_fovs": [],
            "failed_fovs": [],
            "output_files": []
        }
        
        logger.info(f"Processing bounding boxes for all {self.n_fovs} FOVs...")
        
        for fov_idx in range(self.n_fovs):
            try:
                logger.debug(f"Processing FOV {fov_idx}...")
                fov_data = self.extract_bounding_boxes(fov_idx)
                all_fovs_data[fov_idx] = fov_data
                summary["processed_fovs"].append(fov_idx)
                summary["total_patterns"] += fov_data["n_patterns"]
                logger.debug(f"FOV {fov_idx}: {fov_data['n_patterns']} patterns")
                
            except Exception as e:
                logger.error(f"Failed to process FOV {fov_idx}: {e}")
                summary["failed_fovs"].append(fov_idx)
                continue
        
        # Save to HDF5
        hdf5_path = "bounding_boxes_all_fovs.h5"
        save_bounding_boxes_hdf5(all_fovs_data, hdf5_path)
        summary["output_files"].append(hdf5_path)
        
        logger.info(f"Processing complete: {summary['total_patterns']} patterns from {len(summary['processed_fovs'])} FOVs")
        if summary["failed_fovs"]:
            logger.warning(f"Failed FOVs: {summary['failed_fovs']}")
            
        return summary

    def close(self) -> None:
        """Close all open files."""
        self.cropper.close_files()
        logger.info("Closed all files")
