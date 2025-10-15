"""
Core pattern displayer functionality for cell-filter.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
from pathlib import Path
from cell_filter.core import Cropper, CropperParameters

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

    def close(self) -> None:
        """Close all open files."""
        self.cropper.close_files()
        logger.info("Closed all files")
