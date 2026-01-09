"""
Core cell cropper functionality for cell-pattern.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import logging
from cell_pattern.utils import load_nd2, get_nd2_frame, get_nd2_channel_stack

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CropperParameters:
    """Parameters for the Cropper class."""

    gaussian_blur_size: tuple[int, int] = (11, 11)
    bimodal_threshold: float = 0.1
    min_area_ratio: float = 0.5
    max_area_ratio: float = 1.5
    max_iterations: int = 10
    edge_tolerance: int = 5
    morph_dilate_size: tuple[int, int] = (5, 5)
    nuclei_channel: int = 1


class Cropper:
    """Crop and process cell data from images."""

    # Constructor

    def __init__(
        self, patterns_path: str, cells_path: str, parameters: CropperParameters
    ) -> None:
        """Initialize Cropper and set paths."""
        self.patterns_path = Path(patterns_path).resolve()
        self.cells_path = Path(cells_path).resolve()
        self.parameters = parameters

        try:
            self._init_patterns()
            self._init_cells()
            self._validate_files()
            logger.debug(
                f"Successfully initialized Cropper with patterns: {self.patterns_path} and cells: {self.cells_path}"
            )
        except Exception as e:
            self.close_files()
            logger.error(f"Error initializing ND2 readers: {e}")
            raise ValueError(f"Error initializing ND2 readers: {e}")

        self._init_memory()

    def _init_patterns(self) -> None:
        """Initialize the patterns ND2 file reader and extract metadata."""
        try:
            # Load xarray and metadata using new API
            self.patterns_xarr, metadata = load_nd2(self.patterns_path)
            self.pattern_n_channels = metadata.n_channels
            self.pattern_n_frames = metadata.n_frames
            self.pattern_n_fovs = metadata.n_fovs
            logger.debug(
                f"Channels: {self.pattern_n_channels}, Frames: {self.pattern_n_frames}, Views: {self.pattern_n_fovs}"
            )
        except Exception as e:
            logger.error(f"Error initializing patterns reader: {e}")
            raise

    def _init_cells(self) -> None:
        """Initialize the cells reader and metadata."""
        try:
            # Load xarray and metadata using new API
            self.cells_xarr, metadata = load_nd2(self.cells_path)
            self.dtype = self.cells_xarr.dtype
            self.cells_n_channels = metadata.n_channels
            self.cells_n_frames = metadata.n_frames
            self.cells_n_fovs = metadata.n_fovs
            # Store channel names from metadata
            self.cells_channel_names = metadata.channel_names

            # Validate channel count consistency
            if len(self.cells_channel_names) != self.cells_n_channels:
                raise ValueError(
                    f"Channel count mismatch in cells ND2 file: "
                    f"metadata reports {self.cells_n_channels} channels but "
                    f"found {len(self.cells_channel_names)} channel names"
                )

            logger.debug(
                f"Channels: {self.cells_n_channels}, Frames: {self.cells_n_frames}, Views: {self.cells_n_fovs}"
            )
            logger.debug(f"Channel names: {self.cells_channel_names}")
            logger.info(f"Data type: {self.dtype}")
        except Exception as e:
            logger.error(f"Error initializing cells reader: {e}")
            raise

    def _validate_files(self) -> None:
        """Validate the ND2 files meet the required specifications."""
        if self.pattern_n_channels != 1:
            raise ValueError("Patterns ND2 file should have exactly 1 channel")
        if self.pattern_n_frames != 1:
            raise ValueError("Patterns ND2 file must contain exactly 1 frame")
        if self.pattern_n_fovs != self.cells_n_fovs:
            raise ValueError(
                "Patterns and cells ND2 files must contain the same number of views"
            )
        if self.cells_n_channels < 2:
            raise ValueError("Cells ND2 file must contain at least 2 channels")

        self.n_fovs = self.cells_n_fovs
        self.n_frames = self.cells_n_frames
        logger.debug(f"Validated files: {self.n_fovs} views, {self.n_frames} frames")

    def _init_memory(self) -> None:
        """Initialize memory variables to default values."""
        self.current_fov = 0
        self.current_frame = 0
        self.patterns = None
        self.n_patterns = 0
        self.thresh = None
        self.contours = None
        self.bounding_boxes = None
        self.centers = None
        self.frame_nuclei = None
        logger.debug("Initialized memory variables")

    # Private Methods

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize an image to a range of 0-255."""
        if image is None or image.size == 0:
            raise ValueError("Image must not be None or empty")

        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
        image = image.astype(np.uint8)
        return image

    def _normalize_pct(self, image: np.ndarray, low: int, high: int) -> np.ndarray:
        """
        Normalize an image to a range of 0-255.

        Args:
            image (np.ndarray): Input image to normalize
            low (int): Lower percentile
            high (int): Higher percentile

        Returns:
            np.ndarray: Normalized image

        Raises:
            ValueError: If image is None or empty
        """
        if image is None or image.size == 0:
            raise ValueError("Image must not be None or empty")

        percentile_high = np.percentile(image[image > 0], high)
        percentile_low = np.percentile(image[image > 0], low)
        image = np.clip(image, percentile_low, percentile_high)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
        image = image.astype(np.uint8)

        return image

    def _find_contours(self, image: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """Find contours in an image using thresholding and contour detection."""
        if image is None or image.size == 0:
            raise ValueError("Image must not be None or empty")

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(image, self.parameters.gaussian_blur_size, 0)

        # Apply thresholding
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply morphological operations
        kernel = np.ones(self.parameters.morph_dilate_size, np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return list(contours), thresh

    def _filter_contours_by_area(self, contours: list[np.ndarray]) -> list[np.ndarray]:
        """
        Iteratively filter a list of contours based on their area distribution.
        The algorithm removes contours that are far from the mean area until the
        coefficient‑of‑variation (CV) falls below ``self.parameters.bimodal_threshold``
        or ``self.parameters.max_iterations`` is reached.

        Returns
        -------
        list[np.ndarray]
            The filtered contour list.
        """
        if not contours:
            raise ValueError("No contours provided")

        areas = np.array([cv2.contourArea(c) for c in contours])
        current_contours = list(contours)
        current_areas = areas.copy()
        cv = float("inf")

        for iteration in range(self.parameters.max_iterations):
            cv = np.std(current_areas) / np.mean(current_areas)
            if cv < self.parameters.bimodal_threshold:
                break

            mean_area = np.mean(current_areas)
            min_area = self.parameters.min_area_ratio * mean_area
            max_area = self.parameters.max_area_ratio * mean_area

            new_contours = []
            new_areas = []
            for cnt, area in zip(current_contours, current_areas):
                if min_area <= area <= max_area:
                    new_contours.append(cnt)
                    new_areas.append(area)

            current_contours = new_contours
            current_areas = np.array(new_areas)

            if len(current_contours) == 0:
                logger.warning("All contours were removed during iterative filtering")
                break

        logger.debug(f"After {iteration + 1} iterations, CV reduced to {cv:.3f}")
        return current_contours

    def _contours_to_geometry(
        self, contours: list[np.ndarray], image_shape: tuple[int, int]
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int]]]:
        """
        Convert *already filtered* contours into centre points and bounding boxes.
        Returns parallel lists of (center_y, center_x) and (x, y, w, h).
        """
        centers: list[tuple[int, int]] = []
        bboxes: list[tuple[int, int, int, int]] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append((center_y, center_x))
            bboxes.append((x, y, w, h))

        # Deterministic ordering: sort by Y then X.
        sorted_idx = sorted(
            range(len(centers)), key=lambda i: (centers[i][0], centers[i][1])
        )
        centers = [centers[i] for i in sorted_idx]
        bboxes = [bboxes[i] for i in sorted_idx]

        logger.debug(
            f"Converted {len(contours)} contours to {len(centers)} geometry entries"
        )
        return centers, bboxes

    def _filter_by_edge_tolerance(
        self, contours: list[np.ndarray], image_shape: tuple[int, int]
    ) -> list[np.ndarray]:
        """Remove contours whose bounding boxes intersect the edge tolerance margin."""
        kept: list[np.ndarray] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (
                x < self.parameters.edge_tolerance
                or y < self.parameters.edge_tolerance
                or x + w > image_shape[1] - self.parameters.edge_tolerance
                or y + h > image_shape[0] - self.parameters.edge_tolerance
            ):
                continue
            kept.append(contour)
        return kept

    def _extract_region(
        self, frame: np.ndarray, pattern_idx: int, normalize: bool
    ) -> np.ndarray:
        """Extract a region from a frame based on pattern index."""
        if frame is None:
            raise ValueError("Frame not provided")
        if pattern_idx >= self.n_patterns or pattern_idx < 0:
            raise ValueError(
                f"Pattern index {pattern_idx} out of range (0-{self.n_patterns - 1})"
            )
        if self.bounding_boxes is None:
            raise ValueError("No bounding boxes provided")

        try:
            x, y, w, h = self.bounding_boxes[pattern_idx]
            region = frame[y : y + h, x : x + w]
            if normalize:
                region = self._normalize(region)
            return region
        except Exception as e:
            logger.error(f"Error extracting region: {e}")
            raise ValueError(f"Error extracting region: {e}")

    # Public Methods

    def close_files(self) -> None:
        """Safely close all ND2 readers."""
        # When using nd2.imread with xarray/dask, we don't need to explicitly close files
        # The xarray DataArray objects don't have a close method
        # Context managers handle file closure automatically
        logger.debug(
            "ND2 files are managed by xarray/dask - no explicit closure needed"
        )

    def load_fov(self, fov_idx: int) -> None:
        """Load a specific view from the ND2 files."""
        if fov_idx >= self.n_fovs or fov_idx < 0:
            raise ValueError(f"View index {fov_idx} out of range (0-{self.n_fovs - 1})")
        self.current_fov = fov_idx
        logger.debug(f"Loaded view {fov_idx}")

    def load_patterns(self) -> None:
        """Load the patterns from ND2 file."""
        try:
            # Extract the pattern frame using utility function
            # For patterns file, we select frame 0, channel 0, and the current view (P)
            self.patterns = get_nd2_frame(self.patterns_xarr, self.current_fov, 0, 0)
            logger.debug(f"Loaded patterns for view {self.current_fov}")
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            raise ValueError(f"Error loading patterns: {e}")

    def load_nuclei(self, frame_idx: int) -> None:
        """Load nuclei frame from ND2 file."""
        if frame_idx >= self.n_frames:
            raise ValueError(
                f"Frame index {frame_idx} out of range (0-{self.n_frames - 1})"
            )
        try:
            # Extract the nuclei frame using utility function
            # Select the specific frame, nuclei channel, and current view
            self.frame_nuclei = get_nd2_frame(
                self.cells_xarr,
                self.current_fov,
                self.parameters.nuclei_channel,
                frame_idx,
            )
            logger.debug(
                f"Loaded nuclei frame {frame_idx} for view {self.current_fov} from channel {self.parameters.nuclei_channel}"
            )
        except Exception as e:
            logger.error(f"Error loading nuclei: {e}")
            raise ValueError(f"Error loading nuclei: {e}")

    def load_cell(self, frame_idx: int) -> None:
        """Load frames from all channels in the cells ND2 file as a 3D array."""
        if frame_idx >= self.n_frames:
            raise ValueError(
                f"Frame index {frame_idx} out of range (0-{self.n_frames - 1})"
            )
        try:
            # Extract frames from all channels at once
            # Select the specific frame and current view
            self.frame_cell = get_nd2_channel_stack(
                self.cells_xarr, self.current_fov, frame_idx
            )
            logger.debug(
                f"Loaded cell frames {frame_idx} for view {self.current_fov} from all channels at once"
            )
        except Exception as e:
            logger.error(f"Error loading cell: {e}")
            raise ValueError(f"Error loading cell: {e}")

    def process_patterns(self) -> None:
        """Process pattern image to extract contours and their bounding boxes."""
        if self.patterns is None:
            raise ValueError("Patterns must be loaded before processing")

        # Normalise the raw pattern frame (percentile stretch)
        self.patterns_norm = self._normalize_pct(self.patterns, 10, 90)

        # Detect raw contours from the normalised image
        contours, self.thresh = self._find_contours(self.patterns_norm)

        # 1️⃣ Filter contours by area statistics
        contours = self._filter_contours_by_area(contours)

        # 2️⃣ Apply edge‑tolerance filter
        contours = self._filter_by_edge_tolerance(contours, self.patterns_norm.shape)

        # 3️⃣ Convert to centres and bounding boxes
        self.centers, self.bounding_boxes = self._contours_to_geometry(
            contours, self.patterns_norm.shape
        )

        # Store the final list of contours and the count
        self.contours = contours
        self.n_patterns = len(self.contours)
        logger.debug(f"Processed {self.n_patterns} patterns")

    def extract_nuclei(self, pattern_idx: int, normalize: bool = False) -> np.ndarray:
        """Extract nuclei region for a specific pattern."""
        if self.frame_nuclei is None:
            raise ValueError("Nuclei frame must be loaded before extraction")
        return self._extract_region(self.frame_nuclei, pattern_idx, normalize)

    def extract_cell(
        self, pattern_idx: int, normalize: bool = False
    ) -> list[np.ndarray]:
        """Extract regions from all loaded cell channels at once."""
        if self.frame_cell is None:
            raise ValueError("Cell frames must be loaded before extraction")

        # Handle both 3D array (new format) and list of 2D arrays (old format)
        cell_regions = []
        if isinstance(self.frame_cell, np.ndarray) and self.frame_cell.ndim == 3:
            # New format: 3D array with shape (C, H, W)
            for c in range(self.frame_cell.shape[0]):
                region = self._extract_region(
                    self.frame_cell[c], pattern_idx, normalize
                )
                cell_regions.append(region)
        else:
            # Old format: list of 2D arrays
            for channel_frame in self.frame_cell:
                region = self._extract_region(channel_frame, pattern_idx, normalize)
                cell_regions.append(region)

        return cell_regions

    def extract_pattern(self, pattern_idx: int, normalize: bool = False) -> np.ndarray:
        """Extract pattern region."""
        if self.patterns is None:
            raise ValueError("Patterns must be loaded before extraction")
        return self._extract_region(self.patterns, pattern_idx, normalize)
