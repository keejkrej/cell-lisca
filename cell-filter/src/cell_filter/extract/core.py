"""
Core extractor functionality for cell-filter.
"""

import yaml
from pathlib import Path
import numpy as np
from cell_filter.core import Cropper, CropperParameters, CellposeSegmenter
import logging

# Configure logging
logger = logging.getLogger(__name__)


class Extractor:
    """Extract frames and patterns from filtering results."""

    # Constructor

    def __init__(
        self,
        patterns_path: str,
        cells_path: str,
        output_folder: str,
        nuclei_channel: int,
    ) -> None:
        """Initialize Extractor with paths and configuration."""
        self.patterns_path = str(Path(patterns_path).resolve())
        self.cells_path = str(Path(cells_path).resolve())
        self.output_folder = str(Path(output_folder).resolve())

        # Initialize comprehensive segmentation (always enabled)
        # Note: GPU validation should have been performed at application entrypoints
        self.segmenter = CellposeSegmenter()
        logger.info("Comprehensive segmentation initialized")

        try:
            self.cropper = Cropper(
                patterns_path,
                cells_path,
                parameters=CropperParameters(nuclei_channel=nuclei_channel),
            )
            logger.info(
                f"Successfully initialized Extractor with patterns: {patterns_path} and cells: {cells_path}"
            )
        except Exception as e:
            logger.error(f"Error initializing Extractor: {e}")
            raise ValueError(f"Error initializing Extractor: {e}")

    # Private Methods

    def _refine_filter_results(
        self, filter_results: dict[int, list[int]], min_frames: int, max_gap: int = 6
    ) -> dict[int, list[int]]:
        """Refine filter results by splitting gaps and filtering by minimum frames."""
        refined_results = {}

        for pattern_idx, frames in filter_results.items():
            if not frames:
                continue

            # Sort frames to ensure chronological order
            frames.sort()

            # Initialize variables for current sequence
            sequence_start = frames[0]
            current_end = frames[0]
            sequences = []

            # Process frames to find gaps
            for i in range(1, len(frames)):
                gap = frames[i] - frames[i - 1]
                if gap > max_gap:
                    # Large gap found, save current sequence and start new one
                    sequences.append([sequence_start, current_end])
                    sequence_start = frames[i]
                    current_end = frames[i]
                else:
                    current_end = frames[i]

            # Add the last sequence
            sequences.append([sequence_start, current_end])

            # Add sequences to refined time lapse with new pattern indices
            for i, (start, end) in enumerate(sequences):
                # Format pattern index with leading zeros (3 digits)
                if end - start <= min_frames:
                    continue
                new_pattern_idx = int(f"{pattern_idx:03d}{i:03d}")
                refined_results[new_pattern_idx] = [start, end]

                logger.debug(
                    f"Split pattern {pattern_idx} into {len(sequences)} sequences"
                )
                logger.debug(f"Sequence {i}: frames {start} to {end}")

        return refined_results

    def _add_head_tail(
        self, filter_results: dict[int, list[int]], n_frames: int = 3
    ) -> dict[int, list[int]]:
        """Add head and tail frames to filter result sequences."""
        extended_results = {}

        # Get total number of frames from the cropper
        total_frames = self.cropper.n_frames

        for pattern_idx, (start, end) in filter_results.items():
            # Add head frames (before start)
            new_start = max(0, start - n_frames)

            # Add tail frames (after end), but don't exceed total_frames
            new_end = min(total_frames - 1, end + n_frames)

            extended_results[pattern_idx] = [new_start, new_end]

            logger.debug(
                f"Extended pattern {pattern_idx} with {start - new_start} head frames and {new_end - end} tail frames"
            )

        return extended_results

    def _extract_frame_stack(
        self,
        pattern_idx: int,
        start_frame: int,
        end_frame: int,
        npy_output_path: Path,
        yaml_output_path: Path,
    ) -> None:
        """Extract and save frame stack for a pattern sequence with optional segmentation."""

        pattern = self.cropper.extract_pattern(pattern_idx)  # (h, w)

        # Initialize stacks for all cell regions and segmentation
        cell_stack = []
        segmentation_stack = []

        # Extract frames
        for frame_idx in range(start_frame, end_frame + 1):
            # Load all cell channels at once
            self.cropper.load_cell(frame_idx)

            # Extract regions from all cell channels at once
            cell = self.cropper.extract_cell(pattern_idx)

            # Stack all cell regions together
            cell_stack.append(cell)

            # Perform segmentation (always enabled)
            # Combine cell channels for segmentation input
            # cell has shape (n_channels, h, w) - need to transpose for segmentation
            cell_for_segmentation = np.transpose(cell, (1, 2, 0))  # (h, w, n_channels)

            # Perform segmentation with all channels
            # Cellpose can handle both single channel (h, w) and multi-channel (h, w, channels)
            seg_result = self.segmenter.segment_image(cell_for_segmentation)
            segmentation_mask = seg_result["masks"]  # (h, w) with local IDs

            segmentation_stack.append(segmentation_mask)

        # Calculate number of frames in this sequence
        n_frames = end_frame - start_frame + 1

        # Convert cell regions stack to numpy array
        # cell_stack is a list of arrays, each with shape (n_channels, h, w)
        cell_array = np.array(cell_stack)  # Shape: (n_frames, n_channels, h, w)

        # Convert segmentation stack to numpy array
        # segmentation_stack is a list of arrays, each with shape (h, w)
        segmentation_array = np.array(segmentation_stack)  # Shape: (n_frames, h, w)

        # Add channel dimension to segmentation
        segmentation_with_channel = segmentation_array[:, np.newaxis, :, :]  # Shape: (n_frames, 1, h, w)

        # Expand pattern to match frame dimensions
        # pattern has shape (h, w), so we add new axes to make it (1, h, w)
        # then broadcast to (n_frames, h, w)
        pattern_expanded = np.broadcast_to(
            pattern[np.newaxis, :, :], (n_frames, pattern.shape[0], pattern.shape[1])
        )  # Shape: (n_frames, h, w)

        # Add channel dimension to pattern to make it (n_frames, 1, h, w)
        pattern_with_channel = pattern_expanded[:, np.newaxis, :, :]  # Shape: (n_frames, 1, h, w)

        # Stack pattern, cell data, and segmentation along channel axis
        # Order: pattern first, then cell channels, then segmentation
        # Resulting shape: (n_frames, n_channels+2, h, w)
        final_stack = np.concatenate(
            [pattern_with_channel, cell_array, segmentation_with_channel], axis=1
        )

        # Prepare metadata for npy
        pattern_bbox = (
            self.cropper.bounding_boxes[pattern_idx]
            if self.cropper.bounding_boxes
            else None
        )

        # Prepare channel info and simple JSON metadata
        n_cell_channels = cell_array.shape[1]

        # Get actual channel names from ND2 file
        if hasattr(self.cropper, 'cells_channel_names') and self.cropper.cells_channel_names:
            # Use actual channel names from ND2 file (validation done at initialization)
            channel_names = ["pattern"] + self.cropper.cells_channel_names + ["segmentation"]
        else:
            # Fallback to generic names
            channel_names = ["pattern"] + [f"cell_ch_{i}" for i in range(n_cell_channels)] + ["segmentation"]

        # Save as npy
        np.save(npy_output_path, final_stack)

        # Save simple metadata as YAML
        metadata = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "channels": channel_names,
            "pattern_bbox": pattern_bbox
        }

        with open(yaml_output_path, "w") as f:
            yaml.safe_dump(metadata, f, sort_keys=False)

        logger.info(
            f"Saved pattern {pattern_idx} frames from {start_frame} to {end_frame} to {npy_output_path}"
        )

    def _process_filter_results(
        self, filter_results: dict, fov_idx: int, output_dir: Path, min_frames: int, max_gap: int = 6
    ) -> None:
        """Process filter results for a single view."""
        # Refine results
        filter_results = self._refine_filter_results(filter_results, min_frames, max_gap)

        # Add head and tail frames
        filter_results = self._add_head_tail(filter_results)

        # Load view and patterns
        self.cropper.load_fov(fov_idx)
        self.cropper.load_patterns()
        self.cropper.process_patterns()

        # Create FOV directory
        fov_dir = output_dir / f"fov_{fov_idx:03d}"
        fov_dir.mkdir(parents=True, exist_ok=True)

        # Process each pattern
        for pattern_sequence_idx, (start_frame, end_frame) in filter_results.items():
            # Extract original pattern index and sequence number
            pattern_idx = pattern_sequence_idx // 1000
            sequence_idx = pattern_sequence_idx % 1000

            # Filenames
            filename_prefix = f"fov_{fov_idx:03d}_pattern_{pattern_idx:03d}_seq_{sequence_idx:03d}"
            npy_output_path = fov_dir / f"{filename_prefix}.npy"
            yaml_output_path = fov_dir / f"{filename_prefix}.yaml"

            try:
                self._extract_frame_stack(
                    pattern_idx=pattern_idx,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    npy_output_path=npy_output_path,
                    yaml_output_path=yaml_output_path,
                )

            except Exception as e:
                logger.warning(f"Error processing pattern {pattern_idx}: {e}")
                continue

    # Public Methods

    def extract(self, filter_results_dir: str, min_frames: int = 20, max_gap: int = 6) -> None:
        """Extract valid frames and patterns from filtering results."""
        try:
            # Create base output directory
            output_dir = Path(self.output_folder)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Find all filter YAML files in FOV directories
            yaml_files = sorted(list(Path(filter_results_dir).glob("fov_*/fov_*_filter.yaml")))

            if not yaml_files:
                raise ValueError(
                    f"No fov_*_filter.yaml files found in {filter_results_dir}"
                )

            logger.info(f"Found {len(yaml_files)} filter files")

            # Process each filter file
            for yaml_file in yaml_files:
                try:
                    with open(yaml_file, "r") as f:
                        data = yaml.safe_load(f)
                    filter_results = {
                        int(pattern_idx): frames
                        for pattern_idx, frames in data["filter_results"].items()
                    }
                    # Extract view index from file path (fov_XXX_filter.yaml)
                    fov_idx = int(yaml_file.stem.split("_")[1])
                    logger.info(f"Processing filter results for fov {fov_idx}")
                    self._process_filter_results(
                        filter_results, fov_idx, output_dir, min_frames, max_gap
                    )
                except Exception as e:
                    logger.error(f"Error processing {yaml_file}: {e}")
                    raise ValueError(f"Error processing {yaml_file}: {e}")

        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            raise ValueError(f"Error during extraction: {e}")
