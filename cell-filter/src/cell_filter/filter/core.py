"""
Core filter functionality for cell-filter.
"""

import yaml
from datetime import datetime, timezone
import time
from cell_filter.core import Cropper, CropperParameters, CellposeCounter
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class Patterns:
    """Track pattern state during analysis."""

    # Constructor

    def __init__(self, n_patterns: int) -> None:
        """Initialize pattern tracking state."""
        self.tracked: list[int] = list(range(n_patterns))
        self.dropped_zero: list[int] = []
        self.dropped_many: list[int] = []
        self.saved: dict[int, list[int]] = {i: [] for i in range(n_patterns)}

    # Public Methods

    def drop_zero(self, idx: int) -> None:
        """Mark pattern as dropped due to no nuclei found."""
        if idx in self.tracked:
            self.tracked.remove(idx)
            self.dropped_zero.append(idx)
            logger.debug(f"Dropped pattern {idx} due to zero nuclei")

    def drop_many(self, idx: int) -> None:
        """Mark pattern as dropped due to too many nuclei."""
        if idx in self.tracked:
            self.tracked.remove(idx)
            self.dropped_many.append(idx)
            logger.debug(f"Dropped pattern {idx} due to too many nuclei")

    def save_frame(self, idx: int, frame_idx: int) -> None:
        """Save a valid frame index for a pattern."""
        self.saved[idx].append(frame_idx)
        logger.debug(f"Saved frame {frame_idx} for pattern {idx}")

    def get_tracked_indices(self) -> list[int]:
        """Get list of indices being tracked."""
        return list(self.tracked)

    def get_valid_patterns(self) -> dict[int, list[int]]:
        """Get dictionary of patterns with valid frames."""
        return {idx: list(frames) for idx, frames in self.saved.items() if frames}


class Filterer:
    """Filter frame data and track nuclei counts."""

    # Constructor

    def __init__(
        self,
        patterns_path: str,
        cells_path: str,
        output_folder: str,
        n_cells: int,
        nuclei_channel: int,
    ) -> None:
        """Initialize the Filterer with configuration parameters."""
        self.output_folder = str(Path(output_folder).resolve())

        try:
            self._init_cropper(patterns_path, cells_path, nuclei_channel)
            self._init_counter(n_cells)
            logger.debug(
                f"Successfully initialized Filterer with patterns: {patterns_path} and cells: {cells_path}"
            )
        except Exception as e:
            logger.error(f"Error initializing Filterer: {e}")
            raise ValueError(f"Error initializing Filterer: {e}")

    def _init_cropper(
        self,
        patterns_path: str,
        cells_path: str,
        nuclei_channel: int,
    ) -> None:
        """Initialize the cell cropper."""
        try:
            self.cropper = Cropper(
                patterns_path=patterns_path,
                cells_path=cells_path,
                parameters=CropperParameters(nuclei_channel=nuclei_channel),
            )
            logger.debug(
                f"Initialized cropper with {self.cropper.n_fovs} views and {self.cropper.n_frames} frames"
            )
        except Exception as e:
            logger.error(f"Error initializing cropper: {e}")
            raise

    def _init_counter(self, n_cells: int) -> None:
        """Initialize the cell counter."""
        try:
            self.counter = CellposeCounter()
            self.n_cells = n_cells
            logger.debug(f"Initialized counter with n_cells={n_cells}")
        except Exception as e:
            logger.error(f"Error initializing counter: {e}")
            raise

    # Private Methods

    def _process_frame(self, frame_idx: int, min_size: int = 15) -> None:
        """Process a single frame and update pattern tracking."""
        try:
            # Load current frame
            self.cropper.load_nuclei(frame_idx)

            # Collect all nuclei for this frame
            nuclei_list = []
            tracked_indices = self.patterns.get_tracked_indices()
            for pattern_idx in tracked_indices:
                try:
                    nuclei = self.cropper.extract_nuclei(pattern_idx, normalize=True)
                    nuclei_list.append(nuclei)
                except Exception as e:
                    logger.warning(
                        f"Error extracting nuclei for frame {frame_idx}, pattern {pattern_idx}: {e}"
                    )
                    self.patterns.tracked.remove(pattern_idx)
                    continue

            if not nuclei_list:
                logger.warning(f"No valid nuclei regions found in frame {frame_idx}")
                return

            # Count nuclei for all patterns in this frame
            try:
                counts = self.counter.count_nuclei(nuclei_list, min_size=min_size)
            except Exception as e:
                logger.error(f"Error counting nuclei in frame {frame_idx}: {e}")
                return

            # Track changes for this frame
            saved_patterns = []
            # dropped_many = []
            dropped_zero = []

            # Update pattern tracking based on counts
            for pattern_idx, n_count in zip(tracked_indices, counts):
                if n_count == self.n_cells:
                    self.patterns.save_frame(pattern_idx, frame_idx)
                    saved_patterns.append(pattern_idx)
                    logger.debug(
                        f"Pattern {pattern_idx} has {self.n_cells} nuclei in frame {frame_idx}"
                    )
                # elif n_count > self.n_cells:
                #     self.patterns.drop_many(pattern_idx)
                #     dropped_many.append(pattern_idx)
                elif n_count == 0:
                    self.patterns.drop_zero(pattern_idx)
                    dropped_zero.append(pattern_idx)

            # Log detailed results for this frame
            if saved_patterns:
                logger.debug(
                    f"Frame {frame_idx}: Patterns with {self.n_cells} nuclei: {saved_patterns}"
                )
            # if dropped_many:
            #     logger.debug(
            #         f"Frame {frame_idx}: Dropped patterns (too many nuclei): {dropped_many}"
            #     )
            if dropped_zero:
                logger.debug(
                    f"Frame {frame_idx}: Dropped patterns (no nuclei): {dropped_zero}"
                )
            logger.debug(
                f"Frame {frame_idx}: Remaining tracked patterns: {self.patterns.get_tracked_indices()}"
            )

        except Exception as e:
            logger.error(f"Error processing frame {frame_idx}: {e}")
            raise ValueError(f"Error processing frame {frame_idx}: {e}")

    # Public Methods

    def filter_frames(self, fov_idx: int, min_size: int = 15) -> dict:
        """Filter frame data and track nuclei counts for a single view."""
        logger.info(f"Starting frame filtering for fov {fov_idx}")

        try:
            # Initialize filtering
            self.cropper.load_fov(fov_idx)
            self.cropper.load_patterns()
            self.cropper.process_patterns()
            self.patterns = Patterns(self.cropper.n_patterns)

            # Process each frame
            for frame_idx in range(self.cropper.n_frames):
                logger.info(f"Processing frame {frame_idx}/{self.cropper.n_frames}")
                self._process_frame(frame_idx, min_size=min_size)

            # Build results
            results = {"filter_results": self.patterns.get_valid_patterns()}

            # Log final summary
            logger.debug("\nFiltering complete:")
            logger.debug(
                f"  Final valid patterns: {list(results['filter_results'].keys())}"
            )

            # Log final dropped patterns summary
            logger.debug("\nFinal dropped patterns summary:")
            if self.patterns.dropped_many:
                logger.debug(f"  Too many nuclei: {self.patterns.dropped_many}")
            if self.patterns.dropped_zero:
                logger.debug(f"  No nuclei found: {self.patterns.dropped_zero}")

            return results

        except Exception as e:
            logger.error(f"Error in frame filtering: {e}")
            raise ValueError(f"Error in frame filtering: {e}")

    def process_fovs(self, start_fov: int, end_fov: int, min_size: int = 15) -> None:
        """
        Process a range of views sequentially.

        Args:
            start_fov (int): Starting view index (inclusive)
            end_fov (int): Ending view index (inclusive)

        Raises:
            ValueError: If view range is invalid
        """
        print("start_fov", start_fov)
        print("end_fov", end_fov)
        print("n_fovs", self.cropper.n_fovs)
        if start_fov < 0 or end_fov >= self.cropper.n_fovs or start_fov > end_fov:
            raise ValueError(
                f"Invalid view range: {start_fov} to {end_fov} (total views: {self.cropper.n_fovs})"
            )

        # Create output folder if it doesn't exist
        output_path = Path(self.output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Path for tracking file
        tracking_file = output_path / "processed_views.yaml"

        # Load previously processed views if tracking file exists
        # Store as a list of records: [{"fov": int, "datetime": iso_datetime}, ...]
        processed_views_list: list[dict] = []
        processed_views_set: set[int] = set()
        if tracking_file.exists():
            try:
                with open(tracking_file, "r") as f:
                    raw = yaml.safe_load(f) or []
                if not isinstance(raw, list):
                    raise ValueError(
                        "processed_views.yaml must be a list of {fov, datetime} records"
                    )
                for item in raw:
                    try:
                        fov = int(item["fov"])
                        dt = item.get("datetime")
                        processed_views_list.append({"fov": fov, "datetime": dt})
                        processed_views_set.add(fov)
                    except Exception:
                        logger.warning(f"Skipping invalid processed view entry: {item}")
                logger.debug(
                    f"Found {len(processed_views_list)} previously processed views"
                )
            except Exception as e:
                logger.warning(f"Error reading tracking file: {e}")

        logger.debug(
            f"Starting sequential processing for views {start_fov} to {end_fov}"
        )

        for fov_idx in range(start_fov, end_fov + 1):
            # Skip if already processed
            if fov_idx in processed_views_set:
                logger.info(f"Skipping already processed fov {fov_idx}")
                continue

            try:
                # Process the view
                time_start = time.time()
                results = self.filter_frames(fov_idx, min_size=min_size)
                time_end = time.time()
                logger.info(
                    f"Time taken to process fov {fov_idx}: {time_end - time_start} seconds"
                )

                # Save results immediately
                view_output_path = (
                    output_path
                    / f"fov_{fov_idx:03d}"
                    / f"fov_{fov_idx:03d}_filter.yaml"
                )
                view_output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(view_output_path, "w") as f:
                    yaml.safe_dump(results, f, sort_keys=False)

                # Update tracking list with current UTC ISO datetime
                now_iso = datetime.now(timezone.utc).isoformat()
                processed_views_list.append({"fov": fov_idx, "datetime": now_iso})
                processed_views_set.add(fov_idx)
                with open(tracking_file, "w") as f:
                    yaml.safe_dump(processed_views_list, f, sort_keys=False)

                logger.info(f"Saved results for fov {fov_idx} to {view_output_path}")
            except Exception as e:
                logger.error(f"Error processing fov {fov_idx}: {e}")
                # Continue with next fov even if this one fails
                continue

        logger.debug(
            f"Sequential processing complete for fovs {start_fov} to {end_fov}"
        )
        self.cropper.close_files()
