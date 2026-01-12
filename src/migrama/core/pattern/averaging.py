"""Time-averaging for pattern detection from phase contrast images.

This module provides streaming time-averaging of microscopy data to enhance
pattern contrast by smearing out cell motion. It works with multi-FOV
timelapse ND2 files and outputs one averaged TIFF per FOV.
"""

import logging
from pathlib import Path

import numpy as np
import tifffile

from ..io.nikon import get_nd2_frame, load_nd2
from ..progress import ProgressEmitter

logger = logging.getLogger(__name__)


class PatternAverager:
    """Stream and average time-lapse data for pattern detection.

    This class performs manual streaming over frames to avoid memory issues
    with large datasets. It accumulates sum and count per pixel and saves
    an averaged image per FOV immediately after processing.

    Parameters
    ----------
    cells_path : str
        Path to the cells ND2 file (multi-FOV timelapse).
    cell_channel : int
        Channel index for the cell/phase contrast channel.
    t0 : int | None
        Start frame index (inclusive). Negative values are normalized.
        Defaults to 0.
    t1 : int | None
        End frame index (exclusive). Negative values are normalized.
        Defaults to None (last frame).
    output_dir : str | Path
        Output directory for averaged TIFF files.
    """

    def __init__(
        self,
        cells_path: str,
        cell_channel: int,
        t0: int | None = None,
        t1: int | None = None,
        output_dir: str | Path = ".",
    ) -> None:
        """Initialize the PatternAverager."""
        self.cells_path = Path(cells_path).resolve()
        self.cell_channel = cell_channel
        self.t0 = t0
        self.t1 = t1
        self.output_dir = Path(output_dir).resolve()
        self._progress = ProgressEmitter()

        self.cells_xarr, self.metadata = load_nd2(self.cells_path)
        self.n_fovs = self.metadata.n_fovs
        self.n_frames = self.metadata.n_frames
        self.n_channels = self.metadata.n_channels

        if self.cell_channel < 0 or self.cell_channel >= self.n_channels:
            raise ValueError(f"Cell channel {self.cell_channel} out of range (0-{self.n_channels - 1})")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Initialized PatternAverager: {self.n_fovs} FOVs, {self.n_frames} frames, channel {self.cell_channel}"
        )

    @property
    def progress(self):
        """Get the progress signal for connecting callbacks."""
        return self._progress.progress

    def _normalize_frame_index(self, idx: int, n_frames: int) -> int:
        """Normalize a frame index, handling negative values.

        Parameters
        ----------
        idx : int
            Frame index (may be negative).
        n_frames : int
            Total number of frames.

        Returns
        -------
        int
            Normalized non-negative frame index.
        """
        if idx < 0:
            return max(0, n_frames + idx)
        return min(idx, n_frames)

    def _resolve_frame_range(self) -> tuple[int, int]:
        """Resolve t0/t1 to actual frame indices.

        Returns
        -------
        tuple[int, int]
            (start, end) frame indices with end exclusive.
        """
        t0 = self.t0 if self.t0 is not None else 0
        t1 = self.t1 if self.t1 is not None else self.n_frames

        t0 = self._normalize_frame_index(t0, self.n_frames)
        t1 = self._normalize_frame_index(t1, self.n_frames)

        if t0 >= t1:
            raise ValueError(f"Invalid frame range [{t0}, {t1})")

        logger.debug(f"Frame range: [{t0}, {t1}) of {self.n_frames} frames")
        return t0, t1

    def _average_fov(self, fov_idx: int, t0: int, t1: int) -> np.ndarray:
        """Compute time-averaged image for a single FOV.

        Parameters
        ----------
        fov_idx : int
            FOV index.
        t0 : int
            Start frame (inclusive).
        t1 : int
            End frame (exclusive).

        Returns
        -------
        np.ndarray
            Time-averaged image as float32.
        """
        frame_count = t1 - t0
        height = self.metadata.height
        width = self.metadata.width

        sum_accum = np.zeros((height, width), dtype=np.float64)
        count_accum = np.zeros((height, width), dtype=np.float64)

        for frame_idx in range(t0, t1):
            frame = get_nd2_frame(self.cells_xarr, fov_idx, self.cell_channel, frame_idx)

            valid_mask = frame > 0
            sum_accum += frame.astype(np.float64) * valid_mask
            count_accum += valid_mask

            self._progress.emit("averaging", "frame", frame_idx - t0 + 1, frame_count)

        count_accum = np.maximum(count_accum, 1)
        averaged = (sum_accum / count_accum).astype(np.float32)

        logger.debug(f"FOV {fov_idx}: averaged {frame_count} frames")
        return averaged

    def run(self) -> list[Path]:
        """Run averaging for all FOVs and save results.

        Returns
        -------
        list[Path]
            List of output file paths.
        """
        t0, t1 = self._resolve_frame_range()
        output_paths = []

        for fov_idx in range(self.n_fovs):
            output_name = f"patterns_avg_fov_{fov_idx}.tif"
            output_path = self.output_dir / output_name

            logger.info(f"Processing FOV {fov_idx}/{self.n_fovs - 1}")
            averaged = self._average_fov(fov_idx, t0, t1)

            tifffile.imwrite(output_path, averaged)
            output_paths.append(output_path)
            logger.info(f"Saved {output_path}")

        logger.info(f"Completed averaging: {len(output_paths)} files written to {self.output_dir}")
        return output_paths
