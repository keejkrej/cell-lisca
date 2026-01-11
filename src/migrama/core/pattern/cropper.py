"""
Cell cropper - crops cell regions from cell data sources using bounding boxes from CSV.

This module works with CellFovSource (ND2 or per-FOV TIFFs) + CSV bounding boxes.
Input: CellFovSource + patterns.csv (from PatternDetector)
Output: cropped cell regions for analysis
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..cell_source import CellFovSource

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box for a pattern."""

    cell: int
    fov: int
    x: int
    y: int
    w: int
    h: int


def load_bboxes_csv(csv_path: str | Path) -> dict[int, list[BoundingBox]]:
    """Load bounding boxes from CSV file.

    Parameters
    ----------
    csv_path : str | Path
        Path to CSV file with columns: cell, fov, x, y, w, h

    Returns
    -------
    dict[int, list[BoundingBox]]
        Mapping of fov_index -> list of bounding boxes
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    bboxes_by_fov: dict[int, list[BoundingBox]] = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bbox = BoundingBox(
                cell=int(row["cell"]),
                fov=int(row["fov"]),
                x=int(row["x"]),
                y=int(row["y"]),
                w=int(row["w"]),
                h=int(row["h"]),
            )
            if bbox.fov not in bboxes_by_fov:
                bboxes_by_fov[bbox.fov] = []
            bboxes_by_fov[bbox.fov].append(bbox)

    for fov in bboxes_by_fov:
        bboxes_by_fov[fov].sort(key=lambda b: b.cell)

    logger.info(f"Loaded {sum(len(v) for v in bboxes_by_fov.values())} bboxes from {len(bboxes_by_fov)} FOVs")
    return bboxes_by_fov


class CellCropper:
    """Crop cell regions from cell data sources using bounding boxes.

    This class works with CellFovSource (ND2 or per-FOV TIFFs) and uses
    pre-computed bounding boxes from a CSV file (output of PatternDetector).
    """

    def __init__(
        self,
        source: CellFovSource,
        bboxes_csv: str,
        nuclei_channel: int = 1,
    ) -> None:
        """Initialize cropper with cell source and bounding boxes.

        Parameters
        ----------
        source : CellFovSource
            Source of cell timelapse data (ND2 or TIFF)
        bboxes_csv : str
            Path to CSV file with bounding boxes (from PatternDetector)
        nuclei_channel : int
            Channel index for nuclei (default: 1)
        """
        self.source = source
        self.bboxes_csv = Path(bboxes_csv).resolve()
        self.nuclei_channel = nuclei_channel

        self.n_fovs = source.n_fovs
        self.n_frames = source.n_frames
        self.n_channels = source.n_channels
        self.height = source.height
        self.width = source.width
        self.dtype = source.dtype

        self.bboxes_by_fov = load_bboxes_csv(self.bboxes_csv)

        if self.n_channels < 2:
            raise ValueError(f"Cells source must have at least 2 channels, got {self.n_channels}")

        logger.info(
            f"Initialized CellCropper: {self.n_fovs} FOVs, {self.n_frames} frames, "
            f"{self.n_channels} channels, {sum(len(v) for v in self.bboxes_by_fov.values())} patterns"
        )

    def get_bboxes(self, fov: int) -> list[BoundingBox]:
        """Get bounding boxes for a FOV.

        Parameters
        ----------
        fov : int
            Field of view index

        Returns
        -------
        list[BoundingBox]
            Bounding boxes for this FOV
        """
        return self.bboxes_by_fov.get(fov, [])

    def n_patterns(self, fov: int) -> int:
        """Get number of patterns in a FOV."""
        return len(self.get_bboxes(fov))

    def extract_nuclei(
        self,
        fov: int,
        frame: int,
        cell: int,
        normalize: bool = False,
    ) -> np.ndarray:
        """Extract nuclei region for a specific pattern.

        Parameters
        ----------
        fov : int
            Field of view index
        frame : int
            Frame index
        cell : int
            Pattern/cell index within FOV
        normalize : bool
            Whether to normalize to 0-255

        Returns
        -------
        np.ndarray
            Cropped nuclei image (h, w)
        """
        bboxes = self.get_bboxes(fov)
        if cell >= len(bboxes):
            raise ValueError(f"Cell {cell} not found in FOV {fov} (has {len(bboxes)} patterns)")

        bbox = bboxes[cell]
        fov_data = self.source.get_fov(fov)
        img = fov_data[frame, self.nuclei_channel]
        cropped = img[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]

        if normalize:
            cropped = self._normalize(cropped)

        return cropped

    def extract_all_channels(
        self,
        fov: int,
        frame: int,
        cell: int,
        normalize: bool = False,
    ) -> np.ndarray:
        """Extract all channels for a specific pattern.

        Parameters
        ----------
        fov : int
            Field of view index
        frame : int
            Frame index
        cell : int
            Pattern/cell index within FOV
        normalize : bool
            Whether to normalize each channel to 0-255

        Returns
        -------
        np.ndarray
            Cropped image stack (n_channels, h, w)
        """
        bboxes = self.get_bboxes(fov)
        if cell >= len(bboxes):
            raise ValueError(f"Cell {cell} not found in FOV {fov} (has {len(bboxes)} patterns)")

        bbox = bboxes[cell]
        fov_data = self.source.get_fov(fov)
        stack = fov_data[frame]
        cropped = stack[:, bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]

        if normalize:
            cropped = np.stack([self._normalize(c) for c in cropped])

        return cropped

    def extract_timelapse(
        self,
        fov: int,
        cell: int,
        start_frame: int = 0,
        end_frame: int | None = None,
        channels: list[int] | None = None,
    ) -> np.ndarray:
        """Extract timelapse for a specific pattern.

        Parameters
        ----------
        fov : int
            Field of view index
        cell : int
            Pattern/cell index within FOV
        start_frame : int
            Starting frame (inclusive)
        end_frame : int | None
            Ending frame (exclusive), None for all frames
        channels : list[int] | None
            Channel indices to extract, None for all

        Returns
        -------
        np.ndarray
            Timelapse stack (n_frames, n_channels, h, w)
        """
        bboxes = self.get_bboxes(fov)
        if cell >= len(bboxes):
            raise ValueError(f"Cell {cell} not found in FOV {fov} (has {len(bboxes)} patterns)")

        bbox = bboxes[cell]
        end_frame = end_frame or self.n_frames
        channels = channels or list(range(self.n_channels))

        fov_data = self.source.get_fov(fov)
        frames = []
        for t in range(start_frame, end_frame):
            cropped = fov_data[t, channels, bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]
            frames.append(cropped)

        return np.stack(frames)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 uint8."""
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)
