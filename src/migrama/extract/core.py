"""Sequence extraction with segmentation and tracking."""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from ..core import CellposeSegmenter, CellTracker
from ..core.pattern import CellCropper

logger = logging.getLogger(__name__)


@dataclass
class AnalysisRow:
    """Row from analysis CSV."""

    cell: int
    fov: int
    x: int
    y: int
    w: int
    h: int
    t0: int
    t1: int


class Extractor:
    """Extract sequences with segmentation and tracking."""

    # Constructor

    def __init__(
        self,
        cells_path: str,
        analysis_csv: str,
        output_path: str,
        nuclei_channel: int = 1,
        cell_channel: int = 0,
    ) -> None:
        """Initialize extractor.

        Parameters
        ----------
        cells_path : str
            Path to cells ND2 file
        analysis_csv : str
            Path to analysis CSV file
        output_path : str
            Output H5 file path
        nuclei_channel : int
            Channel index for nuclei
        cell_channel : int
            Channel index for cell bodies
        """
        self.cells_path = Path(cells_path).resolve()
        self.analysis_csv = Path(analysis_csv).resolve()
        self.output_path = Path(output_path).resolve()
        self.nuclei_channel = nuclei_channel
        self.cell_channel = cell_channel

        self.cropper = CellCropper(
            cells_path=str(self.cells_path),
            bboxes_csv=str(self.analysis_csv),
            nuclei_channel=nuclei_channel,
        )
        self.segmenter = CellposeSegmenter()

    def extract(self, min_frames: int = 1) -> int:
        """Extract sequences to H5.

        Parameters
        ----------
        min_frames : int
            Minimum frames required to extract a sequence

        Returns
        -------
        int
            Number of sequences extracted
        """
        rows = self._load_analysis_rows(self.analysis_csv)
        sequences_written = 0

        with h5py.File(self.output_path, "w") as h5file:
            h5file.attrs["cells_path"] = str(self.cells_path)
            h5file.attrs["nuclei_channel"] = self.nuclei_channel
            h5file.attrs["cell_channel"] = self.cell_channel

            for row in rows:
                if row.t0 < 0 or row.t1 < row.t0:
                    continue

                n_frames = row.t1 - row.t0 + 1
                if n_frames < min_frames:
                    continue

                timelapse = self.cropper.extract_timelapse(
                    row.fov,
                    row.cell,
                    start_frame=row.t0,
                    end_frame=row.t1 + 1,
                    channels=None,
                )

                nuclei_masks = self._segment_channel(timelapse, self.nuclei_channel)
                cell_masks = self._segment_channel(timelapse, self.cell_channel)

                tracker = CellTracker()
                tracking_maps = tracker.track_frames(nuclei_masks)
                tracked_nuclei_masks = [
                    tracker.get_tracked_mask(mask, track_map)
                    for mask, track_map in zip(nuclei_masks, tracking_maps, strict=False)
                ]

                tracked_cell_masks = [
                    self._map_cells_to_tracks(cell_mask, tracked_nuclei_mask)
                    for cell_mask, tracked_nuclei_mask in zip(cell_masks, tracked_nuclei_masks, strict=False)
                ]

                self._write_sequence(
                    h5file,
                    row,
                    timelapse,
                    np.stack(tracked_nuclei_masks),
                    np.stack(tracked_cell_masks),
                )
                sequences_written += 1

        logger.info(f"Saved {sequences_written} sequences to {self.output_path}")
        return sequences_written

    # Private Methods

    @staticmethod
    def _load_analysis_rows(csv_path: Path) -> list[AnalysisRow]:
        """Load analysis CSV rows."""
        rows: list[AnalysisRow] = []
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rows.append(
                    AnalysisRow(
                        cell=int(row["cell"]),
                        fov=int(row["fov"]),
                        x=int(row["x"]),
                        y=int(row["y"]),
                        w=int(row["w"]),
                        h=int(row["h"]),
                        t0=int(row["t0"]),
                        t1=int(row["t1"]),
                    )
                )
        return rows

    def _segment_channel(self, timelapse: np.ndarray, channel_idx: int) -> list[np.ndarray]:
        """Segment a single channel across frames."""
        masks = []
        for frame_idx in range(timelapse.shape[0]):
            image = timelapse[frame_idx, channel_idx]
            result = self.segmenter.segment_image(image)
            masks.append(result["masks"])
        return masks

    @staticmethod
    def _map_cells_to_tracks(cell_mask: np.ndarray, tracked_nuclei_mask: np.ndarray) -> np.ndarray:
        """Assign cell labels to tracked nuclei IDs by overlap."""
        tracked_cells = np.zeros_like(cell_mask, dtype=np.int32)
        cell_labels = np.unique(cell_mask)
        cell_labels = cell_labels[cell_labels != 0]

        for cell_label in cell_labels:
            overlap_ids = tracked_nuclei_mask[cell_mask == cell_label]
            overlap_ids = overlap_ids[overlap_ids != 0]
            if overlap_ids.size == 0:
                continue
            unique_ids, counts = np.unique(overlap_ids, return_counts=True)
            track_id = int(unique_ids[np.argmax(counts)])
            tracked_cells[cell_mask == cell_label] = track_id

        return tracked_cells

    def _write_sequence(
        self,
        h5file: h5py.File,
        row: AnalysisRow,
        timelapse: np.ndarray,
        nuclei_masks: np.ndarray,
        cell_masks: np.ndarray,
    ) -> None:
        """Write sequence data to H5."""
        fov_group = h5file.require_group(f"fov_{row.fov}")
        cell_group = fov_group.require_group(f"cell_{row.cell}")
        seq_group = cell_group.require_group("sequence_0")

        seq_group.create_dataset("data", data=timelapse, compression="gzip")
        seq_group.create_dataset("nuclei_masks", data=nuclei_masks, compression="gzip")
        seq_group.create_dataset("cell_masks", data=cell_masks, compression="gzip")

        channels = self.cropper.channel_names or [f"channel_{i}" for i in range(timelapse.shape[1])]
        seq_group.create_dataset("channels", data=np.array(channels, dtype="S"))

        seq_group.attrs["t0"] = row.t0
        seq_group.attrs["t1"] = row.t1
        seq_group.attrs["bbox"] = np.array([row.x, row.y, row.w, row.h], dtype=np.int32)
