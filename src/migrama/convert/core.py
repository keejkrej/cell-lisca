"""Convert TIFF files to H5 with segmentation and tracking."""

import logging
from pathlib import Path

import h5py
import numpy as np
import tifffile

from ..core import CellposeSegmenter, CellTracker

logger = logging.getLogger(__name__)


class Converter:
    """Convert TIFF files to H5 with segmentation and tracking."""

    def __init__(
        self,
        input_folder: str,
        output_path: str,
        nuclei_channel: int = 0,
        cell_channel: int = 1,
    ) -> None:
        """Initialize converter.

        Parameters
        ----------
        input_folder : str
            Path to folder containing TIFF files
        output_path : str
            Output H5 file path
        nuclei_channel : int
            Channel index for nuclei
        cell_channel : int
            Channel index for cell bodies
        """
        self.input_folder = Path(input_folder).resolve()
        self.output_path = Path(output_path).resolve()
        self.nuclei_channel = nuclei_channel
        self.cell_channel = cell_channel

        self.segmenter = CellposeSegmenter()

    def convert(self, min_frames: int = 1) -> int:
        """Convert TIFF files to H5.

        Parameters
        ----------
        min_frames : int
            Minimum frames required to process a sequence

        Returns
        -------
        int
            Number of sequences written
        """
        tiff_paths = sorted(self.input_folder.glob("*.tif*"))
        if not tiff_paths:
            raise FileNotFoundError(f"No TIFF files found in {self.input_folder}")

        sequences_written = 0

        with h5py.File(self.output_path, "w") as h5file:
            h5file.attrs["input_folder"] = str(self.input_folder)
            h5file.attrs["nuclei_channel"] = self.nuclei_channel
            h5file.attrs["cell_channel"] = self.cell_channel

            for cell_idx, tiff_path in enumerate(tiff_paths):
                timelapse = self._load_timelapse(tiff_path)

                n_frames = timelapse.shape[0]
                if n_frames < min_frames:
                    logger.info(f"Skipping {tiff_path.name}: only {n_frames} frames")
                    continue

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
                    cell_idx,
                    timelapse,
                    np.stack(tracked_nuclei_masks),
                    np.stack(tracked_cell_masks),
                )
                sequences_written += 1
                logger.info(f"Processed {tiff_path.name} -> cell_{cell_idx}")

        logger.info(f"Saved {sequences_written} sequences to {self.output_path}")
        return sequences_written

    def _load_timelapse(self, tiff_path: Path) -> np.ndarray:
        """Load a TIFF stack as timelapse array (t, c, y, x)."""
        with tifffile.TiffFile(tiff_path) as tif:
            data = tif.asarray()

        if data.ndim == 2:
            data = data[np.newaxis, np.newaxis, ...]
        elif data.ndim == 3:
            data = np.expand_dims(data, axis=1)
        elif data.ndim == 4:
            pass
        else:
            raise ValueError(f"Unexpected TIFF shape: {data.shape}, expected 2-4 dimensions")

        if data.shape[1] < 2:
            raise ValueError(f"TIFF must have at least 2 channels, got {data.shape[1]}")

        logger.debug(f"Loaded {tiff_path.name}: shape {data.shape}")
        return data

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
        cell_idx: int,
        timelapse: np.ndarray,
        nuclei_masks: np.ndarray,
        cell_masks: np.ndarray,
    ) -> None:
        """Write sequence data to H5."""
        fov_group = h5file.require_group("fov_0")
        cell_group = fov_group.require_group(f"cell_{cell_idx}")
        seq_group = cell_group.require_group("sequence_0")

        seq_group.create_dataset("data", data=timelapse, compression="gzip")
        seq_group.create_dataset("nuclei_masks", data=nuclei_masks, compression="gzip")
        seq_group.create_dataset("cell_masks", data=cell_masks, compression="gzip")

        n_channels = timelapse.shape[1]
        channels = [f"channel_{i}" for i in range(n_channels)]
        seq_group.create_dataset("channels", data=np.array(channels, dtype="S"))

        dummy_bbox = np.array([-1, -1, -1, -1], dtype=np.int32)
        seq_group.attrs["t0"] = -1
        seq_group.attrs["t1"] = -1
        seq_group.attrs["bbox"] = dummy_bbox
