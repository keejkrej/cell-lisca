"""H5 data loader for graph analysis."""

import logging
from pathlib import Path

import h5py
import yaml

logger = logging.getLogger(__name__)


class H5SegmentationLoader:
    """Load segmentation masks from migrama extracted H5 files."""

    def __init__(self) -> None:
        """Initialize the loader."""
        return

    def load_cell_filter_data(
        self,
        h5_path: str,
        fov_idx: int,
        pattern_idx: int,
        seq_idx: int,
        yaml_path: str | None = None,
    ) -> dict[str, object]:
        """Load extracted data and segmentation masks from H5."""
        group_path = f"fov_{fov_idx}/cell_{pattern_idx}/sequence_{seq_idx}"

        with h5py.File(h5_path, "r") as h5file:
            if group_path not in h5file:
                raise ValueError(f"Sequence not found: {group_path}")
            group = h5file[group_path]

            data = group["data"][...]
            segmentation_masks = group["cell_masks"][...]

            channels = None
            if "channels" in group:
                channels = [c.decode("utf-8") for c in group["channels"][...]]

            metadata = {
                "t0": int(group.attrs.get("t0", -1)),
                "t1": int(group.attrs.get("t1", -1)),
                "bbox": group.attrs.get("bbox", None),
            }

        if yaml_path is None:
            yaml_path = str(Path(h5_path).with_suffix(".yaml"))

        yaml_metadata = None
        if yaml_path and Path(yaml_path).exists():
            with open(yaml_path) as handle:
                yaml_metadata = yaml.safe_load(handle)

        return {
            "data": data,
            "segmentation_masks": segmentation_masks,
            "metadata": yaml_metadata,
            "channels": channels,
            "sequence_metadata": metadata,
        }

    def list_sequences(self, h5_path: str) -> list[dict[str, int]]:
        """List available sequences in the H5 file."""
        sequences: list[dict[str, int]] = []
        with h5py.File(h5_path, "r") as h5file:
            for fov_key in h5file.keys():
                if not fov_key.startswith("fov_"):
                    continue
                fov_idx = int(fov_key.split("_")[1])
                for cell_key in h5file[fov_key].keys():
                    if not cell_key.startswith("cell_"):
                        continue
                    cell_idx = int(cell_key.split("_")[1])
                    for seq_key in h5file[fov_key][cell_key].keys():
                        if not seq_key.startswith("sequence_"):
                            continue
                        seq_idx = int(seq_key.split("_")[1])
                        sequences.append(
                            {
                                "fov_idx": fov_idx,
                                "pattern_idx": cell_idx,
                                "seq_idx": seq_idx,
                            }
                        )
        return sequences

    def validate_cell_filter_output(self, h5_path: str, yaml_path: str | None = None) -> bool:
        """Validate that the H5 file contains valid extracted data."""
        try:
            if not Path(h5_path).exists():
                logger.error(f"H5 file not found: {h5_path}")
                return False

            sequences = self.list_sequences(h5_path)
            if not sequences:
                logger.error("No sequences found in H5 file")
                return False

            first_seq = sequences[0]
            data = self.load_cell_filter_data(
                h5_path,
                first_seq["fov_idx"],
                first_seq["pattern_idx"],
                first_seq["seq_idx"],
                yaml_path,
            )["data"]

            if data.ndim != 4:
                logger.error(f"Expected 4D data, got shape {data.shape}")
                return False

            if data.shape[1] < 1:
                logger.error("Expected at least one image channel")
                return False

            return True
        except Exception as exc:
            logger.error(f"H5 file validation failed: {exc}")
            return False
