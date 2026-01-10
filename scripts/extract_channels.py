"""
Extract specific channels from multi-channel OME.TIFF files.

Usage:
    conda activate cell-lisca
    python scripts/extract_channels.py
"""

import logging
from pathlib import Path

import tifffile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_channels(input_path: Path, output_path: Path, channels: list[int], keep_channel_dim: bool = True):
    """
    Extract specific channels from a multi-channel TIFF.

    Parameters
    ----------
    input_path : Path
        Input TIFF file path
    output_path : Path
        Output TIFF file path
    channels : list[int]
        List of channel indices to extract
    keep_channel_dim : bool
        If True, keep channel dimension even for single channel (T, C, H, W)
        If False, squeeze single channel to (T, H, W)
    """
    logger.info(f"Loading {input_path}")
    with tifffile.TiffFile(input_path) as tif:
        data = tif.asarray()

    logger.info(f"Input shape: {data.shape}, dtype: {data.dtype}")

    # Extract channels (assuming shape is T, C, H, W)
    extracted = data[:, channels, :, :]

    if not keep_channel_dim and len(channels) == 1:
        extracted = extracted.squeeze(axis=1)

    logger.info(f"Output shape: {extracted.shape}, dtype: {extracted.dtype}")

    tifffile.imwrite(output_path, extracted, imagej=True)
    logger.info(f"Saved to {output_path}")


def main():
    data_dir = Path(__file__).parent.parent / "data"
    input_path = data_dir / "4_up.ome.tif"

    # Channel mapping for 4_up.ome.tif:
    # 0: nuclear fluor
    # 1: membrane fluor
    # 2: phase contrast

    # Extract phase contrast + membrane fluor (channels 2 and 1)
    extract_channels(
        input_path,
        data_dir / "4_up_pc_mem.ome.tif",
        channels=[2, 1],  # phase contrast first, then membrane
        keep_channel_dim=True
    )

    # Extract phase contrast + nuclear fluor (channels 2 and 0)
    extract_channels(
        input_path,
        data_dir / "4_up_pc_nuc.ome.tif",
        channels=[2, 0],  # phase contrast first, then nuclear
        keep_channel_dim=True
    )

    # Extract phase contrast only (channel 2), keep channel dimension
    extract_channels(
        input_path,
        data_dir / "4_up_pc.ome.tif",
        channels=[2],
        keep_channel_dim=True
    )


if __name__ == "__main__":
    main()
