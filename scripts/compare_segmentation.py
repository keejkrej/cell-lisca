"""
Compare cellpose segmentation across different channel combinations:
- All 3 channels (nuclear + membrane + phase contrast)
- 2 channels (phase contrast + membrane)
- 2 channels (phase contrast + nuclear)
- 1 channel (phase contrast only)

Usage:
    conda activate cell-lisca
    python scripts/compare_segmentation.py
"""

import numpy as np
import tifffile
from cellpose import models
from pathlib import Path
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def segment_timelapse(data: np.ndarray, model: models.CellposeModel) -> np.ndarray:
    """
    Segment all frames in a timelapse.

    Parameters
    ----------
    data : np.ndarray
        Timelapse data with shape (frames, height, width) or (frames, channels, height, width)
    model : models.CellposeModel
        Cellpose model instance

    Returns
    -------
    np.ndarray
        Segmentation masks with shape (frames, height, width)
    """
    n_frames = data.shape[0]
    
    # Determine image shape
    if data.ndim == 3:
        height, width = data.shape[1], data.shape[2]
    else:
        height, width = data.shape[2], data.shape[3]
    
    masks = np.zeros((n_frames, height, width), dtype=np.int16)
    
    for frame_idx in range(n_frames):
        if frame_idx % 10 == 0:
            logger.info(f"Processing frame {frame_idx}/{n_frames}")
        
        # Get frame data
        if data.ndim == 3:
            frame = data[frame_idx]
        else:
            # Multi-channel: transpose to (height, width, channels) for cellpose
            frame = np.transpose(data[frame_idx], (1, 2, 0))
        
        # Run segmentation
        result = model.eval(frame)
        masks[frame_idx] = result[0].astype(np.int16)
    
    return masks


def main():
    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "data"
    
    multichannel_path = data_dir / "4_up.ome.tif"
    pc_mem_path = data_dir / "4_up_pc_mem.ome.tif"
    pc_nuc_path = data_dir / "4_up_pc_nuc.ome.tif"
    phase_contrast_path = data_dir / "4_up_pc.ome.tif"
    
    # Output paths (TIFF for ImageJ viewing)
    multichannel_mask_path = output_dir / "4_up_seg.tif"
    pc_mem_mask_path = output_dir / "4_up_pc_mem_seg.tif"
    pc_nuc_mask_path = output_dir / "4_up_pc_nuc_seg.tif"
    phase_contrast_mask_path = output_dir / "4_up_pc_seg.tif"
    
    # Initialize cellpose model (SAM-based in cellpose 4.x)
    logger.info("Initializing Cellpose model with GPU...")
    model = models.CellposeModel(gpu=True)
    
    # Load and segment multi-channel data
    logger.info(f"Loading multi-channel data from {multichannel_path}")
    with tifffile.TiffFile(multichannel_path) as tif:
        multichannel_data = tif.asarray()
    logger.info(f"Multi-channel data shape: {multichannel_data.shape}")
    
    logger.info("Segmenting multi-channel timelapse (all channels)...")
    multichannel_masks = segment_timelapse(multichannel_data, model)
    tifffile.imwrite(multichannel_mask_path, multichannel_masks.astype(np.int16), imagej=True)
    logger.info(f"Saved multi-channel masks to {multichannel_mask_path}")
    
    # Load and segment phase contrast + membrane data
    logger.info(f"Loading PC + membrane data from {pc_mem_path}")
    with tifffile.TiffFile(pc_mem_path) as tif:
        pc_mem_data = tif.asarray()
    logger.info(f"PC + membrane data shape: {pc_mem_data.shape}")
    
    logger.info("Segmenting PC + membrane timelapse (2 channels)...")
    pc_mem_masks = segment_timelapse(pc_mem_data, model)
    tifffile.imwrite(pc_mem_mask_path, pc_mem_masks.astype(np.int16), imagej=True)
    logger.info(f"Saved PC + membrane masks to {pc_mem_mask_path}")
    
    # Load and segment phase contrast + nuclear data
    logger.info(f"Loading PC + nuclear data from {pc_nuc_path}")
    with tifffile.TiffFile(pc_nuc_path) as tif:
        pc_nuc_data = tif.asarray()
    logger.info(f"PC + nuclear data shape: {pc_nuc_data.shape}")
    
    logger.info("Segmenting PC + nuclear timelapse (2 channels)...")
    pc_nuc_masks = segment_timelapse(pc_nuc_data, model)
    tifffile.imwrite(pc_nuc_mask_path, pc_nuc_masks.astype(np.int16), imagej=True)
    logger.info(f"Saved PC + nuclear masks to {pc_nuc_mask_path}")
    
    # Load and segment phase contrast only data
    logger.info(f"Loading phase contrast data from {phase_contrast_path}")
    with tifffile.TiffFile(phase_contrast_path) as tif:
        phase_contrast_data = tif.asarray()
    logger.info(f"Phase contrast data shape: {phase_contrast_data.shape}")
    
    logger.info("Segmenting phase contrast timelapse (single channel)...")
    phase_contrast_masks = segment_timelapse(phase_contrast_data, model)
    tifffile.imwrite(phase_contrast_mask_path, phase_contrast_masks.astype(np.int16), imagej=True)
    logger.info(f"Saved phase contrast masks to {phase_contrast_mask_path}")
    
    # Summary statistics and detailed report
    logger.info("\n=== Segmentation Summary ===")
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("CELLPOSE SEGMENTATION COMPARISON REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    all_results = [
        ("All channels (3)", multichannel_masks),
        ("PC + membrane (2)", pc_mem_masks),
        ("PC + nuclear (2)", pc_nuc_masks),
        ("Phase contrast (1)", phase_contrast_masks)
    ]
    
    for name, masks in all_results:
        cell_counts = [len(np.unique(m)) - 1 for m in masks]  # -1 for background
        
        # Basic stats
        avg_cells = np.mean(cell_counts)
        min_cells = np.min(cell_counts)
        max_cells = np.max(cell_counts)
        
        logger.info(f"{name}: avg cells/frame = {avg_cells:.1f}, "
                   f"min = {min_cells}, max = {max_cells}")
        
        # Detailed distribution
        report_lines.append(f"--- {name} ---")
        report_lines.append(f"Average cells/frame: {avg_cells:.2f}")
        report_lines.append(f"Min: {min_cells}, Max: {max_cells}")
        report_lines.append("")
        report_lines.append("Cell count distribution:")
        
        # Count frames for each cell count
        count_dist = Counter(cell_counts)
        for n_cells in sorted(count_dist.keys()):
            n_frames = count_dist[n_cells]
            pct = 100 * n_frames / len(cell_counts)
            report_lines.append(f"  {n_cells} cells: {n_frames} frames ({pct:.1f}%)")
        
        report_lines.append("")
    
    # Save report
    report_path = output_dir / "segmentation_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    logger.info(f"Saved detailed report to {report_path}")


if __name__ == "__main__":
    main()
