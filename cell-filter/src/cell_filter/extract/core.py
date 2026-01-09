"""
Core extractor functionality for cell-filter.

This module extracts cropped image sequences based on cell count analysis,
applying user-specified criteria and outputting to H5 format.
"""

from pathlib import Path
import numpy as np
from cell_core import Cropper, CropperParameters, CellposeSegmenter
import logging
from typing import Dict, List, Any, Tuple

from cell_core.io.h5_io import (
    get_fov_bounding_boxes,
    load_analysis_h5,
    get_analysis_for_fov,
    append_extracted_sequence,
    finalize_extracted_metadata,
)

# Configure logging
logger = logging.getLogger(__name__)


class Extractor:
    """Extract sequences from analysis results based on cell count criteria."""

    # Constructor

    def __init__(
        self,
        cells_path: str,
        h5_path: str,
        nuclei_channel: int,
    ) -> None:
        """Initialize Extractor with paths and configuration.
        
        Parameters
        ----------
        cells_path : str
            Path to cells ND2 file
        h5_path : str
            Path to H5 file containing bounding boxes and analysis data
        nuclei_channel : int
            Channel index for nuclei
        """
        self.cells_path = str(Path(cells_path).resolve())
        self.h5_path = str(Path(h5_path).resolve())
        self.nuclei_channel = nuclei_channel

        # Initialize segmenter
        self.segmenter = CellposeSegmenter()
        logger.info("Cellpose segmenter initialized")

        try:
            self.cropper = Cropper(
                patterns_path=None,
                cells_path=cells_path,
                parameters=CropperParameters(nuclei_channel=nuclei_channel),
            )
            
            # Load analysis metadata to verify it exists
            self.analysis_metadata = load_analysis_h5(h5_path)
            logger.info(
                f"Initialized Extractor with {self.analysis_metadata['total_records']} analysis records"
            )
        except Exception as e:
            logger.error(f"Error initializing Extractor: {e}")
            raise ValueError(f"Error initializing Extractor: {e}")

    # Private Methods

    def _find_valid_sequences(
        self,
        cell_counts: Dict[int, int],
        n_cells: int,
        tolerance_gap: int,
        min_frames: int,
        total_frames: int,
    ) -> List[Tuple[int, int, int]]:
        """Find valid frame sequences where cell count matches criteria.
        
        Parameters
        ----------
        cell_counts : dict
            Mapping of frame_index -> cell_count for a pattern
        n_cells : int
            Target number of cells
        tolerance_gap : int
            Max consecutive frames with wrong cell count allowed
        min_frames : int
            Minimum sequence length
        total_frames : int
            Total number of frames in the timelapse
            
        Returns
        -------
        list
            List of (seq_idx, start_frame, end_frame) tuples
        """
        if not cell_counts:
            return []
        
        # Find frames where count matches
        matching_frames = sorted([
            frame for frame, count in cell_counts.items()
            if count == n_cells
        ])
        
        if not matching_frames:
            return []
        
        # Group into sequences allowing tolerance_gap gaps
        sequences = []
        seq_idx = 0
        current_start = matching_frames[0]
        current_end = matching_frames[0]
        
        for i in range(1, len(matching_frames)):
            gap = matching_frames[i] - matching_frames[i - 1] - 1
            
            if gap <= tolerance_gap:
                # Continue current sequence
                current_end = matching_frames[i]
            else:
                # Gap too large, save current sequence if long enough
                seq_length = current_end - current_start + 1
                if seq_length >= min_frames:
                    sequences.append((seq_idx, current_start, current_end))
                    seq_idx += 1
                
                # Start new sequence
                current_start = matching_frames[i]
                current_end = matching_frames[i]
        
        # Add last sequence if long enough
        seq_length = current_end - current_start + 1
        if seq_length >= min_frames:
            sequences.append((seq_idx, current_start, current_end))
        
        return sequences

    def _add_head_tail(
        self, 
        start_frame: int, 
        end_frame: int, 
        total_frames: int,
        n_frames: int = 3
    ) -> Tuple[int, int]:
        """Add head and tail frames to a sequence."""
        new_start = max(0, start_frame - n_frames)
        new_end = min(total_frames - 1, end_frame + n_frames)
        return new_start, new_end

    def _extract_sequence(
        self,
        fov_idx: int,
        pattern_idx: int,
        seq_idx: int,
        start_frame: int,
        end_frame: int,
    ) -> np.ndarray:
        """Extract and return frame stack for a sequence with segmentation.
        
        Returns
        -------
        np.ndarray
            Stack of shape (n_frames, n_channels+2, h, w)
            Channels: [pattern, cell_channels..., segmentation]
        """
        pattern = self.cropper.extract_pattern(pattern_idx)  # (h, w)

        cell_stack = []
        segmentation_stack = []

        for frame_idx in range(start_frame, end_frame + 1):
            self.cropper.load_cell(frame_idx)
            cell = self.cropper.extract_cell(pattern_idx)  # (n_channels, h, w)
            cell_stack.append(cell)

            # Segment using all channels
            cell_for_segmentation = np.transpose(cell, (1, 2, 0))  # (h, w, n_channels)
            seg_result = self.segmenter.segment_image(cell_for_segmentation)
            segmentation_stack.append(seg_result["masks"])

        n_frames = end_frame - start_frame + 1
        cell_array = np.array(cell_stack)  # (n_frames, n_channels, h, w)
        segmentation_array = np.array(segmentation_stack)  # (n_frames, h, w)

        # Add channel dimensions
        segmentation_with_channel = segmentation_array[:, np.newaxis, :, :]
        pattern_expanded = np.broadcast_to(
            pattern[np.newaxis, :, :], (n_frames, pattern.shape[0], pattern.shape[1])
        )
        pattern_with_channel = pattern_expanded[:, np.newaxis, :, :]

        # Stack: pattern, cells, segmentation
        final_stack = np.concatenate(
            [pattern_with_channel, cell_array, segmentation_with_channel], axis=1
        )

        return final_stack

    def _get_channel_names(self, n_cell_channels: int) -> List[str]:
        """Get channel names for the extracted data."""
        if hasattr(self.cropper, 'cells_channel_names') and self.cropper.cells_channel_names:
            return ["pattern"] + self.cropper.cells_channel_names + ["segmentation"]
        else:
            return ["pattern"] + [f"cell_ch_{i}" for i in range(n_cell_channels)] + ["segmentation"]

    def _process_fov(
        self,
        fov_idx: int,
        n_cells: int,
        tolerance_gap: int,
        min_frames: int,
    ) -> int:
        """Process a single FOV and extract valid sequences.
        
        Returns
        -------
        int
            Number of sequences extracted
        """
        logger.info(f"Processing FOV {fov_idx}")
        
        # Load bounding boxes
        try:
            fov_bbox_data = get_fov_bounding_boxes(self.h5_path, fov_idx)
        except Exception as e:
            logger.warning(f"Could not load bounding boxes for FOV {fov_idx}: {e}")
            return 0
        
        if not fov_bbox_data['patterns']:
            logger.warning(f"No patterns in FOV {fov_idx}")
            return 0
        
        # Load analysis data
        try:
            analysis_data = get_analysis_for_fov(self.h5_path, fov_idx)
        except Exception as e:
            logger.warning(f"Could not load analysis for FOV {fov_idx}: {e}")
            return 0
        
        if not analysis_data:
            logger.warning(f"No analysis data for FOV {fov_idx}")
            return 0
        
        # Set up cropper
        self.cropper.load_fov(fov_idx)
        
        bounding_boxes = []
        for pattern in sorted(fov_bbox_data['patterns'], key=lambda p: p['pattern_id']):
            bbox = pattern['bbox']
            bounding_boxes.append((bbox['x'], bbox['y'], bbox['width'], bbox['height']))
        
        self.cropper.bounding_boxes = bounding_boxes
        self.cropper.n_patterns = len(bounding_boxes)
        
        # Load patterns for extraction
        self.cropper.load_patterns()
        self.cropper.process_patterns()
        
        total_frames = self.cropper.n_frames
        sequences_extracted = 0
        
        # Process each pattern
        for pattern_idx, cell_counts in analysis_data.items():
            sequences = self._find_valid_sequences(
                cell_counts, n_cells, tolerance_gap, min_frames, total_frames
            )
            
            if not sequences:
                continue
            
            logger.info(f"FOV {fov_idx}, Pattern {pattern_idx}: {len(sequences)} valid sequences")
            
            for seq_idx, start_frame, end_frame in sequences:
                # Add head/tail frames
                ext_start, ext_end = self._add_head_tail(start_frame, end_frame, total_frames)
                
                try:
                    # Extract data
                    data = self._extract_sequence(fov_idx, pattern_idx, seq_idx, ext_start, ext_end)
                    
                    # Get channel names
                    n_cell_channels = data.shape[1] - 2  # subtract pattern and segmentation
                    channel_names = self._get_channel_names(n_cell_channels)
                    
                    # Prepare metadata
                    bbox = bounding_boxes[pattern_idx] if pattern_idx < len(bounding_boxes) else None
                    metadata = {
                        'start_frame': ext_start,
                        'end_frame': ext_end,
                        'channels': channel_names,
                        'bbox': bbox
                    }
                    
                    # Save to H5
                    append_extracted_sequence(
                        self.h5_path, fov_idx, pattern_idx, seq_idx, data, metadata
                    )
                    
                    sequences_extracted += 1
                    logger.debug(
                        f"Extracted FOV {fov_idx}, Pattern {pattern_idx}, Seq {seq_idx}: "
                        f"frames {ext_start}-{ext_end}"
                    )
                    
                except Exception as e:
                    logger.warning(
                        f"Error extracting FOV {fov_idx}, Pattern {pattern_idx}, Seq {seq_idx}: {e}"
                    )
                    continue
        
        return sequences_extracted

    # Public Methods

    def extract(
        self,
        n_cells: int,
        tolerance_gap: int,
        min_frames: int,
    ) -> Dict[str, Any]:
        """Extract sequences from all analyzed FOVs based on cell count criteria.
        
        Parameters
        ----------
        n_cells : int
            Target number of cells per pattern
        tolerance_gap : int
            Max consecutive frames with wrong cell count allowed within a sequence
        min_frames : int
            Minimum sequence length
            
        Returns
        -------
        dict
            Summary of extraction results
        """
        logger.info(f"Starting extraction: n_cells={n_cells}, tolerance_gap={tolerance_gap}, min_frames={min_frames}")
        
        # Get FOVs that have analysis data
        processed_fovs = self.analysis_metadata.get('processed_fovs', [])
        
        if not processed_fovs:
            logger.warning("No FOVs with analysis data found")
            return {'total_sequences': 0, 'fovs_processed': 0}
        
        total_sequences = 0
        fovs_with_sequences = []
        
        for fov_idx in processed_fovs:
            try:
                n_sequences = self._process_fov(fov_idx, n_cells, tolerance_gap, min_frames)
                total_sequences += n_sequences
                
                if n_sequences > 0:
                    fovs_with_sequences.append(fov_idx)
                    
            except Exception as e:
                logger.error(f"Error processing FOV {fov_idx}: {e}")
                continue
        
        # Finalize metadata
        finalize_extracted_metadata(self.h5_path, {
            'n_cells': n_cells,
            'tolerance_gap': tolerance_gap,
            'min_frames': min_frames,
            'cells_path': self.cells_path,
        })
        
        # Close files
        self.cropper.close_files()
        
        summary = {
            'total_sequences': total_sequences,
            'fovs_processed': len(processed_fovs),
            'fovs_with_sequences': fovs_with_sequences,
            'h5_path': self.h5_path,
        }
        
        logger.info(f"Extraction complete: {total_sequences} sequences from {len(fovs_with_sequences)} FOVs")
        
        return summary
