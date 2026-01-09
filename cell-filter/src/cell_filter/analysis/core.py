"""
Core analysis functionality for cell-filter.

This module counts cells for all patterns across all frames without filtering,
storing results in an H5 file for later processing by the extract stage.
"""

import time
from cell_filter.core import Cropper, CropperParameters, CellposeCounter
import logging
from pathlib import Path
from typing import List, Dict, Any

from cell_core.io.h5_io import (
    get_available_fovs,
    get_fov_bounding_boxes,
    append_analysis_h5,
)

# Configure logging
logger = logging.getLogger(__name__)


class Analyzer:
    """Analyze cell counts for all patterns across all frames."""

    # Constructor

    def __init__(
        self,
        cells_path: str,
        h5_path: str,
        nuclei_channel: int,
    ) -> None:
        """Initialize the Analyzer.
        
        Parameters
        ----------
        cells_path : str
            Path to cells ND2 file
        h5_path : str
            Path to H5 file containing bounding boxes (will append analysis)
        nuclei_channel : int
            Channel index for nuclei
        """
        self.cells_path = str(Path(cells_path).resolve())
        self.h5_path = str(Path(h5_path).resolve())
        self.nuclei_channel = nuclei_channel

        try:
            self._init_cropper(cells_path, nuclei_channel)
            self._init_counter()
            
            # Load available FOVs from H5
            self.available_fovs = get_available_fovs(h5_path)
            logger.info(f"H5 file contains bounding boxes for FOVs: {self.available_fovs}")
            
            logger.info(
                f"Successfully initialized Analyzer with cells: {cells_path} and H5: {h5_path}"
            )
        except Exception as e:
            logger.error(f"Error initializing Analyzer: {e}")
            raise ValueError(f"Error initializing Analyzer: {e}")

    def _init_cropper(
        self,
        cells_path: str,
        nuclei_channel: int,
    ) -> None:
        """Initialize the cell cropper."""
        try:
            self.cropper = Cropper(
                patterns_path=None,
                cells_path=cells_path,
                parameters=CropperParameters(nuclei_channel=nuclei_channel),
            )
            logger.debug(
                f"Initialized cropper with {self.cropper.n_fovs} views and {self.cropper.n_frames} frames"
            )
        except Exception as e:
            logger.error(f"Error initializing cropper: {e}")
            raise

    def _init_counter(self) -> None:
        """Initialize the cell counter."""
        try:
            self.counter = CellposeCounter()
            logger.debug("Initialized Cellpose counter")
        except Exception as e:
            logger.error(f"Error initializing counter: {e}")
            raise

    # Private Methods

    def _analyze_frame(
        self, 
        fov_idx: int, 
        frame_idx: int, 
        min_size: int = 15
    ) -> List[Dict[str, int]]:
        """Analyze a single frame and return cell counts for all patterns.
        
        Returns
        -------
        list
            List of {fov_index, pattern_id, frame_index, cell_count} records
        """
        records = []
        
        try:
            # Load current frame
            self.cropper.load_nuclei(frame_idx)

            # Collect all nuclei for this frame
            nuclei_list = []
            pattern_indices = list(range(self.cropper.n_patterns))
            valid_pattern_indices = []
            
            for pattern_idx in pattern_indices:
                try:
                    nuclei = self.cropper.extract_nuclei(pattern_idx, normalize=True)
                    nuclei_list.append(nuclei)
                    valid_pattern_indices.append(pattern_idx)
                except Exception as e:
                    logger.warning(
                        f"Error extracting nuclei for frame {frame_idx}, pattern {pattern_idx}: {e}"
                    )
                    # Record as -1 count to indicate error
                    records.append({
                        'fov_index': fov_idx,
                        'pattern_id': pattern_idx,
                        'frame_index': frame_idx,
                        'cell_count': -1
                    })
                    continue

            if not nuclei_list:
                logger.warning(f"No valid nuclei regions found in frame {frame_idx}")
                return records

            # Count nuclei for all patterns in this frame
            try:
                counts = self.counter.count_nuclei(nuclei_list, min_size=min_size)
            except Exception as e:
                logger.error(f"Error counting nuclei in frame {frame_idx}: {e}")
                # Record all as -1
                for pattern_idx in valid_pattern_indices:
                    records.append({
                        'fov_index': fov_idx,
                        'pattern_id': pattern_idx,
                        'frame_index': frame_idx,
                        'cell_count': -1
                    })
                return records

            # Store counts for all patterns
            for pattern_idx, n_count in zip(valid_pattern_indices, counts):
                records.append({
                    'fov_index': fov_idx,
                    'pattern_id': pattern_idx,
                    'frame_index': frame_idx,
                    'cell_count': n_count
                })
                logger.debug(f"FOV {fov_idx}, Pattern {pattern_idx}, Frame {frame_idx}: {n_count} cells")

            return records

        except Exception as e:
            logger.error(f"Error analyzing frame {frame_idx}: {e}")
            raise ValueError(f"Error analyzing frame {frame_idx}: {e}")

    def _analyze_fov(self, fov_idx: int, min_size: int = 15) -> List[Dict[str, int]]:
        """Analyze all frames for a single FOV.
        
        Returns
        -------
        list
            List of all analysis records for this FOV
        """
        logger.info(f"Starting analysis for FOV {fov_idx}")
        all_records = []

        try:
            # Check if FOV has bounding boxes
            if fov_idx not in self.available_fovs:
                logger.warning(f"FOV {fov_idx} not in H5 bounding boxes, skipping")
                return all_records
            
            # Load bounding boxes for this FOV
            fov_data = get_fov_bounding_boxes(self.h5_path, fov_idx)
            
            if not fov_data['patterns']:
                logger.warning(f"No patterns found for FOV {fov_idx}")
                return all_records
            
            # Load the specified FOV
            self.cropper.load_fov(fov_idx)
            
            # Set up cropper with bounding boxes from H5
            bounding_boxes = []
            for pattern in sorted(fov_data['patterns'], key=lambda p: p['pattern_id']):
                bbox = pattern['bbox']
                bounding_boxes.append((bbox['x'], bbox['y'], bbox['width'], bbox['height']))
            
            self.cropper.bounding_boxes = bounding_boxes
            self.cropper.n_patterns = len(bounding_boxes)
            
            logger.info(f"FOV {fov_idx}: {self.cropper.n_patterns} patterns, {self.cropper.n_frames} frames")

            # Process each frame
            for frame_idx in range(self.cropper.n_frames):
                if frame_idx % 10 == 0:
                    logger.info(f"FOV {fov_idx}: Processing frame {frame_idx}/{self.cropper.n_frames}")
                
                frame_records = self._analyze_frame(fov_idx, frame_idx, min_size=min_size)
                all_records.extend(frame_records)

            logger.info(f"FOV {fov_idx}: Completed with {len(all_records)} records")
            return all_records

        except Exception as e:
            logger.error(f"Error analyzing FOV {fov_idx}: {e}")
            raise ValueError(f"Error analyzing FOV {fov_idx}: {e}")

    # Public Methods

    def analyze(
        self, 
        start_fov: int, 
        end_fov: int, 
        min_size: int = 15
    ) -> Dict[str, Any]:
        """
        Analyze cell counts for a range of FOVs and append results to H5.

        Parameters
        ----------
        start_fov : int
            Starting FOV index (inclusive)
        end_fov : int
            Ending FOV index (inclusive)
        min_size : int
            Minimum cell size for Cellpose

        Returns
        -------
        dict
            Summary of analysis results
        """
        logger.info(f"Starting analysis for FOVs {start_fov} to {end_fov}")
        
        all_records: List[Dict[str, int]] = []
        processed_fovs: List[int] = []
        skipped_fovs: List[int] = []
        
        for fov_idx in range(start_fov, end_fov + 1):
            # Skip if not in available FOVs
            if fov_idx not in self.available_fovs:
                logger.warning(f"FOV {fov_idx} not found in H5 bounding boxes, skipping")
                skipped_fovs.append(fov_idx)
                continue
            
            try:
                time_start = time.time()
                fov_records = self._analyze_fov(fov_idx, min_size=min_size)
                time_elapsed = time.time() - time_start
                
                all_records.extend(fov_records)
                processed_fovs.append(fov_idx)
                
                logger.info(f"FOV {fov_idx}: {len(fov_records)} records in {time_elapsed:.1f}s")
                
            except Exception as e:
                logger.error(f"Error processing FOV {fov_idx}: {e}")
                skipped_fovs.append(fov_idx)
                continue
        
        # Save all records to H5
        if all_records:
            analysis_data = {
                'records': all_records,
                'metadata': {
                    'cells_path': self.cells_path,
                    'nuclei_channel': self.nuclei_channel,
                    'min_size': min_size,
                    'processed_fovs': processed_fovs
                }
            }
            append_analysis_h5(self.h5_path, analysis_data)
            logger.info(f"Saved {len(all_records)} records to {self.h5_path}")
        
        # Close files
        self.cropper.close_files()
        
        # Return summary
        summary = {
            'total_records': len(all_records),
            'processed_fovs': processed_fovs,
            'skipped_fovs': skipped_fovs,
            'h5_path': self.h5_path
        }
        
        logger.info(f"Analysis complete: {summary['total_records']} records from {len(processed_fovs)} FOVs")
        if skipped_fovs:
            logger.warning(f"Skipped FOVs: {skipped_fovs}")
        
        return summary
