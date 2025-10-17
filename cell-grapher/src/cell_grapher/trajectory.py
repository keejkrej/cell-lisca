"""
Cell trajectory tracking for nucleus and center of mass analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import ndimage


class TrajectoryAnalyzer:
    """
    Tracks nucleus positions and center of mass for each tracked cell over time.
    """

    def __init__(self, nucleus_threshold: float = 0.0):
        """
        Initialize the trajectory analyzer.

        Parameters
        ----------
        nucleus_threshold : float
            Threshold value for nucleus detection (default: 0.0, no threshold)
        """
        self.nucleus_threshold = nucleus_threshold

        # Storage for trajectory data
        # Format: {global_id: {'frames': [], 'nucleus_positions': [], 'cytoplasm_positions': []}}
        self.trajectories = {}

    def reset(self):
        """Reset all trajectory data."""
        self.trajectories = {}

    def analyze_frame(
        self,
        frame_idx: int,
        segmentation_mask: np.ndarray,
        tracking_map: Dict[int, int],
        nucleus_channel: np.ndarray
    ) -> Dict[int, Dict[str, Tuple[float, float]]]:
        """
        Analyze nucleus and center of mass positions for all cells in a frame.

        Parameters
        ----------
        frame_idx : int
            Frame index
        segmentation_mask : np.ndarray
            Segmentation mask with local labels
        tracking_map : Dict[int, int]
            Mapping from local labels to global track IDs
        nucleus_channel : np.ndarray
            Nucleus fluorescence channel image

        Returns
        -------
        Dict[int, Dict[str, Tuple[float, float]]]
            Dictionary mapping global_id to positions:
            {global_id: {
                'nucleus_pos': (y, x),
                'cytoplasm_pos': (y, x),
                'nucleus_intensity': float
            }}
        """
        frame_results = {}

        for local_label, global_id in tracking_map.items():
            # Get binary mask for this cell
            cell_mask = (segmentation_mask == local_label)

            if not cell_mask.any():
                continue

            # Calculate center of mass from segmentation mask
            cytoplasm_y, cytoplasm_x = ndimage.center_of_mass(cell_mask)

            # Extract nucleus within this cell
            nucleus_in_cell = nucleus_channel * cell_mask

            # Apply threshold if specified
            if self.nucleus_threshold > 0:
                nucleus_in_cell = nucleus_in_cell * (nucleus_in_cell > self.nucleus_threshold)

            # Calculate nucleus position (weighted centroid by intensity)
            if nucleus_in_cell.sum() > 0:
                nucleus_y, nucleus_x = ndimage.center_of_mass(nucleus_in_cell)
                nucleus_intensity = nucleus_in_cell.sum()
            else:
                # If no nucleus detected, use center of mass as fallback
                nucleus_y, nucleus_x = cytoplasm_y, cytoplasm_x
                nucleus_intensity = 0.0

            # Store results
            frame_results[global_id] = {
                'nucleus_pos': (nucleus_y, nucleus_x),
                'cytoplasm_pos': (cytoplasm_y, cytoplasm_x),
                'nucleus_intensity': nucleus_intensity
            }

            # Update trajectory storage
            if global_id not in self.trajectories:
                self.trajectories[global_id] = {
                    'frames': [],
                    'nucleus_positions': [],
                    'cytoplasm_positions': [],
                    'nucleus_intensities': []
                }

            self.trajectories[global_id]['frames'].append(frame_idx)
            self.trajectories[global_id]['nucleus_positions'].append((nucleus_y, nucleus_x))
            self.trajectories[global_id]['cytoplasm_positions'].append((cytoplasm_y, cytoplasm_x))
            self.trajectories[global_id]['nucleus_intensities'].append(nucleus_intensity)

        return frame_results

    def get_trajectory(self, global_id: int) -> Optional[Dict]:
        """
        Get trajectory data for a specific cell.

        Parameters
        ----------
        global_id : int
            Global track ID

        Returns
        -------
        Optional[Dict]
            Trajectory data or None if not found
        """
        return self.trajectories.get(global_id)

    def get_all_trajectories(self) -> Dict[int, Dict]:
        """
        Get all trajectory data.

        Returns
        -------
        Dict[int, Dict]
            All trajectories indexed by global_id
        """
        return self.trajectories

    def calculate_displacement(
        self,
        global_id: int,
        use_nucleus: bool = True
    ) -> Optional[np.ndarray]:
        """
        Calculate frame-to-frame displacement for a cell.

        Parameters
        ----------
        global_id : int
            Global track ID
        use_nucleus : bool
            If True, use nucleus position; if False, use center of mass

        Returns
        -------
        Optional[np.ndarray]
            Array of displacements (N-1, 2) or None if not found
        """
        if global_id not in self.trajectories:
            return None

        positions = (self.trajectories[global_id]['nucleus_positions']
                     if use_nucleus
                     else self.trajectories[global_id]['cytoplasm_positions'])

        if len(positions) < 2:
            return None

        positions = np.array(positions)
        displacements = np.diff(positions, axis=0)

        return displacements

    def calculate_velocity(
        self,
        global_id: int,
        use_nucleus: bool = True,
        time_per_frame: float = 1.0
    ) -> Optional[np.ndarray]:
        """
        Calculate velocity for a cell.

        Parameters
        ----------
        global_id : int
            Global track ID
        use_nucleus : bool
            If True, use nucleus position; if False, use center of mass
        time_per_frame : float
            Time interval between frames (for velocity calculation)

        Returns
        -------
        Optional[np.ndarray]
            Array of velocities (N-1, 2) or None if not found
        """
        displacements = self.calculate_displacement(global_id, use_nucleus)

        if displacements is None:
            return None

        velocities = displacements / time_per_frame
        return velocities

    def export_trajectories_to_dict(self) -> List[Dict]:
        """
        Export all trajectories to a list of dictionaries (for CSV export).

        Returns
        -------
        List[Dict]
            List of trajectory points with format:
            [{
                'global_id': int,
                'frame': int,
                'nucleus_y': float,
                'nucleus_x': float,
                'cytoplasm_y': float,
                'cytoplasm_x': float,
                'nucleus_intensity': float
            }, ...]
        """
        export_data = []

        for global_id, traj_data in self.trajectories.items():
            for i, frame in enumerate(traj_data['frames']):
                nucleus_y, nucleus_x = traj_data['nucleus_positions'][i]
                cytoplasm_y, cytoplasm_x = traj_data['cytoplasm_positions'][i]
                nucleus_intensity = traj_data['nucleus_intensities'][i]

                export_data.append({
                    'global_id': global_id,
                    'frame': frame,
                    'nucleus_y': nucleus_y,
                    'nucleus_x': nucleus_x,
                    'cytoplasm_y': cytoplasm_y,
                    'cytoplasm_x': cytoplasm_x,
                    'nucleus_intensity': nucleus_intensity
                })

        return export_data

    def get_trajectory_summary(self) -> Dict[str, any]:
        """
        Get summary statistics for all trajectories.

        Returns
        -------
        Dict[str, any]
            Summary statistics
        """
        num_tracks = len(self.trajectories)
        track_lengths = [len(traj['frames']) for traj in self.trajectories.values()]

        if not track_lengths:
            return {
                'num_tracks': 0,
                'mean_track_length': 0,
                'max_track_length': 0,
                'min_track_length': 0
            }

        return {
            'num_tracks': num_tracks,
            'mean_track_length': np.mean(track_lengths),
            'max_track_length': np.max(track_lengths),
            'min_track_length': np.min(track_lengths)
        }
