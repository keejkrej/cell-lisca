"""
Cell tracking functionality using btrack (Bayesian Tracker).
"""

import numpy as np
from typing import Dict, Optional
import btrack
import btrack.config
import btrack.datasets
from skimage import measure


class CellTracker:
    """
    Tracks cells across frames using btrack's Bayesian tracking approach.

    This is a robust probabilistic tracking method that handles cell divisions,
    crowded fields, and occlusions better than simple IoU-based tracking.
    """

    def __init__(
        self,
        search_radius: float = 100.0,
        volume: Optional[tuple] = None,
        config: Optional[str] = None,
        min_area: float = 1000.0
    ):
        """
        Initialize the btrack cell tracker.

        Parameters
        ----------
        search_radius : float
            Maximum distance to search for matches between frames (in pixels)
        volume : tuple, optional
            Tracking volume as ((min_x, max_x), (min_y, max_y), (min_z, max_z))
            If None, will be set automatically from first frame
        config : str, optional
            Path to btrack config JSON file. If None, uses default cell config
        min_area : float
            Minimum area (in pixels) for a region to be considered a cell (default: 1000)
        """
        self.search_radius = search_radius
        self.volume = volume
        self.config = config
        self.min_area = min_area

        # Storage for segmentation masks and frame indices
        self.masks = []
        self.frame_indices = []
        self.next_global_id = 1

        # Will be populated after tracking
        self.tracks = None
        self._frame_to_tracking_map = {}

    def reset(self):
        """Reset tracking state for a new sequence."""
        self.masks = []
        self.frame_indices = []
        self.next_global_id = 1
        self.tracks = None
        self._frame_to_tracking_map = {}

    def add_frame(self, mask: np.ndarray, frame_idx: int):
        """
        Add a segmentation mask for a frame to be tracked.

        Parameters
        ----------
        mask : np.ndarray
            Segmentation mask with integer labels for each cell
        frame_idx : int
            Frame index (time point)
        """
        self.masks.append(mask)
        self.frame_indices.append(frame_idx)

    def track_all_frames(self) -> Dict[int, Dict[int, int]]:
        """
        Run btrack on all accumulated frames and generate tracking maps.

        Returns
        -------
        Dict[int, Dict[int, int]]
            Mapping from frame_idx to tracking_map (local_label -> global_id)
        """
        if not self.masks:
            return {}

        # Convert segmentation masks to btrack objects manually
        # We need to extract region properties for each labeled region in each frame
        objects = []

        for t, (mask, frame_idx) in enumerate(zip(self.masks, self.frame_indices)):
            # Get region properties for this frame
            regions = measure.regionprops(mask)

            for region in regions:
                # Filter out small regions (noise)
                if region.area < self.min_area:
                    continue

                # Create btrack object for this cell
                y, x = region.centroid
                area = region.area
                label = region.label

                # Create object dictionary (btrack expects this format)
                obj = {
                    't': t,  # Time index (0-based for btrack)
                    'x': x,
                    'y': y,
                    'z': 0.0,  # 2D data
                    'label': label,
                    'area': area
                }
                objects.append(obj)

        print(f"DEBUG: Created {len(objects)} objects from {len(self.masks)} frames")

        # Convert list of dicts to dict of arrays (format btrack expects)
        if objects:
            objects_dict = {
                't': np.array([obj['t'] for obj in objects]),
                'x': np.array([obj['x'] for obj in objects]),
                'y': np.array([obj['y'] for obj in objects]),
                'z': np.array([obj['z'] for obj in objects]),
                'label': np.array([obj['label'] for obj in objects]),
                'area': np.array([obj['area'] for obj in objects])
            }

            # Convert to btrack objects using the proper API
            btrack_objects = btrack.utils.objects_from_dict(objects_dict)
            print(f"DEBUG: Converted to {len(btrack_objects)} btrack objects")
        else:
            print("ERROR: No objects created!")
            return {}

        # Initialize tracker WITHOUT context manager
        # (context manager seems to close prematurely)
        tracker = btrack.BayesianTracker()

        # Configure tracker
        if self.config:
            tracker.configure(self.config)
        else:
            # Use default cell config and customize for better tracking
            default_config = btrack.datasets.cell_config()
            tracker.configure(default_config)

            # Increase max_lost to allow cells to be missing for more frames
            # Default is 5, increase to 10 for more robust tracking
            tracker.motion_model.max_lost = 10

            # Increase search radius to handle larger movements
            tracker.max_search_radius = int(self.search_radius)

        # Set volume if not provided
        if self.volume is None:
            height, width = self.masks[0].shape
            self.volume = ((0, width), (0, height))

        tracker.volume = self.volume

        # Append objects to tracker
        tracker.append(btrack_objects)
        print("DEBUG: Appended objects to tracker")

        # Run tracking
        print("DEBUG: Before track")
        tracker.track(step_size=100)
        print("DEBUG: After track")

        # Optimize tracks
        print("DEBUG: Before optimize")
        tracker.optimize()
        print("DEBUG: After optimize")

        # Store tracks and tracker (need tracker._objects for label lookup)
        self.tracks = tracker.tracks
        self._tracker = tracker
        print(f"DEBUG: Got {len(self.tracks)} tracks")

        # Build frame-by-frame tracking maps
        self._build_tracking_maps()

        return self._frame_to_tracking_map

    def _build_tracking_maps(self):
        """
        Build tracking maps from btrack results.

        Creates a mapping from (frame_idx, local_label) -> global_track_id
        """
        if not self.tracks:
            return

        # Initialize tracking maps for each frame
        for frame_idx in self.frame_indices:
            self._frame_to_tracking_map[frame_idx] = {}

        # For each track, map its objects to global track ID
        for track in self.tracks:
            global_id = track.ID

            # track.refs contains indices into tracker._objects
            # Use track.t array for frame indices and access objects via refs
            for i, ref_idx in enumerate(track.refs):
                frame_idx = int(track.t[i])

                # Get the original object to access its label
                obj = self._tracker._objects[ref_idx]
                local_label = int(obj.label)

                # Map local label to global track ID
                if frame_idx in self._frame_to_tracking_map:
                    self._frame_to_tracking_map[frame_idx][local_label] = global_id

    def get_tracking_map(self, frame_idx: int) -> Dict[int, int]:
        """
        Get tracking map for a specific frame.

        Parameters
        ----------
        frame_idx : int
            Frame index

        Returns
        -------
        Dict[int, int]
            Mapping from local labels to global track IDs
        """
        return self._frame_to_tracking_map.get(frame_idx, {})

    def create_tracked_mask(self, original_mask: np.ndarray, tracking_map: Dict[int, int]) -> np.ndarray:
        """
        Create a mask with global track IDs for consistent visualization.

        Parameters
        ----------
        original_mask : np.ndarray
            Original segmentation mask with local labels
        tracking_map : Dict[int, int]
            Mapping from local labels to global track IDs

        Returns
        -------
        np.ndarray
            Mask with global track IDs instead of local labels
        """
        tracked_mask = np.zeros_like(original_mask)
        for local_label, global_id in tracking_map.items():
            tracked_mask[original_mask == local_label] = global_id
        return tracked_mask
