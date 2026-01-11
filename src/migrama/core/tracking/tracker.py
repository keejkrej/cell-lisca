"""Cell tracking functionality."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class CellTracker:
    """Tracks cells across frames using segmentation masks."""

    def __init__(self):
        """Initialize the cell tracker."""
        self.tracks = {}
        self.next_track_id = 1

    def track_frames(self, masks: list[np.ndarray]) -> list[dict[int, int]]:
        """
        Track cells across a sequence of frames.

        Parameters
        ----------
        masks : List[np.ndarray]
            List of segmentation masks for each frame

        Returns
        -------
        List[Dict[int, int]]
            List of tracking maps for each frame, mapping local labels to track IDs
        """
        tracking_maps = []

        if not masks:
            return tracking_maps

        # Initialize tracking with first frame
        prev_mask = masks[0]
        prev_labels = np.unique(prev_mask)
        prev_labels = prev_labels[prev_labels != 0]

        # Assign track IDs to cells in first frame
        track_map = {}
        for label in prev_labels:
            track_map[label] = self.next_track_id
            self.tracks[self.next_track_id] = [label]
            self.next_track_id += 1

        tracking_maps.append(track_map)

        # Track subsequent frames
        for frame_idx in range(1, len(masks)):
            curr_mask = masks[frame_idx]
            curr_labels = np.unique(curr_mask)
            curr_labels = curr_labels[curr_labels != 0]

            # Simple IoU-based tracking
            new_track_map = self._track_iou(prev_mask, curr_mask, tracking_maps[-1])
            tracking_maps.append(new_track_map)
            prev_mask = curr_mask

        return tracking_maps

    def _track_iou(
        self, prev_mask: np.ndarray, curr_mask: np.ndarray, prev_track_map: dict[int, int]
    ) -> dict[int, int]:
        """
        Track cells between two frames using Intersection over Union.

        Parameters
        ----------
        prev_mask : np.ndarray
            Previous frame segmentation mask
        curr_mask : np.ndarray
            Current frame segmentation mask
        prev_track_map : Dict[int, int]
            Track mapping from previous frame

        Returns
        -------
        Dict[int, int]
            New track mapping for current frame
        """
        prev_labels = np.unique(prev_mask)
        prev_labels = prev_labels[prev_labels != 0]
        curr_labels = np.unique(curr_mask)
        curr_labels = curr_labels[curr_labels != 0]

        # Calculate IoU between all pairs
        iou_matrix = np.zeros((len(prev_labels), len(curr_labels)))

        for i, prev_label in enumerate(prev_labels):
            for j, curr_label in enumerate(curr_labels):
                prev_binary = prev_mask == prev_label
                curr_binary = curr_mask == curr_label

                intersection = np.logical_and(prev_binary, curr_binary)
                union = np.logical_or(prev_binary, curr_binary)

                iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
                iou_matrix[i, j] = iou

        # Simple greedy matching based on IoU
        used_curr = set()
        new_track_map = {}

        # Sort by IoU and match
        matches = []
        for i in range(len(prev_labels)):
            for j in range(len(curr_labels)):
                if iou_matrix[i, j] > 0.1:  # Minimum IoU threshold
                    matches.append((iou_matrix[i, j], i, j))

        matches.sort(reverse=True)

        for _iou, prev_idx, curr_idx in matches:
            prev_label = prev_labels[prev_idx]
            curr_label = curr_labels[curr_idx]

            if curr_label not in used_curr:
                # Continue the track
                track_id = prev_track_map[prev_label]
                new_track_map[curr_label] = track_id
                self.tracks[track_id].append(curr_label)
                used_curr.add(curr_label)

        # Assign new track IDs to unmatched cells
        for curr_label in curr_labels:
            if curr_label not in used_curr:
                new_track_map[curr_label] = self.next_track_id
                self.tracks[self.next_track_id] = [curr_label]
                self.next_track_id += 1

        return new_track_map

    def get_tracked_mask(self, mask: np.ndarray, track_map: dict[int, int]) -> np.ndarray:
        """
        Convert a mask with local labels to tracked mask with global track IDs.

        Parameters
        ----------
        mask : np.ndarray
            Original segmentation mask with local labels
        track_map : Dict[int, int]
            Mapping from local labels to global track IDs

        Returns
        -------
        np.ndarray
            Mask with global track IDs
        """
        tracked_mask = np.zeros_like(mask)
        for local_label, track_id in track_map.items():
            tracked_mask[mask == local_label] = track_id
        return tracked_mask
