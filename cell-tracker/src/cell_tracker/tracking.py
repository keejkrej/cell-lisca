"""
Cell tracking functionality using IoU overlap between frames.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Intersection over Union between two binary masks.
    
    Parameters
    ----------
    mask1, mask2 : np.ndarray
        Binary masks to compare
        
    Returns
    -------
    float
        IoU score between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union


class CellTracker:
    """
    Tracks cells across frames using IoU overlap to maintain consistent identities.
    """
    
    def __init__(self, iou_threshold: float = 0.3):
        """
        Initialize the cell tracker.
        
        Parameters
        ----------
        iou_threshold : float
            Minimum IoU score to consider cells as the same across frames
        """
        self.iou_threshold = iou_threshold
        self.previous_mask = None
        self.previous_tracking_map = {}
        self.next_global_id = 1
        
    def reset(self):
        """Reset tracking state for a new sequence."""
        self.previous_mask = None
        self.previous_tracking_map = {}
        self.next_global_id = 1
        
    def track_frame(self, current_mask: np.ndarray) -> Dict[int, int]:
        """
        Track cells in the current frame against the previous frame.
        
        Parameters
        ----------
        current_mask : np.ndarray
            Segmentation mask with integer labels for each cell
            
        Returns
        -------
        Dict[int, int]
            Mapping from current frame labels to global cell IDs
        """
        tracking_map = {}
        current_labels = np.unique(current_mask)[1:]  # Exclude background (0)
        
        if self.previous_mask is None:
            # First frame - assign initial global IDs
            for i, label in enumerate(current_labels):
                tracking_map[label] = self.next_global_id + i
            self.next_global_id += len(current_labels)
        else:
            # Track against previous frame
            tracking_map = self._match_cells(current_mask, current_labels)
        
        # Update state for next frame
        self.previous_mask = current_mask.copy()
        self.previous_tracking_map = tracking_map.copy()
        
        return tracking_map
    
    def _match_cells(self, current_mask: np.ndarray, current_labels: np.ndarray) -> Dict[int, int]:
        """
        Match cells between current and previous frames using IoU.
        
        Parameters
        ----------
        current_mask : np.ndarray
            Current frame segmentation mask
        current_labels : np.ndarray
            Array of current frame cell labels
            
        Returns
        -------
        Dict[int, int]
            Mapping from current labels to global IDs
        """
        tracking_map = {}
        previous_labels = np.unique(self.previous_mask)[1:]  # Exclude background (0)
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(current_labels), len(previous_labels)))
        for i, curr_label in enumerate(current_labels):
            curr_mask_binary = current_mask == curr_label
            for j, prev_label in enumerate(previous_labels):
                prev_mask_binary = self.previous_mask == prev_label
                iou_matrix[i, j] = calculate_iou(curr_mask_binary, prev_mask_binary)
        
        # Assign global IDs based on best IoU matches
        used_previous = set()
        
        for i, curr_label in enumerate(current_labels):
            best_j = np.argmax(iou_matrix[i, :])
            best_iou = iou_matrix[i, best_j]
            prev_label = previous_labels[best_j]
            
            if best_iou > self.iou_threshold and prev_label not in used_previous:
                # Good match found - use existing global ID
                tracking_map[curr_label] = self.previous_tracking_map[prev_label]
                used_previous.add(prev_label)
            else:
                # New cell or poor match - assign new global ID
                tracking_map[curr_label] = self.next_global_id
                self.next_global_id += 1
        
        return tracking_map
    
    def create_tracked_mask(self, original_mask: np.ndarray, tracking_map: Dict[int, int]) -> np.ndarray:
        """
        Create a mask with global IDs for consistent visualization.
        
        Parameters
        ----------
        original_mask : np.ndarray
            Original segmentation mask with local labels
        tracking_map : Dict[int, int]
            Mapping from local labels to global IDs
            
        Returns
        -------
        np.ndarray
            Mask with global IDs instead of local labels
        """
        tracked_mask = np.zeros_like(original_mask)
        for local_label, global_id in tracking_map.items():
            tracked_mask[original_mask == local_label] = global_id
        return tracked_mask