"""Pixel-level boundary and junction tracking for cell segmentation masks."""

import logging
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.ndimage import binary_dilation

from migrama.core.voronoi import centroids_from_mask, generate_voronoi_labels

logger = logging.getLogger(__name__)


class BoundaryPixelTracker:
    """Track pixel-level boundaries and junctions in cell segmentation masks.

    This class extracts and stores the exact pixel coordinates where cells
    meet, enabling flexible geometric and topological analysis.

    Uses dilation-based intersection to estimate boundary regions, which
    may overestimate but provides robust boundary detection.
    """

    def __init__(self, structuring_element=None):
        """Initialize the BoundaryPixelTracker.

        Parameters
        ----------
        structuring_element : np.ndarray, optional
            Structuring element for dilation. If None, uses 3x3 cross-shaped
            element (connectivity=1) which is appropriate for boundary detection.
        """
        if structuring_element is None:
            self.structuring_element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        else:
            self.structuring_element = structuring_element

    def extract_boundaries(
        self, mask: np.ndarray, include_2cell: bool = True, include_3cell: bool = True, include_4cell: bool = True
    ) -> dict[tuple[int, ...], np.ndarray]:
        """Extract all boundary pixels from a segmentation mask.

        Parameters
        ----------
        mask : np.ndarray
            2D segmentation mask with integer labels (0 = background)
        include_2cell : bool
            Extract 2-cell boundaries (edges)
        include_3cell : bool
            Extract 3-cell junctions
        include_4cell : bool
            Extract 4-cell vertices

        Returns
        -------
        dict
            Dictionary mapping sorted cell tuples to boundary pixel arrays.
            Each value is a numpy array of shape (N, 2) with (x, y) coordinates.
        """
        boundaries = {}

        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels != 0]

        if include_2cell:
            pair_boundaries = self._extract_pairwise_boundaries(mask, unique_labels)
            boundaries.update(pair_boundaries)

        if include_3cell or include_4cell:
            multi_boundaries = self._extract_multicell_junctions(mask, unique_labels, include_3cell, include_4cell)
            boundaries.update(multi_boundaries)

        return boundaries

    def _extract_pairwise_boundaries(self, mask: np.ndarray, labels: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
        """Extract 2-cell boundary pixels using dilation intersection.

        Parameters
        ----------
        mask : np.ndarray
            Segmentation mask
        labels : np.ndarray
            Unique non-zero labels

        Returns
        -------
        dict
            Dictionary mapping (cell_i, cell_j) tuples to boundary pixel arrays
        """
        boundaries = {}

        for i, j in combinations(labels, 2):
            mask_i = mask == i
            mask_j = mask == j

            dilated_i = binary_dilation(mask_i, structure=self.structuring_element)
            dilated_j = binary_dilation(mask_j, structure=self.structuring_element)

            intersection = np.logical_and(dilated_i, dilated_j)

            coords = self._boolean_to_coords(mask, intersection)
            if len(coords) > 0:
                key = tuple(sorted([int(i), int(j)]))
                boundaries[key] = coords

        return boundaries

    def _extract_multicell_junctions(
        self, mask: np.ndarray, labels: np.ndarray, include_3cell: bool, include_4cell: bool
    ) -> dict[tuple[int, ...], np.ndarray]:
        """Extract 3-cell and 4-cell junction pixels using dilation intersection.

        Parameters
        ----------
        mask : np.ndarray
            Segmentation mask
        labels : np.ndarray
            Unique non-zero labels
        include_3cell : bool
            Include 3-cell junctions
        include_4cell : bool
            Include 4-cell vertices

        Returns
        -------
        dict
            Dictionary mapping cell tuples to junction pixel arrays
        """
        boundaries = {}

        dilated_masks = {}
        for label in labels:
            dilated_masks[label] = binary_dilation(mask == label, structure=self.structuring_element)

        if include_3cell:
            for triplet in combinations(labels, 3):
                intersection = np.logical_and(
                    dilated_masks[triplet[0]], np.logical_and(dilated_masks[triplet[1]], dilated_masks[triplet[2]])
                )
                coords = self._boolean_to_coords(mask, intersection)
                if len(coords) > 0:
                    key = tuple(sorted([int(x) for x in triplet]))
                    boundaries[key] = coords

        if include_4cell:
            for quad in combinations(labels, 4):
                intersection = dilated_masks[quad[0]]
                for label in quad[1:]:
                    intersection = np.logical_and(intersection, dilated_masks[label])

                coords = self._boolean_to_coords(mask, intersection)
                if len(coords) > 0:
                    key = tuple(sorted([int(x) for x in quad]))
                    boundaries[key] = coords

        return boundaries

    def _boolean_to_coords(self, mask: np.ndarray, boolean_array: np.ndarray) -> np.ndarray:
        """Convert boolean array to (x, y) coordinate array.

        Parameters
        ----------
        mask : np.ndarray
            Original segmentation mask (used for shape validation)
        boolean_array : np.ndarray
            Boolean array where True indicates boundary pixels

        Returns
        -------
        np.ndarray
            Array of shape (N, 2) with (x, y) coordinates
        """
        rows, cols = np.where(boolean_array)
        if len(rows) == 0:
            return np.array([], dtype=np.int64).reshape(0, 2)

        coords = np.column_stack([cols, rows])
        return coords.astype(np.int64)

    def visualize_boundaries(
        self, mask: np.ndarray, boundaries: dict[tuple[int, ...], np.ndarray], figsize: tuple[int, int] = (15, 5)
    ) -> Any:
        """Visualize boundaries overlaid on segmentation mask.

        Parameters
        ----------
        mask : np.ndarray
            Segmentation mask
        boundaries : dict
            Boundary data from extract_boundaries()
        figsize : tuple
            Figure size

        Returns
        -------
        plt.Figure
            The generated figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        ax1 = axes[0]
        ax1.imshow(mask, cmap="tab20", interpolation="nearest")
        ax1.set_title("Segmentation Mask")
        ax1.axis("off")

        ax2 = axes[1]
        ax2.imshow(mask, cmap="tab20", interpolation="nearest", alpha=0.3)

        for key, coords in boundaries.items():
            if len(key) == 2:
                ax2.scatter(coords[:, 0], coords[:, 1], s=2, alpha=0.7)
        ax2.set_title("2-Cell Boundaries")
        ax2.axis("off")

        ax3 = axes[2]
        ax3.imshow(mask, cmap="tab20", interpolation="nearest", alpha=0.3)

        for key, coords in boundaries.items():
            if len(key) == 3:
                ax3.scatter(coords[:, 0], coords[:, 1], s=20, marker="o", alpha=0.8)
            elif len(key) == 4:
                ax3.scatter(coords[:, 0], coords[:, 1], s=50, marker="*", alpha=0.8)
        ax3.set_title("Junctions (3-cell: circles, 4-cell: stars)")
        ax3.axis("off")

        plt.tight_layout()
        return fig

    def get_boundary_statistics(self, boundaries: dict) -> dict:
        """Compute statistics for boundary data.

        Parameters
        ----------
        boundaries : dict
            Boundary data from extract_boundaries()

        Returns
        -------
        dict
            Statistics summary including counts and lengths
        """
        stats = {
            "n_2cell": 0,
            "n_3cell": 0,
            "n_4cell": 0,
            "total_boundary_pixels": 0,
            "edge_lengths": {},
            "junction_sizes": {},
        }

        for key, coords in boundaries.items():
            n_pixels = len(coords)
            stats["total_boundary_pixels"] += n_pixels

            if len(key) == 2:
                stats["n_2cell"] += 1
                stats["edge_lengths"][key] = n_pixels
            elif len(key) == 3:
                stats["n_3cell"] += 1
                stats["junction_sizes"][key] = n_pixels
            elif len(key) == 4:
                stats["n_4cell"] += 1
                stats["junction_sizes"][key] = n_pixels

        if stats["n_2cell"] > 0:
            stats["avg_boundary_length"] = stats["total_boundary_pixels"] / stats["n_2cell"]
        else:
            stats["avg_boundary_length"] = 0

        return stats

    def create_boundary_mask(
        self,
        mask: np.ndarray,
        boundaries: dict[tuple[int, ...], np.ndarray],
    ) -> np.ndarray:
        """Create an RGB mask with unique colors for each boundary tuple.

        Parameters
        ----------
        mask : np.ndarray
            2D segmentation mask with integer labels (0 = background)
        boundaries : dict
            Boundary data from extract_boundaries()

        Returns
        -------
        np.ndarray
            RGB image (H, W, 3) with unique colors for each boundary tuple
        """
        boundary_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

        if not boundaries:
            return boundary_mask

        n_boundaries = len(boundaries)
        cmap = (
            plt.cm.get_cmap("hsv", n_boundaries)
            if n_boundaries <= 20
            else plt.cm.get_cmap("tab20", min(20, n_boundaries))
        )
        cmap = plt.cm.get_cmap("tab20", 20)

        for idx, (_key, coords) in enumerate(boundaries.items()):
            if len(coords) == 0:
                continue
            color = np.array(cmap(idx)[:3]) * 255
            for coord in coords:
                x, y = coord[0], coord[1]
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    boundary_mask[y, x] = color.astype(np.uint8)

        return boundary_mask

    def plot_boundaries_figure(
        self,
        mask: np.ndarray,
        boundaries: dict[tuple[int, ...], np.ndarray],
        frame_idx: int = 0,
    ) -> tuple[Figure, Axes]:
        """Create a 2-panel figure with segmentation mask and boundary mask.

        Parameters
        ----------
        mask : np.ndarray
            2D segmentation mask with integer labels (0 = background)
        boundaries : dict
            Boundary data from extract_boundaries()
        frame_idx : int
            Frame index for title

        Returns
        -------
        tuple
            (figure, axes) - matplotlib Figure and array of Axes
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        ax1 = axes[0]
        ax1.imshow(mask, cmap="tab20", interpolation="nearest")
        ax1.set_title(f"Segmentation Mask (Frame {frame_idx})")
        ax1.axis("off")

        boundary_mask = self.create_boundary_mask(mask, boundaries)
        ax2 = axes[1]
        ax2.imshow(boundary_mask)
        ax2.set_title(f"Boundaries ({len(boundaries)} tuples)")
        ax2.axis("off")

        plt.tight_layout()
        return fig, axes

    def plot_4panel_figure(
        self,
        cell_mask: np.ndarray,
        nuclei_mask: np.ndarray | None,
        boundaries: dict[tuple[int, ...], np.ndarray],
        frame_idx: int = 0,
    ) -> tuple[Figure, np.ndarray]:
        """Create a 4-panel figure with cell masks, boundaries, nuclei mask, and Voronoi.

        Parameters
        ----------
        cell_mask : np.ndarray
            2D segmentation mask for cells with integer labels (0 = background)
        nuclei_mask : np.ndarray
            2D labeled mask for nuclei with integer labels (0 = background)
        boundaries : dict
            Boundary data from extract_boundaries()
        frame_idx : int
            Frame index for title

        Returns
        -------
        tuple
            (figure, axes) - matplotlib Figure and array of Axes (4 axes)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        ax1 = axes[0, 0]
        ax1.imshow(cell_mask, cmap="tab20", interpolation="nearest")
        ax1.set_title(f"Cell Segmentation (Frame {frame_idx})")
        ax1.axis("off")

        boundary_mask = self.create_boundary_mask(cell_mask, boundaries)
        ax2 = axes[0, 1]
        ax2.imshow(boundary_mask)
        ax2.set_title(f"Boundaries ({len(boundaries)} tuples)")
        ax2.axis("off")

        ax3 = axes[1, 0]
        if nuclei_mask is not None:
            ax3.imshow(nuclei_mask, cmap="viridis", interpolation="nearest")
            ax3.set_title("Nuclei Mask")
        else:
            ax3.text(0.5, 0.5, "No nuclei mask available", ha="center", va="center", transform=ax3.transAxes)
            ax3.set_title("Nuclei Mask (N/A)")
        ax3.axis("off")

        ax4 = axes[1, 1]
        if nuclei_mask is not None:
            centroids = centroids_from_mask(nuclei_mask)
            if len(centroids) > 0:
                voronoi_labels = generate_voronoi_labels(cell_mask.shape, centroids, cell_mask)
                ax4.imshow(voronoi_labels, cmap="viridis", interpolation="nearest")
                ax4.set_title(f"Nuclei Voronoi ({len(centroids)} regions)")
                for _, (cx, cy) in enumerate(centroids):
                    ax4.plot(cx, cy, "ko", markersize=3)
            else:
                ax4.text(0.5, 0.5, "No nuclei detected", ha="center", va="center", transform=ax4.transAxes)
                ax4.set_title("Nuclei Voronoi (N/A)")
        else:
            ax4.text(0.5, 0.5, "No nuclei mask available", ha="center", va="center", transform=ax4.transAxes)
            ax4.set_title("Nuclei Voronoi (N/A)")
        ax4.axis("off")

        plt.tight_layout()
        return fig, axes
