"""Voronoi tessellation utilities for nuclei mask analysis."""

import logging

import numpy as np
from matplotlib.path import Path
from scipy.spatial import Voronoi

logger = logging.getLogger(__name__)


def centroids_from_mask(mask: np.ndarray) -> np.ndarray:
    """Extract centroid coordinates from labeled mask.

    Parameters
    ----------
    mask : np.ndarray
        2D labeled mask with integer labels (0 = background)

    Returns
    -------
    np.ndarray
        Array of shape (n_labels, 2) with (x, y) centroid coordinates
    """
    labels = np.unique(mask)
    labels = labels[labels != 0]

    if len(labels) == 0:
        return np.empty((0, 2), dtype=np.float64)

    centroids = []
    for label in labels:
        coords = np.where(mask == label)
        cx = np.mean(coords[1])
        cy = np.mean(coords[0])
        centroids.append([cx, cy])

    return np.array(centroids, dtype=np.float64)


def generate_voronoi_labels(
    mask_shape: tuple[int, int],
    centroids: np.ndarray,
    cell_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Generate Voronoi tessellation labels for nuclei centroids.

    Parameters
    ----------
    mask_shape : tuple[int, int]
        Shape of the output mask (height, width)
    centroids : np.ndarray
        Array of shape (n_centroids, 2) with (x, y) centroid coordinates
    cell_mask : np.ndarray, optional
        Restrict Voronoi regions to pixels where cell_mask > 0

    Returns
    -------
    np.ndarray
        2D labeled mask with Voronoi regions (0 = background/unrestricted)
    """
    if len(centroids) == 0:
        return np.zeros(mask_shape, dtype=np.int32)

    h, w = mask_shape

    points_with_borders = np.vstack([centroids, [[-100, -100], [-100, w + 100], [h + 100, -100], [h + 100, w + 100]]])

    vor = Voronoi(points_with_borders)

    labels = np.zeros(mask_shape, dtype=np.int32)

    valid_centroid_indices = []
    for i, region_idx in enumerate(vor.point_region[: len(centroids)]):
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue
        valid_centroid_indices.append(i)

    if len(valid_centroid_indices) == 0:
        return labels

    region_map = {}
    for idx in valid_centroid_indices:
        region_idx = vor.point_region[idx]
        region = vor.regions[region_idx]
        region_map[idx] = region

    all_vertices = []
    for region in region_map.values():
        all_vertices.extend(region)
    all_vertices = list(set(all_vertices))

    vertex_coords = vor.vertices[all_vertices]

    max_x = int(np.max(vertex_coords[:, 0])) + 1 if len(vertex_coords) > 0 else w
    max_y = int(np.max(vertex_coords[:, 1])) + 1 if len(vertex_coords) > 0 else h
    min_x = int(np.min(vertex_coords[:, 0])) if len(vertex_coords) > 0 else 0
    min_y = int(np.min(vertex_coords[:, 1])) if len(vertex_coords) > 0 else 0

    x_min = max(0, min_x - 1)
    x_max = min(w, max_x + 1)
    y_min = max(0, min_y - 1)
    y_max = min(h, max_y + 1)

    for idx in valid_centroid_indices:
        region = region_map[idx]
        polygon = vor.vertices[region]

        yi, xi = np.ogrid[y_min:y_max, x_min:x_max]
        xi = xi + x_min
        yi = yi + y_min

        path = plt_polygon_path(polygon)
        mask_in_patch = path.contains_points(np.column_stack([xi.ravel(), yi.ravel()]))
        mask_in_patch = mask_in_patch.reshape(xi.shape)

        labels[yi[mask_in_patch], xi[mask_in_patch]] = idx + 1

    if cell_mask is not None:
        labels[cell_mask == 0] = 0

    return labels


def plt_polygon_path(polygon: np.ndarray):
    """Create a matplotlib Path from polygon vertices.

    Parameters
    ----------
    polygon : np.ndarray
        Array of shape (n_vertices, 2) with (x, y) coordinates

    Returns
    -------
    matplotlib.path.Path
        Path object for the polygon
    """
    codes = [Path.LINETO] * len(polygon)
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    return Path(polygon, codes=codes)
