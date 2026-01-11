"""Cell graph construction from tracked segmentation masks."""

import logging
from itertools import combinations

import networkx as nx
import numpy as np
from scipy.ndimage import binary_dilation
from skimage import graph

logger = logging.getLogger(__name__)


class CellGrapher:
    """Constructs adjacency graphs from tracked cell segmentation masks."""

    def __init__(self):
        """Initialize the CellGrapher."""
        pass

    # ========================================================
    # Private Methods
    # ========================================================

    def _graph(self, mask: np.ndarray, track_ids: bool = False) -> nx.Graph:
        """Create a NetworkX graph from a segmentation mask with adjacency-weighted edges."""
        mask_copy = mask.copy()

        rag = graph.rag_mean_color(np.stack([mask_copy, mask_copy, mask_copy], axis=-1), mask_copy)
        if 0 in rag.nodes():
            rag.remove_node(0)

        regions = np.unique(mask_copy)
        logger.debug(f"Found following regions in mask: {regions}")
        regions = regions[regions != 0]

        dilated_regions: dict[int, np.ndarray] = {}
        for region in regions:
            region_mask = mask_copy == region
            dilated_regions[region] = binary_dilation(region_mask)

        for region1, region2 in combinations(regions, 2):
            if not rag.has_edge(region1, region2):
                continue
            overlap = np.logical_and(dilated_regions[region1], dilated_regions[region2])
            adjacency = np.sum(overlap) / 2
            logger.debug(f"Adjacency between region {region1} and region {region2}: {adjacency}")
            rag[region1][region2]["adjacency"] = adjacency

        # Add track ID information if available
        if track_ids:
            for node in rag.nodes():
                rag.nodes[node]["track_id"] = node

        return rag

    # ========================================================
    # Public Methods
    # ========================================================

    def graph_from_tracked_mask(self, mask: np.ndarray) -> nx.Graph:
        """Build adjacency graph from a single tracked segmentation mask."""
        return self._graph(mask, track_ids=True)

    def graph_from_tracked_masks(self, masks: list[np.ndarray]) -> list[nx.Graph]:
        """Build adjacency graphs from multiple tracked segmentation masks."""
        graphs: list[nx.Graph] = []
        for i, mask in enumerate(masks):
            logger.debug(f"Processing mask {i}")
            graph = self._graph(mask, track_ids=True)
            graphs.append(graph)
        return graphs

    def build_temporal_graphs(
        self, tracked_masks: list[np.ndarray], track_maps: list[dict[int, int]]
    ) -> list[nx.Graph]:
        """
        Build graphs for each frame with consistent track IDs across time.

        Parameters
        ----------
        tracked_masks : List[np.ndarray]
            List of masks with global track IDs
        track_maps : List[Dict[int, int]]
            List of track mappings for each frame

        Returns
        -------
        List[nx.Graph]
            List of graphs with consistent node IDs across frames
        """
        graphs = []

        for frame_idx, (mask, track_map) in enumerate(zip(tracked_masks, track_maps, strict=False)):
            g = self._graph(mask, track_ids=True)

            for node in g.nodes():
                g.nodes[node]["frame"] = frame_idx
                g.nodes[node]["local_label"] = self._find_local_label(node, track_map)

            graphs.append(g)

        return graphs

    def _find_local_label(self, track_id: int, track_map: dict[int, int]) -> int:
        """Find the local label for a given track ID in the current frame."""
        for local_label, tid in track_map.items():
            if tid == track_id:
                return local_label
        return -1
