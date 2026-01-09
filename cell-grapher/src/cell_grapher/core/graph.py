"""Cell segmentation and graph construction using Cellpose models."""

from cellpose import models
import numpy as np
import networkx as nx
from skimage import graph
from scipy.ndimage import binary_dilation
from itertools import combinations
import logging

logger = logging.getLogger(__name__)

class CellGrapher:
    """Performs cell segmentation and constructs adjacency graphs from microscopy images."""
    
    # ========================================================
    # Constructor
    # ========================================================
    
    def __init__(self,
            use_gpu: bool = True,
        ):
        """Initialize the Cellpose model with optional GPU acceleration."""
        self.model = models.CellposeModel(
            gpu=use_gpu,
        )

    # ========================================================
    # Private Methods
    # ========================================================
    
    def _segment(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Segment cells in images using the Cellpose model and return masks."""
        masks, _, _, _ = self.model.eval(images)
        return masks

    def _graph(self, mask: np.ndarray) -> nx.Graph:
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
            region_mask = (mask_copy == region)
            dilated_regions[region] = binary_dilation(region_mask)
        
        for region1, region2 in combinations(regions, 2):
            if not rag.has_edge(region1, region2):
                continue
            overlap = np.logical_and(dilated_regions[region1], dilated_regions[region2])
            adjacency = np.sum(overlap) / 2
            logger.debug(f"Adjacency between region {region1} and region {region2}: {adjacency}")
            rag[region1][region2]['adjacency'] = adjacency
        
        return rag
    
    # ========================================================
    # Public Methods
    # ========================================================
    
    def graph_cells(self, images: list[np.ndarray]) -> list[nx.Graph]:
        """Segment cells in images and return corresponding adjacency graphs."""
        masks = self._segment(images)
        graphs: list[nx.Graph] = []
        for i, mask in enumerate(masks):
            logger.debug(f"Processing mask {i}")
            graph = self._graph(mask)
            graphs.append(graph)
        return graphs



        