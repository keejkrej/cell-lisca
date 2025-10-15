import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from skimage import graph as ski_graph
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import colors


class AdjacencyGraphBuilder:
    """
    A class for building weighted adjacency graphs from cell segmentation masks.
    
    This class calculates region adjacency graphs (RAG) where edge weights
    represent the length of contact boundaries between adjacent cells.
    """
    
    def __init__(self, method: str = 'boundary_length'):
        """
        Initialize the AdjacencyGraphBuilder.
        
        Parameters
        ----------
        method : str
            Method for calculating edge weights:
            - 'boundary_length': Count pixels along cell boundaries
            - 'overlap_area': Use dilation-based overlap (faster but approximate)
        """
        self.method = method
        
    def build_graph(self, mask: np.ndarray) -> nx.Graph:
        """
        Build a weighted adjacency graph from a segmentation mask.
        
        Parameters
        ----------
        mask : np.ndarray
            2D segmentation mask where each cell has a unique integer label
            
        Returns
        -------
        nx.Graph
            NetworkX graph with nodes as cell labels and weighted edges
        """
        if self.method == 'boundary_length':
            return self._build_boundary_length_graph(mask)
        elif self.method == 'overlap_area':
            return self._build_overlap_area_graph(mask)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _build_boundary_length_graph(self, mask: np.ndarray) -> nx.Graph:
        """
        Build graph with edge weights as actual boundary lengths.
        
        This method counts pixels where two regions meet.
        """
        rag = ski_graph.RAG(mask)
        
        # Remove background node (label 0) if it exists
        if 0 in rag.nodes():
            rag.remove_node(0)
        
        for edge in rag.edges():
            region1, region2 = edge
            
            mask1 = mask == region1
            mask2 = mask == region2
            
            boundary_length = 0
            
            # Check horizontal adjacencies
            horizontal_adj = np.logical_and(
                mask1[:, :-1],
                mask2[:, 1:]
            )
            boundary_length += np.sum(horizontal_adj)
            
            horizontal_adj_rev = np.logical_and(
                mask2[:, :-1],
                mask1[:, 1:]
            )
            boundary_length += np.sum(horizontal_adj_rev)
            
            # Check vertical adjacencies
            vertical_adj = np.logical_and(
                mask1[:-1, :],
                mask2[1:, :]
            )
            boundary_length += np.sum(vertical_adj)
            
            vertical_adj_rev = np.logical_and(
                mask2[:-1, :],
                mask1[1:, :]
            )
            boundary_length += np.sum(vertical_adj_rev)
            
            rag[region1][region2]['weight'] = boundary_length
            rag[region1][region2]['boundary_length'] = boundary_length
        
        return rag
    
    def _build_overlap_area_graph(self, mask: np.ndarray) -> nx.Graph:
        """
        Build graph using dilation-based overlap area method.
        
        This is faster but provides an approximation of contact area.
        """
        G = nx.Graph()
        
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background
        
        G.add_nodes_from(unique_labels)
        
        for label in unique_labels:
            region_mask = mask == label
            dilated = ndimage.binary_dilation(region_mask)
            
            neighbors = np.unique(mask[dilated])
            neighbors = neighbors[(neighbors != 0) & (neighbors != label)]
            
            for neighbor in neighbors:
                if not G.has_edge(label, neighbor):
                    neighbor_mask = mask == neighbor
                    neighbor_dilated = ndimage.binary_dilation(neighbor_mask)
                    
                    overlap = np.logical_and(dilated, neighbor_dilated)
                    weight = np.sum(overlap) / 2
                    
                    G.add_edge(label, neighbor, weight=weight, overlap_area=weight)
        
        return G
    
    def analyze_four_cell_cluster(
        self,
        graph: nx.Graph,
        cells: List[int]
    ) -> Dict[str, Union[float, bool]]:
        """
        Analyze a four-cell cluster for T1 transition monitoring.
        
        Parameters
        ----------
        graph : nx.Graph
            The adjacency graph
        cells : list of int
            List of 4 cell labels [a, b, c, d]
            
        Returns
        -------
        dict
            Analysis results including:
            - 'ac_contact': Length/weight of edge between cells a and c
            - 'bd_contact': Length/weight of edge between cells b and d
            - 'is_t1_configuration': Whether this is a T1 configuration
        """
        if len(cells) != 4:
            raise ValueError("Exactly 4 cells required for T1 analysis")
        
        a, b, c, d = cells
        
        ac_contact = graph.get_edge_data(a, c, default={'weight': 0})['weight']
        bd_contact = graph.get_edge_data(b, d, default={'weight': 0})['weight']
        
        is_t1 = (ac_contact > 0 and bd_contact == 0) or (ac_contact == 0 and bd_contact > 0)
        
        return {
            'ac_contact': ac_contact,
            'bd_contact': bd_contact,
            'is_t1_configuration': is_t1,
            'ab_contact': graph.get_edge_data(a, b, default={'weight': 0})['weight'],
            'bc_contact': graph.get_edge_data(b, c, default={'weight': 0})['weight'],
            'cd_contact': graph.get_edge_data(c, d, default={'weight': 0})['weight'],
            'da_contact': graph.get_edge_data(d, a, default={'weight': 0})['weight']
        }
    
    def visualize_graph(
        self,
        graph: nx.Graph,
        mask: np.ndarray,
        figsize: Tuple[int, int] = (15, 5),
        highlight_edges: Optional[List[Tuple[int, int]]] = None
    ) -> plt.Figure:
        """
        Visualize the segmentation mask and adjacency graph.
        
        Parameters
        ----------
        graph : nx.Graph
            The adjacency graph
        mask : np.ndarray
            Segmentation mask
        figsize : tuple
            Figure size
        highlight_edges : list of tuples, optional
            Edges to highlight in red
            
        Returns
        -------
        plt.Figure
            The generated figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Show segmentation mask
        unique_labels = np.unique(mask)
        n_labels = len(unique_labels)
        cmap = colors.ListedColormap(plt.cm.tab20(np.linspace(0, 1, n_labels)))
        
        ax1.imshow(mask, cmap=cmap, interpolation='nearest')
        ax1.set_title('Segmentation Mask')
        ax1.axis('off')
        
        # Show adjacency graph on mask
        ax2.imshow(mask, cmap=cmap, interpolation='nearest', alpha=0.3)
        
        # Calculate node positions as centroids
        pos = {}
        for label in graph.nodes():
            y, x = np.where(mask == label)
            if len(x) > 0:
                pos[label] = (np.mean(x), np.mean(y))
        
        # Draw graph
        nx.draw_networkx_nodes(graph, pos, ax=ax2, node_size=300, node_color='white', 
                              edgecolors='black', linewidths=2)
        
        # Draw edges with weights
        edges = graph.edges()
        weights = [graph[u][v]['weight'] for u, v in edges]
        
        if highlight_edges:
            edge_colors = ['red' if (u, v) in highlight_edges or (v, u) in highlight_edges 
                          else 'black' for u, v in edges]
        else:
            edge_colors = 'black'
            
        nx.draw_networkx_edges(graph, pos, ax=ax2, width=2, edge_color=edge_colors)
        nx.draw_networkx_labels(graph, pos, ax=ax2, font_size=10, font_weight='bold')
        
        # Add edge weights as labels
        edge_labels = {(u, v): f"{graph[u][v]['weight']:.0f}" for u, v in edges}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, ax=ax2, font_size=8)
        
        ax2.set_title('Adjacency Graph')
        ax2.axis('off')
        
        # Show graph structure only
        ax3.axis('off')
        nx.draw(graph, pos, ax=ax3, with_labels=True, node_size=500, 
                node_color='lightblue', font_size=12, font_weight='bold',
                edge_color='gray', width=[w/10 for w in weights])
        ax3.set_title('Graph Structure')
        
        plt.tight_layout()
        return fig
    
    def process_timelapse(
        self,
        masks: List[np.ndarray]
    ) -> List[nx.Graph]:
        """
        Process a series of segmentation masks to build graphs for each frame.
        
        Parameters
        ----------
        masks : list of np.ndarray
            List of segmentation masks for each timepoint
            
        Returns
        -------
        list of nx.Graph
            List of adjacency graphs for each timepoint
        """
        graphs = []
        for mask in masks:
            graph = self.build_graph(mask)
            graphs.append(graph)
        return graphs
    
    def track_t1_transitions(
        self,
        graphs: List[nx.Graph],
        four_cell_labels: List[int]
    ) -> List[Dict[str, Union[float, bool]]]:
        """
        Track T1 transitions across a timelapse for a specific four-cell cluster.
        
        Parameters
        ----------
        graphs : list of nx.Graph
            List of adjacency graphs for each timepoint
        four_cell_labels : list of int
            Labels of the four cells to track [a, b, c, d]
            
        Returns
        -------
        list of dict
            T1 analysis results for each timepoint
        """
        results = []
        for i, graph in enumerate(graphs):
            try:
                analysis = self.analyze_four_cell_cluster(graph, four_cell_labels)
                analysis['timepoint'] = i
                results.append(analysis)
            except Exception as e:
                print(f"Warning: Could not analyze timepoint {i}: {e}")
                results.append({'timepoint': i, 'error': str(e)})
        
        return results