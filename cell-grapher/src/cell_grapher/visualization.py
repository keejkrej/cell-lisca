"""
Visualization utilities for cell tracking analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional
from matplotlib import colors


class CellVisualizationManager:
    """
    Manages consistent visualization of tracked cells across frames.
    """
    
    def __init__(self, max_colors: int = 10):
        """
        Initialize visualization manager.
        
        Parameters
        ----------
        max_colors : int
            Maximum number of distinct colors to use
        """
        self.max_colors = max_colors
        self.colors = plt.cm.Set1(np.linspace(0, 1, max_colors))
        self.global_color_map = {}
        
    def get_color_for_global_id(self, global_id: int) -> Tuple[float, ...]:
        """
        Get consistent color for a global cell ID.
        
        Parameters
        ----------
        global_id : int
            Global cell identifier
            
        Returns
        -------
        Tuple[float, ...]
            RGBA color tuple
        """
        if global_id not in self.global_color_map:
            color_idx = (global_id - 1) % len(self.colors)
            self.global_color_map[global_id] = self.colors[color_idx]
        return self.global_color_map[global_id]
    
    def update_color_map(self, tracking_map: Dict[int, int]):
        """
        Update color map with new global IDs.
        
        Parameters
        ----------
        tracking_map : Dict[int, int]
            Mapping from local labels to global IDs
        """
        for global_id in tracking_map.values():
            self.get_color_for_global_id(global_id)
    
    def create_multi_panel_figure(
        self,
        nuclei_image: np.ndarray,
        cytoplasm_image: np.ndarray,
        tracked_mask: np.ndarray,
        graph: nx.Graph,
        original_mask: np.ndarray,
        tracking_map: Dict[int, int],
        frame_idx: int,
        figsize: Tuple[int, int] = (15, 5)
    ) -> plt.Figure:
        """
        Create a 3-panel visualization figure.
        
        Parameters
        ----------
        nuclei_image : np.ndarray
            Nuclei fluorescence channel
        cytoplasm_image : np.ndarray
            Cytoplasm fluorescence channel
        tracked_mask : np.ndarray
            Segmentation mask with global IDs
        graph : nx.Graph
            Adjacency graph with local labels
        original_mask : np.ndarray
            Original segmentation mask with local labels
        tracking_map : Dict[int, int]
            Mapping from local to global labels
        frame_idx : int
            Frame number for titles
        figsize : Tuple[int, int]
            Figure size
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Panel 1: Nuclei channel
        axes[0].imshow(nuclei_image, cmap='gray')
        axes[0].set_title(f'Frame {frame_idx} - Nuclei')
        axes[0].axis('off')
        
        # Panel 2: Cytoplasm channel
        axes[1].imshow(cytoplasm_image, cmap='gray')
        axes[1].set_title(f'Frame {frame_idx} - Cytoplasm')
        axes[1].axis('off')
        
        # Panel 3: Graph on segmentation with consistent colors
        axes[2].imshow(tracked_mask, cmap='tab20', alpha=0.5)
        
        # Draw graph with consistent node colors
        self._draw_graph_with_tracking(
            axes[2], graph, original_mask, tracking_map, frame_idx
        )
        
        plt.tight_layout()
        return fig
    
    def _draw_graph_with_tracking(
        self,
        ax: plt.Axes,
        graph: nx.Graph,
        original_mask: np.ndarray,
        tracking_map: Dict[int, int],
        frame_idx: int
    ):
        """
        Draw graph with consistent node colors and global ID labels.
        
        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes to draw on
        graph : nx.Graph
            NetworkX graph with local labels
        original_mask : np.ndarray
            Original segmentation mask
        tracking_map : Dict[int, int]
            Mapping from local to global labels
        frame_idx : int
            Frame number for title
        """
        # Calculate node positions as centroids
        pos = {}
        node_colors = []
        node_labels = {}
        
        for label in graph.nodes():
            y, x = np.where(original_mask == label)
            if len(x) > 0:
                pos[label] = (np.mean(x), np.mean(y))
                # Use global ID for consistent coloring and labeling
                global_id = tracking_map[label]
                node_colors.append(self.get_color_for_global_id(global_id))
                node_labels[label] = str(global_id)
        
        # Draw graph components
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, node_size=300,
            node_color=node_colors, edgecolors='black', linewidths=2
        )
        nx.draw_networkx_edges(
            graph, pos, ax=ax, width=3, edge_color='red'
        )
        nx.draw_networkx_labels(
            graph, pos, node_labels, ax=ax, font_size=12, font_weight='bold'
        )
        
        # Add edge weights as labels
        edge_labels = {
            (u, v): f"{graph[u][v]['weight']:.0f}" for u, v in graph.edges()
        }
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels, ax=ax, font_size=10
        )
        
        ax.set_title(f'Frame {frame_idx} - Segmentation + Graph ({len(graph.edges())} edges)')
        ax.axis('off')


def plot_t1_transition_analysis(
    frame_numbers: List[int],
    t1_edge_weights: List[float],
    output_path: str,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Create and save T1 transition analysis plot.
    
    Parameters
    ----------
    frame_numbers : List[int]
        Frame numbers
    t1_edge_weights : List[float]
        T1 edge weights for each frame
    output_path : str
        Path to save the plot
    figsize : Tuple[int, int]
        Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(frame_numbers, t1_edge_weights, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Frame Number')
    plt.ylabel('T1 Edge Weight (boundary length)')
    plt.title('T1 Transition Edge Weight Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_t1_data_csv(
    frame_numbers: List[int],
    t1_edge_weights: List[float],
    output_path: str
) -> None:
    """
    Save T1 transition data to CSV file.
    
    Parameters
    ----------
    frame_numbers : List[int]
        Frame numbers
    t1_edge_weights : List[float]
        T1 edge weights for each frame
    output_path : str
        Path to save the CSV file
    """
    import csv
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame', 't1_edge_weight'])
        for frame, weight in zip(frame_numbers, t1_edge_weights):
            writer.writerow([frame, weight])