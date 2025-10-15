"""
Analysis functions for T1 transitions and cell topology.
"""

import networkx as nx
from typing import List, Tuple, Optional, Dict, Any


class T1TransitionAnalyzer:
    """
    Analyzes T1 transitions in four-cell clusters.
    """
    
    def __init__(self):
        """Initialize the T1 transition analyzer."""
        pass
    
    def find_t1_edge(self, graph: nx.Graph) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Find the T1 transition edge in a four-cell cluster.
        
        In a T1 transition, the critical edge connects the two cells that each
        have exactly 3 adjacencies (neighbors).
        
        Parameters
        ----------
        graph : nx.Graph
            Adjacency graph of the cell cluster
            
        Returns
        -------
        Tuple[Optional[Tuple[int, int]], float]
            - Edge tuple (node1, node2) if T1 edge found, None otherwise
            - Edge weight (0 if no T1 edge found)
        """
        # Find nodes with exactly 3 adjacencies
        nodes_with_3_adj = [node for node in graph.nodes() if graph.degree(node) == 3]
        
        if len(nodes_with_3_adj) == 2:
            node1, node2 = nodes_with_3_adj
            if graph.has_edge(node1, node2):
                edge_weight = graph[node1][node2]['weight']
                return (node1, node2), edge_weight
            else:
                # Two nodes with 3 adjacencies but no direct edge
                return None, 0
        else:
            # Unexpected number of nodes with 3 adjacencies
            return None, 0
    
    def analyze_t1_transition_over_time(
        self,
        graphs: List[nx.Graph],
        frame_numbers: List[int]
    ) -> Dict[str, List[Any]]:
        """
        Analyze T1 transitions across multiple frames.
        
        Parameters
        ----------
        graphs : List[nx.Graph]
            List of adjacency graphs for each frame
        frame_numbers : List[int]
            Frame numbers corresponding to each graph
            
        Returns
        -------
        Dict[str, List[Any]]
            Dictionary containing:
            - 'frames': List of frame numbers
            - 'edge_weights': List of T1 edge weights
            - 'edges': List of T1 edge tuples (or None)
            - 'has_t1_edge': List of boolean values
        """
        results = {
            'frames': [],
            'edge_weights': [],
            'edges': [],
            'has_t1_edge': []
        }
        
        for graph, frame_num in zip(graphs, frame_numbers):
            edge, weight = self.find_t1_edge(graph)
            
            results['frames'].append(frame_num)
            results['edge_weights'].append(weight)
            results['edges'].append(edge)
            results['has_t1_edge'].append(edge is not None)
        
        return results
    
    def detect_t1_events(
        self,
        edge_weights: List[float],
        frame_numbers: List[int],
        threshold_change: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Detect T1 transition events based on edge weight changes.
        
        Parameters
        ----------
        edge_weights : List[float]
            T1 edge weights over time
        frame_numbers : List[int]
            Corresponding frame numbers
        threshold_change : float
            Minimum change in edge weight to consider an event
            
        Returns
        -------
        List[Dict[str, Any]]
            List of detected events with frame numbers and weight changes
        """
        events = []
        
        for i in range(1, len(edge_weights)):
            weight_change = abs(edge_weights[i] - edge_weights[i-1])
            
            if weight_change > threshold_change:
                events.append({
                    'frame_start': frame_numbers[i-1],
                    'frame_end': frame_numbers[i],
                    'weight_change': weight_change,
                    'weight_before': edge_weights[i-1],
                    'weight_after': edge_weights[i],
                    'event_type': 'weight_drop' if edge_weights[i] < edge_weights[i-1] else 'weight_increase'
                })
        
        return events


class TopologyAnalyzer:
    """
    Analyzes cell cluster topology and connectivity patterns.
    """
    
    def __init__(self):
        """Initialize the topology analyzer."""
        pass
    
    def analyze_cluster_topology(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Analyze the topology of a cell cluster.
        
        Parameters
        ----------
        graph : nx.Graph
            Adjacency graph of the cell cluster
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing topology metrics:
            - 'num_cells': Number of cells
            - 'num_edges': Number of adjacency relationships
            - 'degree_sequence': List of node degrees
            - 'avg_degree': Average degree
            - 'is_connected': Whether graph is connected
            - 'clustering_coefficient': Global clustering coefficient
        """
        num_cells = len(graph.nodes())
        num_edges = len(graph.edges())
        degree_sequence = [graph.degree(node) for node in graph.nodes()]
        avg_degree = sum(degree_sequence) / len(degree_sequence) if degree_sequence else 0
        
        return {
            'num_cells': num_cells,
            'num_edges': num_edges,
            'degree_sequence': degree_sequence,
            'avg_degree': avg_degree,
            'is_connected': nx.is_connected(graph),
            'clustering_coefficient': nx.average_clustering(graph) if num_cells > 0 else 0
        }
    
    def identify_four_cell_configuration(self, graph: nx.Graph) -> Optional[str]:
        """
        Identify the configuration type of a four-cell cluster.
        
        Parameters
        ----------
        graph : nx.Graph
            Adjacency graph (should have 4 nodes)
            
        Returns
        -------
        Optional[str]
            Configuration type: 'linear', 'square', 'triangle_plus_one', 't1_ready', or None
        """
        if len(graph.nodes()) != 4:
            return None
        
        degree_sequence = sorted([graph.degree(node) for node in graph.nodes()])
        
        if degree_sequence == [1, 1, 2, 2]:
            return 'linear'  # Chain configuration
        elif degree_sequence == [2, 2, 2, 2]:
            # Could be square or diamond
            if len(graph.edges()) == 4:
                return 'square'  # Square with 4 edges
            else:
                return 'diamond'  # Diamond with more edges
        elif degree_sequence == [2, 3, 3, 4] or degree_sequence == [2, 2, 3, 3]:
            return 't1_ready'  # Configuration ready for T1 transition
        elif degree_sequence == [3, 3, 3, 3]:
            return 'fully_connected'  # All cells touching all others
        else:
            return 'other'
    
    def track_topology_changes(
        self,
        graphs: List[nx.Graph],
        frame_numbers: List[int]
    ) -> Dict[str, List[Any]]:
        """
        Track topology changes across frames.
        
        Parameters
        ----------
        graphs : List[nx.Graph]
            List of graphs for each frame
        frame_numbers : List[int]
            Corresponding frame numbers
            
        Returns
        -------
        Dict[str, List[Any]]
            Dictionary tracking topology metrics over time
        """
        results = {
            'frames': frame_numbers,
            'num_edges': [],
            'avg_degrees': [],
            'configurations': [],
            'clustering_coeffs': []
        }
        
        for graph in graphs:
            topology = self.analyze_cluster_topology(graph)
            config = self.identify_four_cell_configuration(graph)
            
            results['num_edges'].append(topology['num_edges'])
            results['avg_degrees'].append(topology['avg_degree'])
            results['configurations'].append(config)
            results['clustering_coeffs'].append(topology['clustering_coefficient'])
        
        return results