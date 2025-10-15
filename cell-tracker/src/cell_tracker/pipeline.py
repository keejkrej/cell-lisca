"""
Integrated pipeline for cell tracking, segmentation, and T1 transition analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple, Optional, Any

from .segmentation import CellposeSegmenter
from .adjacency import AdjacencyGraphBuilder
from .tracking import CellTracker
from .visualization import CellVisualizationManager, plot_t1_transition_analysis, save_t1_data_csv
from .analysis import T1TransitionAnalyzer, TopologyAnalyzer


class CellTrackingPipeline:
    """
    Complete pipeline for cell tracking and T1 transition analysis.
    """
    
    def __init__(
        self,
        segmentation_params: Optional[Dict] = None,
        adjacency_params: Optional[Dict] = None,
        tracking_params: Optional[Dict] = None
    ):
        """
        Initialize the cell tracking pipeline.
        
        Parameters
        ----------
        segmentation_params : dict, optional
            Parameters for CellposeSegmenter
        adjacency_params : dict, optional
            Parameters for AdjacencyGraphBuilder
        tracking_params : dict, optional
            Parameters for CellTracker
        """
        # Initialize components with default parameters
        seg_params = segmentation_params or {'gpu': True}
        adj_params = adjacency_params or {'method': 'boundary_length'}
        track_params = tracking_params or {'iou_threshold': 0.3}
        
        self.segmenter = CellposeSegmenter(**seg_params)
        self.graph_builder = AdjacencyGraphBuilder(**adj_params)
        self.tracker = CellTracker(**track_params)
        self.visualizer = CellVisualizationManager()
        self.t1_analyzer = T1TransitionAnalyzer()
        self.topology_analyzer = TopologyAnalyzer()
        
        # Storage for results
        self.results = {
            'frames': [],
            'segmentations': [],
            'graphs': [],
            'tracking_maps': [],
            't1_data': [],
            'topology_data': []
        }
    
    def reset(self):
        """Reset pipeline state for a new analysis."""
        self.tracker.reset()
        self.visualizer = CellVisualizationManager()
        self.results = {
            'frames': [],
            'segmentations': [],
            'graphs': [],
            'tracking_maps': [],
            't1_data': [],
            'topology_data': []
        }
    
    def process_frame(
        self,
        image: np.ndarray,
        frame_idx: int
    ) -> Dict[str, Any]:
        """
        Process a single frame through the complete pipeline.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (height, width, channels)
        frame_idx : int
            Frame number
            
        Returns
        -------
        Dict[str, Any]
            Frame processing results
        """
        # Segmentation
        seg_result = self.segmenter.segment_image(image)
        masks = seg_result['masks']
        
        # Tracking
        tracking_map = self.tracker.track_frame(masks)
        tracked_mask = self.tracker.create_tracked_mask(masks, tracking_map)
        
        # Update visualization colors
        self.visualizer.update_color_map(tracking_map)
        
        # Adjacency graph
        graph = self.graph_builder.build_graph(masks)
        
        # T1 analysis
        t1_edge, t1_weight = self.t1_analyzer.find_t1_edge(graph)
        
        # Topology analysis
        topology = self.topology_analyzer.analyze_cluster_topology(graph)
        
        # Store results
        frame_result = {
            'frame_idx': frame_idx,
            'masks': masks,
            'tracked_mask': tracked_mask,
            'tracking_map': tracking_map,
            'graph': graph,
            't1_edge': t1_edge,
            't1_weight': t1_weight,
            'topology': topology,
            'num_cells': masks.max(),
            'num_edges': len(graph.edges())
        }
        
        # Update pipeline storage
        self.results['frames'].append(frame_idx)
        self.results['segmentations'].append(masks)
        self.results['graphs'].append(graph)
        self.results['tracking_maps'].append(tracking_map)
        self.results['t1_data'].append(t1_weight)
        self.results['topology_data'].append(topology)
        
        return frame_result
    
    def process_timelapse(
        self,
        data: np.ndarray,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        channels: Optional[List[int]] = None,
        output_dir: str = 'analysis_output'
    ) -> Dict[str, Any]:
        """
        Process a complete timelapse dataset.
        
        Parameters
        ----------
        data : np.ndarray
            Timelapse data (frames, channels, height, width)
        start_frame : int
            First frame to process
        end_frame : int, optional
            Last frame to process (None for all frames)
        channels : List[int], optional
            Channel indices for segmentation
        output_dir : str
            Directory to save outputs
            
        Returns
        -------
        Dict[str, Any]
            Complete analysis results
        """
        # Setup
        if end_frame is None:
            end_frame = data.shape[0]
        
        os.makedirs(output_dir, exist_ok=True)
        self.reset()
        
        print(f"Processing frames {start_frame} to {end_frame-1}...")
        
        # Process each frame
        for frame_idx in range(start_frame, end_frame):
            print(f"Processing frame {frame_idx}/{end_frame-1}...")
            
            # Extract and prepare image
            if len(data.shape) == 4:  # (frames, channels, height, width)
                frame_data = data[frame_idx, 1:4]  # Skip pattern channel
                image = np.transpose(frame_data, (1, 2, 0))  # (height, width, channels)
            else:
                image = data[frame_idx]
            
            # Process frame
            frame_result = self.process_frame(image, frame_idx)
            
            # Create visualization
            self._save_frame_visualization(
                frame_result, image, output_dir, frame_idx
            )
            
            # Print progress
            print(f"  Found {frame_result['num_cells']} cells, "
                  f"{frame_result['num_edges']} edges")
            if frame_result['t1_edge']:
                print(f"  T1 edge: {frame_result['t1_edge']} "
                      f"weight: {frame_result['t1_weight']:.1f}")
            print(f"  Tracking: {frame_result['tracking_map']}")
        
        # Generate summary analysis
        summary = self._generate_summary_analysis(output_dir)
        
        print(f"\nCompleted! Analysis saved to {output_dir}/")
        return summary
    
    def _save_frame_visualization(
        self,
        frame_result: Dict[str, Any],
        image: np.ndarray,
        output_dir: str,
        frame_idx: int
    ):
        """Save visualization for a single frame."""
        # Extract channels for visualization
        nuclei_image = image[:, :, 1] if image.shape[2] > 1 else image[:, :, 0]
        cytoplasm_image = image[:, :, 0]
        
        # Create visualization
        fig = self.visualizer.create_multi_panel_figure(
            nuclei_image=nuclei_image,
            cytoplasm_image=cytoplasm_image,
            tracked_mask=frame_result['tracked_mask'],
            graph=frame_result['graph'],
            original_mask=frame_result['masks'],
            tracking_map=frame_result['tracking_map'],
            frame_idx=frame_idx
        )
        
        # Save
        png_path = os.path.join(output_dir, f'frame_{frame_idx:03d}_analysis.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _generate_summary_analysis(self, output_dir: str) -> Dict[str, Any]:
        """Generate and save summary analysis."""
        # T1 transition analysis
        t1_plot_path = os.path.join(output_dir, 't1_transition_analysis.png')
        plot_t1_transition_analysis(
            self.results['frames'],
            self.results['t1_data'],
            t1_plot_path
        )
        
        # Save T1 data as CSV
        csv_path = os.path.join(output_dir, 't1_transition_data.csv')
        save_t1_data_csv(
            self.results['frames'],
            self.results['t1_data'],
            csv_path
        )
        
        # Topology analysis over time
        topology_tracking = self.topology_analyzer.track_topology_changes(
            self.results['graphs'],
            self.results['frames']
        )
        
        # T1 event detection
        t1_events = self.t1_analyzer.detect_t1_events(
            self.results['t1_data'],
            self.results['frames']
        )
        
        # Summary statistics
        summary = {
            'total_frames': len(self.results['frames']),
            'frame_range': (min(self.results['frames']), max(self.results['frames'])),
            't1_weight_range': (min(self.results['t1_data']), max(self.results['t1_data'])),
            't1_events_detected': len(t1_events),
            't1_events': t1_events,
            'topology_tracking': topology_tracking,
            'output_files': {
                't1_plot': t1_plot_path,
                't1_data': csv_path,
                'frame_analyses': f"{output_dir}/frame_*_analysis.png"
            }
        }
        
        # Print summary
        print(f"T1 transition analysis plot saved to {t1_plot_path}")
        print(f"T1 edge weight range: {summary['t1_weight_range'][0]:.1f} - {summary['t1_weight_range'][1]:.1f}")
        print(f"T1 events detected: {len(t1_events)}")
        print(f"T1 transition data saved to {csv_path}")
        
        return summary


# Convenience function for quick analysis
def analyze_timelapse_data(
    data_path: str,
    output_dir: str = 'analysis_output',
    start_frame: int = 21,
    channels: Optional[List[int]] = None,
    **pipeline_params
) -> Dict[str, Any]:
    """
    Convenience function to analyze timelapse data from a file.
    
    Parameters
    ----------
    data_path : str
        Path to .npy data file
    output_dir : str
        Output directory for results
    start_frame : int
        Starting frame number
    channels : List[int], optional
        Channels to use for segmentation
    **pipeline_params
        Additional parameters for pipeline components
        
    Returns
    -------
    Dict[str, Any]
        Analysis summary
    """
    # Load data
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    print(f"Data shape: {data.shape}")
    
    # Initialize pipeline
    pipeline = CellTrackingPipeline(**pipeline_params)
    
    # Run analysis
    return pipeline.process_timelapse(
        data=data,
        start_frame=start_frame,
        channels=channels,
        output_dir=output_dir
    )