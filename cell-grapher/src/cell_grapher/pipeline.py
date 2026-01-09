"""
Integrated pipeline for cell tracking, segmentation, and T1 transition analysis.
"""

import numpy as np
import os
from typing import Dict, Optional, Any

from .segmentation import SegmentationLoader
from .adjacency import AdjacencyGraphBuilder
from .tracking import CellTracker
from .trajectory import TrajectoryAnalyzer
from .visualization import CellVisualizationManager, plot_t1_transition_analysis, save_t1_data_csv, create_video_from_arrays
from .analysis import T1TransitionAnalyzer, TopologyAnalyzer
from .trajectory_visualization import plot_trajectory_visualizations


class CellTrackingPipeline:
    """
    Complete pipeline for cell tracking and T1 transition analysis using pre-computed segmentations.
    """
    
    def __init__(
        self,
        adjacency_params: Optional[Dict] = None,
        tracking_params: Optional[Dict] = None,
        trajectory_params: Optional[Dict] = None,
        nucleus_channel_name: str = 'cell_ch_1'
    ):
        """
        Initialize the cell tracking pipeline.

        Parameters
        ----------
        adjacency_params : dict, optional
            Parameters for AdjacencyGraphBuilder
        tracking_params : dict, optional
            Parameters for CellTracker (btrack-based)
        trajectory_params : dict, optional
            Parameters for TrajectoryAnalyzer
        nucleus_channel_name : str
            Name of nucleus channel in YAML (default: 'cell_ch_1')
        """
        # Initialize components with default parameters
        adj_params = adjacency_params or {'method': 'boundary_length'}
        track_params = tracking_params or {'search_radius': 100.0}
        traj_params = trajectory_params or {'nucleus_threshold': 0.0}

        self.segmentation_loader = SegmentationLoader()
        self.graph_builder = AdjacencyGraphBuilder(**adj_params)
        self.tracker = CellTracker(**track_params)
        self.trajectory_analyzer = TrajectoryAnalyzer(**traj_params)
        self.visualizer = CellVisualizationManager()
        self.t1_analyzer = T1TransitionAnalyzer()
        self.topology_analyzer = TopologyAnalyzer()

        self.nucleus_channel_name = nucleus_channel_name

        # Storage for results
        self.results = {
            'frames': [],
            'segmentations': [],
            'graphs': [],
            'tracking_maps': [],
            't1_data': [],
            'topology_data': [],
            'trajectory_data': []
        }
    
    def reset(self):
        """Reset pipeline state for a new analysis."""
        self.tracker.reset()
        self.trajectory_analyzer.reset()
        self.visualizer = CellVisualizationManager()
        self.results = {
            'frames': [],
            'segmentations': [],
            'graphs': [],
            'tracking_maps': [],
            't1_data': [],
            'topology_data': [],
            'trajectory_data': []
        }
    
    def process_frame(
        self,
        segmentation_mask: np.ndarray,
        cell_image: Optional[np.ndarray] = None,
        frame_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Process a single frame through the complete pipeline.
        
        Parameters
        ----------
        segmentation_mask : np.ndarray
            Pre-computed segmentation mask with integer labels for each cell
        cell_image : np.ndarray, optional
            Cell image data for visualization (height, width, channels)
        frame_idx : int
            Frame number
            
        Returns
        -------
        Dict[str, Any]
            Frame processing results
        """
        # Use pre-computed segmentation
        masks = segmentation_mask
        
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
            'num_edges': len(graph.edges()),
            'cell_image': cell_image
        }
        
        # Update pipeline storage
        self.results['frames'].append(frame_idx)
        self.results['segmentations'].append(masks)
        self.results['graphs'].append(graph)
        self.results['tracking_maps'].append(tracking_map)
        self.results['t1_data'].append(t1_weight)
        self.results['topology_data'].append(topology)
        
        return frame_result
    
    def process_cell_filter_data(
        self,
        npy_path: str,
        yaml_path: Optional[str] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        output_dir: str = None
    ) -> Dict[str, Any]:
        """
        Process cell-filter timelapse data using pre-computed segmentations.
        
        Parameters
        ----------
        npy_path : str
            Path to the cell-filter NPY file
        yaml_path : str, optional
            Path to the corresponding YAML metadata file
        start_frame : int
            First frame to process
        end_frame : int, optional
            Last frame to process (None for all frames)
        output_dir : str
            Directory to save outputs
            
        Returns
        -------
        Dict[str, Any]
            Complete analysis results
        """
        # Validate and load cell-filter data
        if not self.segmentation_loader.validate_cell_filter_output(npy_path, yaml_path):
            raise ValueError(f"Invalid cell-filter output format: {npy_path}")
        
        cell_filter_data = self.segmentation_loader.load_cell_filter_data(npy_path, yaml_path)
        
        # Extract cell channels for visualization
        cell_channels = self.segmentation_loader.get_cell_channels(cell_filter_data)
        
        # Setup
        n_frames = len(cell_filter_data['segmentation_masks'])
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = n_frames
        
        os.makedirs(output_dir, exist_ok=True)
        self.reset()

        print(f"Processing frames {start_frame} to {end_frame-1}...")
        print(f"Data shape: {cell_filter_data['data'].shape}")
        print(f"Channels: {cell_filter_data['channels']}")

        # Setup video frames list
        video_frames = []

        # Collect all masks first and run tracking in batch (btrack approach)
        print("Using btrack (Bayesian tracker) for cell tracking...")

        # Collect all segmentation masks
        print("Collecting segmentation masks...")
        for frame_idx in range(start_frame, end_frame):
            segmentation_mask = self.segmentation_loader.get_frame_mask(
                cell_filter_data, frame_idx
            )
            self.tracker.add_frame(segmentation_mask, frame_idx)

        # Run tracking on all frames at once
        print("Running btrack on all frames...")
        self.tracker.track_all_frames()
        print(f"Tracking complete! Found {len(self.tracker.tracks)} tracks.")

        # Now process each frame with tracking results
        for frame_idx in range(start_frame, end_frame):
            print(f"Processing frame {frame_idx}/{end_frame-1}...")

            # Get segmentation mask
            segmentation_mask = self.segmentation_loader.get_frame_mask(
                cell_filter_data, frame_idx
            )

            # Get cell image for visualization if available
            cell_image = None
            if cell_channels.size > 0:
                frame_cell_data = cell_channels[frame_idx]
                if frame_cell_data.ndim == 3:
                    cell_image = np.transpose(frame_cell_data, (1, 2, 0))
                elif frame_cell_data.ndim == 2:
                    cell_image = frame_cell_data

            # Get tracking map from btrack results
            tracking_map = self.tracker.get_tracking_map(frame_idx)
            tracked_mask = self.tracker.create_tracked_mask(segmentation_mask, tracking_map)

            # Update visualization colors
            self.visualizer.update_color_map(tracking_map)

            # Get nucleus channel for trajectory analysis
            nucleus_channel = self.segmentation_loader.get_nucleus_channel(
                cell_filter_data, frame_idx, self.nucleus_channel_name
            )

            # Trajectory analysis (nucleus and center of mass)
            if nucleus_channel is not None:
                self.trajectory_analyzer.analyze_frame(
                    frame_idx, segmentation_mask, tracking_map, nucleus_channel
                )

            # Build adjacency graph
            graph = self.graph_builder.build_graph(segmentation_mask)

            # T1 analysis
            t1_edge, t1_weight = self.t1_analyzer.find_t1_edge(graph)

            # Topology analysis
            topology = self.topology_analyzer.analyze_cluster_topology(graph)

            # Store frame result
            frame_result = {
                'frame_idx': frame_idx,
                'masks': segmentation_mask,
                'tracked_mask': tracked_mask,
                'tracking_map': tracking_map,
                'graph': graph,
                't1_edge': t1_edge,
                't1_weight': t1_weight,
                'topology': topology,
                'num_cells': segmentation_mask.max(),
                'num_edges': len(graph.edges()),
                'cell_image': cell_image
            }

            # Update pipeline storage
            self.results['frames'].append(frame_idx)
            self.results['segmentations'].append(segmentation_mask)
            self.results['graphs'].append(graph)
            self.results['tracking_maps'].append(tracking_map)
            self.results['t1_data'].append(t1_weight)
            self.results['topology_data'].append(topology)

            # Create visualization frame (in-memory, no PNG save)
            try:
                frame_array = self._create_frame_visualization_array(
                    frame_result, frame_idx
                )
                video_frames.append(frame_array)
            except Exception as e:
                print(f"  Warning: Could not create visualization for frame {frame_idx}: {e}")
                # Skip this frame's visualization but continue processing
                continue

            # Print progress
            print(f"  Found {frame_result['num_cells']} cells, "
                  f"{frame_result['num_edges']} edges")
            if frame_result['t1_edge']:
                print(f"  T1 edge: {frame_result['t1_edge']} "
                      f"weight: {frame_result['t1_weight']:.1f}")
            print(f"  Tracking: {frame_result['tracking_map']}")
        
        # Create video from frames
        try:
            video_path = create_video_from_arrays(
                video_frames=video_frames,
                output_dir=output_dir,
                video_filename="cell_analysis_video.mp4",
                fps=10
            )
            print(f"Video created: {video_path}")
        except Exception as e:
            print(f"Warning: Could not create video: {e}")
            video_path = None
        
        # Generate summary analysis
        summary = self._generate_summary_analysis(output_dir, video_path)
        
        print(f"\nCompleted! Analysis saved to {output_dir}/")
        return summary
    
    def _create_frame_visualization_array(
        self,
        frame_result: Dict[str, Any],
        frame_idx: int
    ) -> np.ndarray:
        """Create visualization frame as numpy array (in-memory, no PNG save)."""
        cell_image = frame_result.get('cell_image')
        
        if cell_image is not None:
            if cell_image.ndim == 3:
                # Multi-channel image
                nuclei_image = cell_image[:, :, 1] if cell_image.shape[2] > 1 else cell_image[:, :, 0]
                cytoplasm_image = cell_image[:, :, 0]
            else:
                # Single channel image
                nuclei_image = cell_image
                cytoplasm_image = cell_image
        else:
            # No cell image available - use segmentation masks
            nuclei_image = np.zeros_like(frame_result['masks'])
            cytoplasm_image = np.zeros_like(frame_result['masks'])
        
        # Use original mask with local labels for graph, but show tracked IDs
        segmentation_mask = frame_result['masks']  # Original with local labels
        tracking_map = frame_result['tracking_map']
        graph = frame_result['graph']
        t1_edge = frame_result['t1_edge']

        # Create figure and get the visualization as array
        fig_array = self.visualizer.create_composite_frame_array(
            nuclei_image=nuclei_image,
            cytoplasm_image=cytoplasm_image,
            segmentation_mask=segmentation_mask,
            tracking_map=tracking_map,
            graph=graph,
            t1_edge=t1_edge,
            frame_idx=frame_idx
        )
        
        return fig_array

    def _save_trajectory_csv(self, csv_path: str):
        """Save trajectory data to CSV file."""
        import csv

        trajectory_data = self.trajectory_analyzer.export_trajectories_to_dict()

        if not trajectory_data:
            print("Warning: No trajectory data to save")
            return

        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['global_id', 'frame', 'nucleus_y', 'nucleus_x',
                          'cytoplasm_y', 'cytoplasm_x', 'nucleus_intensity']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in trajectory_data:
                writer.writerow(row)

        print(f"Cell trajectories saved to {csv_path}")

    def _generate_summary_analysis(self, output_dir: str, video_path: Optional[str] = None) -> Dict[str, Any]:
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

        # Save trajectory data as CSV
        trajectory_csv_path = os.path.join(output_dir, 'cell_trajectories.csv')
        self._save_trajectory_csv(trajectory_csv_path)

        # Generate trajectory visualization plots
        print("\nGenerating trajectory visualizations...")
        trajectory_plots = plot_trajectory_visualizations(
            self.trajectory_analyzer.get_all_trajectories(),
            output_dir
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

        # Trajectory summary
        trajectory_summary = self.trajectory_analyzer.get_trajectory_summary()

        # Summary statistics
        summary = {
            'total_frames': len(self.results['frames']),
            'frame_range': (min(self.results['frames']), max(self.results['frames'])),
            't1_weight_range': (min(self.results['t1_data']), max(self.results['t1_data'])),
            't1_events_detected': len(t1_events),
            't1_events': t1_events,
            'topology_tracking': topology_tracking,
            'trajectory_summary': trajectory_summary,
            'output_files': {
                't1_plot': t1_plot_path,
                't1_data': csv_path,
                'trajectories': trajectory_csv_path
            }
        }

        # Add trajectory plot paths if generated
        if trajectory_plots:
            summary['output_files'].update(trajectory_plots)
        
        # Add video path if available
        if video_path:
            summary['output_files']['video'] = video_path
        
        # Print summary
        print(f"T1 transition analysis plot saved to {t1_plot_path}")
        print(f"T1 edge weight range: {summary['t1_weight_range'][0]:.1f} - {summary['t1_weight_range'][1]:.1f}")
        print(f"T1 events detected: {len(t1_events)}")
        print(f"T1 transition data saved to {csv_path}")
        print(f"Trajectory data: {trajectory_summary['num_tracks']} cells tracked, "
              f"mean track length: {trajectory_summary['mean_track_length']:.1f} frames")
        if video_path:
            print(f"Video created: {video_path}")

        return summary


# Convenience function for quick analysis
def analyze_cell_filter_data(
    npy_path: str,
    yaml_path: Optional[str] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    output_dir: str = None,
    **pipeline_params
) -> Dict[str, Any]:
    """
    Convenience function to analyze cell-filter data with pre-computed segmentations.
    
    Parameters
    ----------
    npy_path : str
        Path to cell-filter NPY data file
    yaml_path : str, optional
        Path to corresponding YAML metadata file
    output_dir : str
        Output directory for results
    start_frame : int
        Starting frame number
    **pipeline_params
        Additional parameters for pipeline components
        
    Returns
    -------
    Dict[str, Any]
        Analysis summary
    """
    # Initialize pipeline
    pipeline = CellTrackingPipeline(**pipeline_params)
    
    # Run analysis
    return pipeline.process_cell_filter_data(
        npy_path=npy_path,
        yaml_path=yaml_path,
        start_frame=start_frame,
        end_frame=end_frame,
        output_dir=output_dir
    )