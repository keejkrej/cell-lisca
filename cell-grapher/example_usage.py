#!/usr/bin/env python3
"""
Example script demonstrating how to use the updated cell-grapher 
with cell-filter output data that includes pre-computed segmentation masks.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cell_grapher.pipeline import analyze_cell_filter_data


def main():
    """Example usage of cell-grapher with cell-filter output."""
    
    # Example paths - replace with your actual file paths
    npy_file = "path/to/your/fov_000_pattern_000_seq_000.npy"
    yaml_file = "path/to/your/fov_000_pattern_000_seq_000.yaml"  # Optional
    
    # Output directory for analysis results
    output_dir = "graph_analysis_output"
    
    # Run the analysis
    print("Starting cell graph analysis...")
    results = analyze_cell_filter_data(
        npy_path=npy_file,
        yaml_path=yaml_file,
        output_dir=output_dir,
        start_frame=0,  # Start from first frame
        # Optional pipeline parameters:
        # tracking_params={'iou_threshold': 0.3},
        # adjacency_params={'method': 'boundary_length'}
    )
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total frames processed: {results['total_frames']}")
    print(f"Frame range: {results['frame_range']}")
    print(f"T1 edge weight range: {results['t1_weight_range']}")
    print(f"T1 events detected: {results['t1_events_detected']}")
    print(f"Output files saved to: {output_dir}/")
    
    # List output files
    for file_type, path in results['output_files'].items():
        print(f"  {file_type}: {path}")


if __name__ == "__main__":
    main()