"""
Main entry point for cell-grapher module.

This allows running: python -m cell_grapher
"""

import argparse
import sys
from pathlib import Path

from .pipeline import analyze_cell_filter_data


def main():
    """Main CLI entry point for cell-grapher."""
    parser = argparse.ArgumentParser(
        description="Create region adjacency graphs and analyze T1 transitions using pre-computed segmentations from cell-filter."
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to cell-filter NPY file containing timelapse data with segmentation channel"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for analysis results"
    )
    
    parser.add_argument(
        "--yaml", "-y",
        help="Path to corresponding YAML metadata file (optional, auto-detected if not provided)"
    )
    
    parser.add_argument(
        "--start-frame", "-s",
        type=int,
        help="Starting frame number (processes from first frame if not specified)"
    )
    
    parser.add_argument(
        "--end-frame", "-e",
        type=int,
        help="Ending frame number (exclusive, processes all frames if not specified)"
    )
    
    parser.add_argument(
        "--search-radius",
        type=float,
        default=100.0,
        help="Maximum search radius for cell tracking in pixels (default: 100.0)"
    )

    parser.add_argument(
        "--tracking-config",
        help="Path to btrack config JSON file (optional, uses default if not specified)"
    )

    parser.add_argument(
        "--adjacency-method",
        choices=["boundary_length", "centroid_distance"],
        default="boundary_length",
        help="Method for building adjacency graphs (default: boundary_length)"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate input format, don't run analysis"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file does not exist: {args.input}")
        sys.exit(1)
    
    # Auto-detect YAML file if not provided
    yaml_path = args.yaml
    if yaml_path is None:
        input_path = Path(args.input)
        yaml_path = str(input_path.with_suffix('.yaml'))
        if not Path(yaml_path).exists():
            yaml_path = None
    
    print("Cell Grapher CLI")
    print(f"Input NPY: {args.input}")
    if yaml_path:
        print(f"Input YAML: {yaml_path}")
    else:
        print("Input YAML: Not found, using default channel settings")
    print(f"Output: {args.output}")
    print(f"Start frame: {args.start_frame if args.start_frame is not None else 'auto (first frame)'}")
    if args.end_frame:
        print(f"End frame: {args.end_frame}")
    print()
    
    if args.validate_only:
        print("Validating input format...")
        from .segmentation import SegmentationLoader
        loader = SegmentationLoader()
        if loader.validate_cell_filter_output(args.input, yaml_path):
            print("✓ Input format is valid")
            sys.exit(0)
        else:
            print("✗ Input format validation failed")
            sys.exit(1)
    
    try:
        # Build tracking params
        tracking_params = {'search_radius': args.search_radius}
        if args.tracking_config:
            tracking_params['config'] = args.tracking_config

        # Run analysis
        results = analyze_cell_filter_data(
            npy_path=args.input,
            yaml_path=yaml_path,
            output_dir=args.output,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            tracking_params=tracking_params,
            adjacency_params={'method': args.adjacency_method}
        )
        
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        print(f"Total frames processed: {results['total_frames']}")
        print(f"Frame range: {results['frame_range'][0]} - {results['frame_range'][1]}")
        print(f"T1 edge weight range: {results['t1_weight_range'][0]:.2f} - {results['t1_weight_range'][1]:.2f}")
        print(f"T1 events detected: {results['t1_events_detected']}")
        print()
        print("Output files:")
        for file_type, path in results['output_files'].items():
            print(f"  {file_type}: {path}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()