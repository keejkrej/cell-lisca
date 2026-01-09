"""
Main entry point for cell-grapher module.

This allows running: python -m cell_grapher
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from .pipeline import analyze_cell_filter_data

# Configure app
app = typer.Typer(help="Cell-grapher: Create region adjacency graphs and analyze T1 transitions using pre-computed segmentations from cell-filter.")

@app.command()
def list_sequences(
    input: str = typer.Option(..., "--input", "-i", help="Path to cell-filter H5 file"),
):
    """List all available sequences in an H5 file."""
    # Validate input file exists
    if not Path(input).exists():
        typer.echo(f"Error: H5 file does not exist: {input}", err=True)
        raise typer.Exit(1)
    
    # Check if it's an H5 file
    input_path = Path(input)
    if input_path.suffix.lower() not in ['.h5', '.hdf5']:
        typer.echo(f"Error: Expected H5 file, got {input_path.suffix}", err=True)
        raise typer.Exit(1)
    
    try:
        from .h5_loader import H5SegmentationLoader
        loader = H5SegmentationLoader()
        
        typer.echo(f"Listing sequences in: {input}")
        sequences = loader.list_sequences(input)
        
        if not sequences:
            typer.echo("No sequences found in H5 file")
            return
        
        typer.echo(f"\nFound {len(sequences)} sequences:")
        typer.echo()
        
        # Group by FOV for better readability
        by_fov = {}
        for seq in sequences:
            fov = seq['fov_idx']
            if fov not in by_fov:
                by_fov[fov] = []
            by_fov[fov].append(seq)
        
        for fov in sorted(by_fov.keys()):
            typer.echo(f"FOV {fov}:")
            for seq in by_fov[fov]:
                typer.echo(f"  Pattern {seq['pattern_idx']}, Sequence {seq['seq_idx']}")
        
    except Exception as e:
        typer.echo(f"Error listing sequences: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def analyze(
    input: str = typer.Option(..., "--input", "-i", help="Path to cell-filter NPY or H5 file containing timelapse data with segmentation channel"),
    output: str = typer.Option(..., "--output", "-o", help="Output directory for analysis results"),
    yaml: Optional[str] = typer.Option(None, "--yaml", "-y", help="Path to corresponding YAML metadata file (optional, auto-detected if not provided)"),
    fov_idx: Optional[int] = typer.Option(None, "--fov", help="FOV index (required for H5 files)"),
    pattern_idx: Optional[int] = typer.Option(None, "--pattern", help="Pattern index (required for H5 files)"),
    seq_idx: Optional[int] = typer.Option(None, "--sequence", help="Sequence index (required for H5 files)"),
    start_frame: Optional[int] = typer.Option(None, "--start-frame", "-s", help="Starting frame number (processes from first frame if not specified)"),
    end_frame: Optional[int] = typer.Option(None, "--end-frame", "-e", help="Ending frame number (exclusive, processes all frames if not specified)"),
    search_radius: float = typer.Option(100.0, "--search-radius", help="Maximum search radius for cell tracking in pixels"),
    tracking_config: Optional[str] = typer.Option(None, "--tracking-config", help="Path to btrack config JSON file (optional, uses default if not specified)"),
    adjacency_method: str = typer.Option("boundary_length", "--adjacency-method", help="Method for building adjacency graphs"),
    validate_only: bool = typer.Option(False, "--validate-only", help="Only validate input format, don't run analysis"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
):
    """Create region adjacency graphs and analyze T1 transitions."""
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(levelname)s - %(name)s - %(message)s"
    )
    
    # Validate input file exists
    if not Path(input).exists():
        typer.echo(f"Error: Input file does not exist: {input}", err=True)
        raise typer.Exit(1)
    
    # Determine file type and validate accordingly
    input_path = Path(input)
    is_h5 = input_path.suffix.lower() in ['.h5', '.hdf5']
    
    if is_h5:
        # For H5 files, we need fov, pattern, and sequence indices
        if fov_idx is None or pattern_idx is None or seq_idx is None:
            typer.echo("Error: For H5 files, --fov, --pattern, and --sequence are required", err=True)
            
            # List available sequences if possible
            try:
                from .h5_loader import H5SegmentationLoader
                loader = H5SegmentationLoader()
                sequences = loader.list_sequences(input)
                if sequences:
                    typer.echo("\nAvailable sequences:")
                    for seq in sequences[:10]:  # Show first 10
                        typer.echo(f"  FOV {seq['fov_idx']}, Pattern {seq['pattern_idx']}, Sequence {seq['seq_idx']}")
                    if len(sequences) > 10:
                        typer.echo(f"  ... and {len(sequences) - 10} more")
            except Exception as e:
                logger.debug(f"Could not list sequences: {e}")
            
            raise typer.Exit(1)
        
        # Use H5 loader
        from .h5_loader import H5SegmentationLoader
        loader = H5SegmentationLoader()
        
        if validate_only:
            typer.echo("Validating H5 file format...")
            if loader.validate_cell_filter_output(input, yaml):
                typer.echo("✓ H5 file format is valid")
                raise typer.Exit(0)
            else:
                typer.echo("✗ H5 file format validation failed", err=True)
                raise typer.Exit(1)
        
        # Load data from H5
        typer.echo(f"Loading sequence from H5 file: FOV {fov_idx}, Pattern {pattern_idx}, Sequence {seq_idx}")
        loaded_data = loader.load_cell_filter_data(input, fov_idx, pattern_idx, seq_idx, yaml)
        
        # Create a temporary NPY-like structure for the pipeline
        # This maintains compatibility with existing pipeline code
        npy_path = input  # Keep original path for reference
        
    else:
        # Original NPY file handling
        if validate_only:
            typer.echo("Validating NPY file format...")
            from .segmentation import SegmentationLoader
            loader = SegmentationLoader()
            if loader.validate_cell_filter_output(input, yaml):
                typer.echo("✓ NPY file format is valid")
                raise typer.Exit(0)
            else:
                typer.echo("✗ NPY file format validation failed", err=True)
                raise typer.Exit(1)
        
        # Load data using original NPY loader
        from .segmentation import SegmentationLoader
        loader = SegmentationLoader()
        loaded_data = loader.load_cell_filter_data(input, yaml)
        npy_path = input
    
    # Auto-detect YAML file if not provided
    yaml_path = yaml
    if yaml_path is None:
        yaml_path = str(input_path.with_suffix('.yaml'))
        if not Path(yaml_path).exists():
            yaml_path = None
    
    typer.echo("Cell Grapher CLI")
    typer.echo(f"Input: {input} ({'H5' if is_h5 else 'NPY'} format)")
    if yaml_path:
        typer.echo(f"Input YAML: {yaml_path}")
    else:
        typer.echo("Input YAML: Not found, using default channel settings")
    typer.echo(f"Output: {output}")
    typer.echo(f"Start frame: {start_frame if start_frame is not None else 'auto (first frame)'}")
    if end_frame:
        typer.echo(f"End frame: {end_frame}")
    typer.echo()
    
    try:
        # Build tracking params
        tracking_params = {'search_radius': search_radius}
        if tracking_config:
            tracking_params['config'] = tracking_config

        # Run analysis - pass loaded_data directly for H5 files
        if is_h5:
            # For H5 files, we need to adapt the pipeline to accept pre-loaded data
            # This is a temporary solution - ideally the pipeline would be refactored
            # to accept data directly rather than file paths
            import tempfile
            import os
            
            # Create temporary NPY file for compatibility
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
                # Combine image data and segmentation masks
                combined_data = np.concatenate([
                    loaded_data['data'],
                    loaded_data['segmentation_masks'][:, np.newaxis, :, :]
                ], axis=1)
                np.save(tmp.name, combined_data)
                tmp_npy_path = tmp.name
            
            # Save temporary YAML if needed
            tmp_yaml_path = None
            if loaded_data['metadata']:
                tmp_yaml_path = tmp_npy_path.replace('.npy', '.yaml')
                with open(tmp_yaml_path, 'w') as f:
                    yaml.dump(loaded_data['metadata'], f)
            
            try:
                results = analyze_cell_filter_data(
                    npy_path=tmp_npy_path,
                    yaml_path=tmp_yaml_path,
                    output_dir=output,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    tracking_params=tracking_params,
                    adjacency_params={'method': adjacency_method}
                )
            finally:
                # Clean up temporary files
                os.unlink(tmp_npy_path)
                if tmp_yaml_path and os.path.exists(tmp_yaml_path):
                    os.unlink(tmp_yaml_path)
        else:
            # Original NPY pipeline
            results = analyze_cell_filter_data(
                npy_path=npy_path,
                yaml_path=yaml_path,
                output_dir=output,
                start_frame=start_frame,
                end_frame=end_frame,
                tracking_params=tracking_params,
                adjacency_params={'method': adjacency_method}
            )
        
        # Print summary
        typer.echo("\n" + "="*50)
        typer.echo("ANALYSIS COMPLETE")
        typer.echo("="*50)
        typer.echo(f"Total frames processed: {results['total_frames']}")
        typer.echo(f"Frame range: {results['frame_range'][0]} - {results['frame_range'][1]}")
        typer.echo(f"T1 edge weight range: {results['t1_weight_range'][0]:.2f} - {results['t1_weight_range'][1]:.2f}")
        typer.echo(f"T1 events detected: {results['t1_events_detected']}")
        typer.echo()
        typer.echo("Output files:")
        for file_type, path in results['output_files'].items():
            typer.echo(f"  {file_type}: {path}")
        
    except Exception as e:
        typer.echo(f"Error during analysis: {e}", err=True)
        if debug:
            import traceback
            typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(1)


def main():
    """Main CLI entry point for cell-grapher."""
    app()


if __name__ == "__main__":
    app()