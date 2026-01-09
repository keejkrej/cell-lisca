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
def analyze(
    input: str = typer.Option(..., "--input", "-i", help="Path to cell-filter NPY file containing timelapse data with segmentation channel"),
    output: str = typer.Option(..., "--output", "-o", help="Output directory for analysis results"),
    yaml: Optional[str] = typer.Option(None, "--yaml", "-y", help="Path to corresponding YAML metadata file (optional, auto-detected if not provided)"),
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
    
    # Auto-detect YAML file if not provided
    yaml_path = yaml
    if yaml_path is None:
        input_path = Path(input)
        yaml_path = str(input_path.with_suffix('.yaml'))
        if not Path(yaml_path).exists():
            yaml_path = None
    
    typer.echo("Cell Grapher CLI")
    typer.echo(f"Input NPY: {input}")
    if yaml_path:
        typer.echo(f"Input YAML: {yaml_path}")
    else:
        typer.echo("Input YAML: Not found, using default channel settings")
    typer.echo(f"Output: {output}")
    typer.echo(f"Start frame: {start_frame if start_frame is not None else 'auto (first frame)'}")
    if end_frame:
        typer.echo(f"End frame: {end_frame}")
    typer.echo()
    
    if validate_only:
        typer.echo("Validating input format...")
        from .segmentation import SegmentationLoader
        loader = SegmentationLoader()
        if loader.validate_cell_filter_output(input, yaml_path):
            typer.echo("✓ Input format is valid")
            raise typer.Exit(0)
        else:
            typer.echo("✗ Input format validation failed", err=True)
            raise typer.Exit(1)
    
    try:
        # Build tracking params
        tracking_params = {'search_radius': search_radius}
        if tracking_config:
            tracking_params['config'] = tracking_config

        # Run analysis
        results = analyze_cell_filter_data(
            npy_path=input,
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
        raise typer.Exit(1)


def main():
    """Main CLI entry point for cell-grapher."""
    app()


if __name__ == "__main__":
    app()