"""Unified entry point for cell-filter submodules."""

import logging

import typer

app = typer.Typer(help="Cell-filter: Analyze and extract micropatterned timelapse microscopy data\n\nNote: Pattern detection is handled by the separate cell-pattern package.")


def parse_range(range_str: str) -> tuple[int, int]:
    """Parse a range string like '0:10' into (start, end) tuple."""
    parts = range_str.split(':')
    if len(parts) != 2:
        raise typer.BadParameter(f"Range must be in format 'start:end', got '{range_str}'")
    try:
        start = int(parts[0])
        end = int(parts[1])
        return start, end
    except ValueError:
        raise typer.BadParameter(f"Range values must be integers, got '{range_str}'")


@app.command()
def analysis(
    cells: str = typer.Option(..., "--cells", help="Path to cells ND2 file"),
    h5: str = typer.Option(..., "--h5", help="Path to H5 file with bounding boxes (analysis will be appended)"),
    nuclei_channel: int = typer.Option(1, "--nuclei-channel", help="Channel index for nuclei"),
    range_str: str = typer.Option(..., "--range", help="FOV range to process (e.g., '0:10')"),
    min_size: int = typer.Option(15, "--min-size", help="Minimum object size for nuclei detection"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
):
    """Analyze cell counts for all patterns across all frames.
    
    Counts cells in each pattern for every frame in the specified FOV range.
    Results are appended to the H5 file for later processing by the extract command.
    FOVs not found in the H5 bounding boxes will be skipped with a warning.
    """
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(levelname)s - %(name)s - %(message)s"
    )
    
    # Parse range
    start_fov, end_fov = parse_range(range_str)
    
    from cell_filter.analysis import Analyzer
    
    analyzer = Analyzer(
        cells_path=cells,
        h5_path=h5,
        nuclei_channel=nuclei_channel,
    )
    
    summary = analyzer.analyze(
        start_fov=start_fov,
        end_fov=end_fov,
        min_size=min_size
    )
    
    typer.echo(f"Analysis complete: {summary['total_records']} records from {len(summary['processed_fovs'])} FOVs")
    if summary['skipped_fovs']:
        typer.echo(f"Skipped FOVs (not in H5): {summary['skipped_fovs']}")


@app.command()
def extract(
    cells: str = typer.Option(..., "--cells", help="Path to cells ND2 file"),
    h5: str = typer.Option(..., "--h5", help="Path to H5 file with bounding boxes and analysis data"),
    nuclei_channel: int = typer.Option(1, "--nuclei-channel", help="Channel index for nuclei"),
    n_cells: int = typer.Option(4, "--n-cells", help="Target number of cells per pattern"),
    tolerance_gap: int = typer.Option(6, "--tolerance-gap", help="Max consecutive frames with wrong cell count"),
    min_frames: int = typer.Option(20, "--min-frames", help="Minimum frames for valid sequences"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
):
    """Extract sequences from analysis results based on cell count criteria.
    
    Reads analysis data from H5, finds sequences matching the cell count criteria,
    and extracts cropped image data with segmentation to the same H5 file.
    """
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s - %(name)s - %(message)s")
    
    from cell_filter.extract import Extractor
    
    extractor = Extractor(
        cells_path=cells,
        h5_path=h5,
        nuclei_channel=nuclei_channel,
    )
    
    summary = extractor.extract(
        n_cells=n_cells,
        tolerance_gap=tolerance_gap,
        min_frames=min_frames
    )
    
    typer.echo(f"Extraction complete: {summary['total_sequences']} sequences extracted")


def main():
    """Main entry point for cell-filter."""
    app()


if __name__ == "__main__":
    app()
