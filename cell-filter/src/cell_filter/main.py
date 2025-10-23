"""Unified entry point for cell-filter submodules."""

import logging
from typing import Optional

import typer

app = typer.Typer(help="Cell-filter: Process micropatterned timelapse microscopy images")


@app.command()
def pattern(
    patterns: str = typer.Option("data/20250806_patterns_after.nd2", "--patterns"),
    cells: str = typer.Option("data/20250806_MDCK_timelapse_crop_fov0004.nd2", "--cells"),
    nuclei_channel: int = typer.Option(1, "--nuclei-channel"),
    fov: int = typer.Option(0, "--fov"),
    fov_all: bool = typer.Option(False, "--fov-all"),
    output: Optional[str] = typer.Option(None, "--output"),
    debug: bool = typer.Option(False, "--debug"),
):
    """Detect and annotate micropatterns."""
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(levelname)s - %(name)s - %(message)s"
    )
    
    from cell_filter.pattern import Patterner
    
    patterner = Patterner(
        patterns_path=patterns,
        cells_path=cells,
        nuclei_channel=nuclei_channel,
    )
    if fov_all:
        for fov_idx in range(patterner.n_fovs):
            patterner.plot_view(fov_idx, output)
    else:
        patterner.plot_view(fov, output)
    patterner.close()


@app.command()
def filter(
    patterns: str = typer.Option("data/20250806_patterns_after.nd2", "--patterns"),
    cells: str = typer.Option("data/20250806_MDCK_timelapse_crop_fov0004.nd2", "--cells"),
    nuclei_channel: int = typer.Option(1, "--nuclei-channel"),
    output: str = typer.Option("data/analysis/", "--output"),
    n_cells: int = typer.Option(4, "--n-cells"),
    debug: bool = typer.Option(False, "--debug"),
    all: bool = typer.Option(False, "--all"),
    range: str = typer.Option("0:1", "--range"),
    min_size: int = typer.Option(15, "--min-size"),
):
    """Filter patterns based on cell count."""
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(levelname)s - %(name)s - %(message)s"
    )
    
    from cell_filter.filter import Filterer
    
    filter_processor = Filterer(
        patterns_path=patterns,
        cells_path=cells,
        output_folder=output,
        n_cells=n_cells,
        nuclei_channel=nuclei_channel,
    )
    if all:
        filter_processor.process_fovs(
            0, filter_processor.cropper.n_fovs - 1, min_size=min_size
        )
    else:
        fov_range = list(map(int, range.split(":")))
        filter_processor.process_fovs(
            fov_range[0], fov_range[1], min_size=min_size
        )


@app.command()
def extract(
    patterns: str = typer.Option("data/20250806_patterns_after.nd2", "--patterns"),
    cells: str = typer.Option("data/20250806_MDCK_timelapse_crop_fov0004.nd2", "--cells"),
    nuclei_channel: int = typer.Option(1, "--nuclei-channel"),
    filter_results: str = typer.Option("data/analysis/", "--filter-results"),
    output: str = typer.Option("data/analysis/", "--output"),
    min_frames: int = typer.Option(20, "--min-frames"),
    max_gap: int = typer.Option(6, "--max-gap", help="Maximum frame gap before splitting sequences"),
    debug: bool = typer.Option(False, "--debug"),
):
    """Extract sequences from filtered patterns."""
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s - %(name)s - %(message)s")
    
    from cell_filter.extract import Extractor
    
    extractor = Extractor(
        patterns_path=patterns,
        cells_path=cells,
        output_folder=output,
        nuclei_channel=nuclei_channel,
    )
    extractor.extract(
        filter_results_dir=filter_results,
        min_frames=min_frames,
        max_gap=max_gap
    )


def main():
    """Main entry point for cell-filter."""
    app()


if __name__ == "__main__":
    app()
