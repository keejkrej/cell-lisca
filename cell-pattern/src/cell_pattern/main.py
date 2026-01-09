"""Entry point for cell-pattern module."""

import logging
from typing import Optional

import typer

app = typer.Typer(help="Cell-pattern: Detect and annotate micropatterns in microscopy images")


@app.command()
def detect(
    patterns: str = typer.Option("data/20250806_patterns_after.nd2", "--patterns"),
    cells: str = typer.Option("data/20250806_MDCK_timelapse_crop_fov0004.nd2", "--cells"),
    nuclei_channel: int = typer.Option(1, "--nuclei-channel"),
    fov: int = typer.Option(0, "--fov"),
    fov_all: bool = typer.Option(False, "--fov-all"),
    output: Optional[str] = typer.Option(None, "--output"),
    debug: bool = typer.Option(False, "--debug"),
):
    """Detect and annotate micropatterns (visualization only)."""
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(levelname)s - %(name)s - %(message)s"
    )
    
    from cell_core import Patterner
    
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
def extract(
    patterns: str = typer.Option("data/20250806_patterns_after.nd2", "--patterns"),
    cells: str = typer.Option("data/20250806_MDCK_timelapse_crop_fov0004.nd2", "--cells"),
    nuclei_channel: int = typer.Option(1, "--nuclei-channel"),
    output: str = typer.Option("./bounding_boxes.h5", "--output"),
    debug: bool = typer.Option(False, "--debug"),
):
    """Extract pattern bounding boxes for all FOVs and save to a single HDF5 file."""
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(levelname)s - %(name)s - %(message)s"
    )
    
    from cell_core import Patterner
    
    patterner = Patterner(
        patterns_path=patterns,
        cells_path=cells,
        nuclei_channel=nuclei_channel,
    )
    
    try:
        typer.echo(f"Extracting bounding boxes for all {patterner.n_fovs} FOVs...")
        patterner.process_all_fovs_bounding_boxes()
        typer.echo(f"✓ Saved bounding boxes to: {output}")
    finally:
        patterner.close()


def main():
    """Main entry point for cell-pattern."""
    app()


if __name__ == "__main__":
    app()
