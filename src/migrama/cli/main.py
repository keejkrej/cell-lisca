"""Unified CLI for all migrama modules."""

import logging
import sys
from pathlib import Path
from typing import cast

import typer

# Create main app
app = typer.Typer(help="Migrama: A comprehensive toolkit for micropatterned timelapse microscopy analysis")


@app.command()
def pattern(
    patterns: str = typer.Option(..., "--patterns", "-p", help="Path to patterns ND2 file"),
    output: str = typer.Option("./patterns.csv", "--output", "-o", help="Output CSV file path"),
    fov: int | None = typer.Option(None, "--fov", help="Process only this FOV (default: all FOVs)"),
    debug: bool = typer.Option(False, "--debug"),
):
    """Detect micropatterns and save bounding boxes to CSV.

    Output CSV format: cell,fov,x,y,w,h
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s - %(name)s - %(message)s")

    from ..core.pattern import PatternDetector

    detector = PatternDetector(patterns_path=patterns)

    if fov is not None:
        records = detector.detect_fov(fov)
        typer.echo(f"Detected {len(records)} patterns in FOV {fov}")
    else:
        records = detector.detect_all()
        typer.echo(f"Detected {len(records)} patterns across {detector.n_fovs} FOVs")

    detector.save_csv(records, output)
    typer.echo(f"Saved to: {output}")


@app.command()
def analyze(
    cells: str = typer.Option(..., "--cells", "-c", help="Path to cells ND2 file"),
    csv: str = typer.Option(..., "--csv", help="Path to patterns CSV file"),
    output: str = typer.Option("./analysis.csv", "--output", "-o", help="Output CSV file path"),
    nuclei_channel: int = typer.Option(1, "--nuclei-channel", help="Channel index for nuclei"),
    n_cells: int = typer.Option(4, "--n-cells", help="Target number of cells per pattern"),
    min_size: int = typer.Option(15, "--min-size", help="Minimum object size for Cellpose"),
    debug: bool = typer.Option(False, "--debug"),
):
    """Analyze cell counts and output t0/t1 ranges."""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s - %(name)s - %(message)s")

    from ..analyze import Analyzer

    analyzer = Analyzer(
        cells_path=cells,
        csv_path=csv,
        nuclei_channel=nuclei_channel,
        n_cells=n_cells,
        min_size=min_size,
    )
    records = analyzer.analyze(output)
    typer.echo(f"Saved {len(records)} records to {output}")


@app.command()
def extract(
    cells: str = typer.Option(..., "--cells", "-c", help="Path to cells ND2 file"),
    csv: str = typer.Option(..., "--csv", help="Path to analysis CSV file"),
    output: str = typer.Option("./extracted.h5", "--output", "-o", help="Output H5 file path"),
    nuclei_channel: int = typer.Option(1, "--nuclei-channel", help="Channel index for nuclei"),
    cell_channel: int = typer.Option(0, "--cell-channel", help="Channel index for cell bodies"),
    min_frames: int = typer.Option(1, "--min-frames", help="Minimum frames per sequence"),
    debug: bool = typer.Option(False, "--debug"),
):
    """Extract sequences with segmentation and tracking."""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s - %(name)s - %(message)s")

    from ..extract import Extractor

    extractor = Extractor(
        cells_path=cells,
        analysis_csv=csv,
        output_path=output,
        nuclei_channel=nuclei_channel,
        cell_channel=cell_channel,
    )
    sequences = extractor.extract(min_frames=min_frames)
    typer.echo(f"Saved {sequences} sequences to {output}")


@app.command()
def graph(
    input: str = typer.Option(..., "--input", "-i", help="Path to H5 file with segmentation data"),
    output: str = typer.Option(..., "--output", "-o", help="Output directory for analysis results"),
    fov: int = typer.Option(..., "--fov", help="FOV index"),
    pattern: int = typer.Option(..., "--pattern", help="Pattern index"),
    sequence: int = typer.Option(..., "--sequence", help="Sequence index"),
    start_frame: int | None = typer.Option(None, "--start-frame", "-s", help="Starting frame"),
    end_frame: int | None = typer.Option(None, "--end-frame", "-e", help="Ending frame (exclusive)"),
    search_radius: float = typer.Option(100.0, "--search-radius", help="Max search radius for tracking"),
    debug: bool = typer.Option(False, "--debug"),
):
    """Create region adjacency graphs and analyze T1 transitions."""
    import os
    import tempfile

    import numpy as np
    import yaml as yaml_module

    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s - %(name)s - %(message)s")

    if not Path(input).exists():
        typer.echo(f"Error: Input file does not exist: {input}", err=True)
        raise typer.Exit(1)

    from ..graph.h5_loader import H5SegmentationLoader
    from ..graph.pipeline import analyze_cell_filter_data

    loader = H5SegmentationLoader()

    typer.echo(f"Loading sequence: FOV {fov}, Pattern {pattern}, Sequence {sequence}")
    loaded_data = loader.load_cell_filter_data(input, fov, pattern, sequence, None)
    data = cast(np.ndarray, loaded_data["data"])
    segmentation_masks = cast(np.ndarray, loaded_data["segmentation_masks"])

    # Create temporary NPY file for pipeline compatibility
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        combined_data = np.concatenate([data, segmentation_masks[:, np.newaxis, :, :]], axis=1)
        np.save(tmp.name, combined_data)
        tmp_npy_path = tmp.name

    tmp_yaml_path = None
    if loaded_data["metadata"]:
        tmp_yaml_path = tmp_npy_path.replace(".npy", ".yaml")
        with open(tmp_yaml_path, "w") as f:
            yaml_module.dump(loaded_data["metadata"], f)

    try:
        results = analyze_cell_filter_data(
            npy_path=tmp_npy_path,
            yaml_path=tmp_yaml_path,
            output_dir=output,
            start_frame=start_frame,
            end_frame=end_frame,
            tracking_params={"search_radius": search_radius},
        )

        typer.echo(f"\nAnalysis complete: {results['total_frames']} frames, {results['t1_events_detected']} T1 events")
        for file_type, path in results["output_files"].items():
            typer.echo(f"  {file_type}: {path}")

    finally:
        os.unlink(tmp_npy_path)
        if tmp_yaml_path and os.path.exists(tmp_yaml_path):
            os.unlink(tmp_yaml_path)


@app.command()
def tension(
    mask: str = typer.Option(..., "--mask", help="Path to segmentation mask .npy file"),
    output: str | None = typer.Option(None, "--output", "-o", help="Path to save the VMSI model (pickle)"),
    is_labelled: bool = typer.Option(True, "--labelled", help="Mask already labelled"),
    optimiser: str = typer.Option("nlopt", "--optimiser", help="Optimiser for VMSI (nlopt or matlab)"),
    verbose: bool = typer.Option(False, "--verbose"),
):
    """Run TensionMap VMSI analysis on a segmentation mask."""
    import pickle

    import numpy as np

    from ..tension.integration import run_tensionmap_analysis

    data = np.load(mask, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data_dict = data.item()
        if isinstance(data_dict, dict) and "segmentation_mask" in data_dict:
            mask_arr = data_dict["segmentation_mask"]
        else:
            mask_arr = data
    elif isinstance(data, dict) and "segmentation_mask" in data:
        mask_arr = data["segmentation_mask"]
    else:
        mask_arr = data

    mask_arr = np.asarray(mask_arr)

    model = run_tensionmap_analysis(
        mask_arr,
        is_labelled=is_labelled,
        optimiser=optimiser,
        verbose=verbose,
    )

    if output:
        with open(output, "wb") as f:
            pickle.dump(model, f)
        typer.echo(f"VMSI model saved to {output}")
    else:
        typer.echo("VMSI analysis completed (model not saved)")


@app.command()
def viewer():
    """Launch the interactive NPY viewer."""
    from PySide6.QtWidgets import QApplication

    from ..viewer.ui.main_window import MainWindow

    qt_app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(qt_app.exec())


def main():
    """Main entry point for migrama CLI."""
    app()


if __name__ == "__main__":
    main()
