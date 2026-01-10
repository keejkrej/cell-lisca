import pickle
from pathlib import Path

import numpy as np
import typer

from .integration import run_tensionmap_analysis

app = typer.Typer(add_completion=False)

@app.command()
def run(
    mask: Path = typer.Option(..., "--mask", help="Path to segmentation mask .npy file"),
    output: Path = typer.Option(None, "--output", help="Path to save the VMSI model (pickle)"),
    is_labelled: bool = typer.Option(True, "--labelled", help="Mask already labelled"),
    optimiser: str = typer.Option("nlopt", "--optimiser", help="Optimiser for VMSI (nlopt or matlab)"),
    verbose: bool = typer.Option(False, "--verbose"),
):
    """Run TensionMap VMSI analysis on a Cell‑LISCA segmentation mask.

    The mask should be a NumPy array saved with ``np.save`` (shape ``(T, H, W)``).
    If ``--output`` is provided the VMSI model is pickled to that location.
    """
    data = np.load(mask, allow_pickle=True)
    if isinstance(data, dict) and "segmentation_mask" in data:
        mask_arr = data["segmentation_mask"]
    else:
        mask_arr = data
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

if __name__ == "__main__":
    app()
