"""Integration module for running TensionMap VMSI on Cell‑LISCA data.

The function ``run_tensionmap_analysis`` expects a segmentation mask as a NumPy
array (e.g. the ``segmentation_mask`` produced by the cell‑grapher pipeline) and
returns the VMSI model instance.  Additional keyword arguments are passed
directly to ``run_VMSI``.
"""

from typing import Any
import numpy as np

try:
    from VMSI import run_VMSI  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Failed to import TensionMap VMSI. Ensure the TensionMap package is installed."
    ) from exc


def run_tensionmap_analysis(
    mask: np.ndarray,
    *,
    is_labelled: bool = True,
    holes_mask: np.ndarray | None = None,
    tile: bool = False,
    cells_per_tile: int = 150,
    verbose: bool = False,
    overlap: float = 0.3,
    optimiser: str = "nlopt",
    **kwargs: Any,
) -> Any:
    """Run VMSI tension inference on a segmentation mask.

    Parameters
    ----------
    mask:
        Segmentation mask where each cell has a unique integer label.
    is_labelled:
        ``True`` because the mask already contains integer labels.
    holes_mask:
        Optional mask indicating holes in the tissue.
    tile, cells_per_tile, overlap, optimiser:
        Passed straight through to ``run_VMSI``.
    verbose:
        Enable verbose output from VMSI.
    **kwargs:
        Additional arguments forwarded to ``run_VMSI``.

    Returns
    -------
    Any
        The VMSI model object containing stress tensors and morphometrics.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a NumPy ndarray")
    if mask.dtype.kind not in {"i", "u"}:
        mask = mask.astype(int)

    vmsi_model = run_VMSI(
        mask,
        is_labelled=is_labelled,
        holes_mask=holes_mask,
        tile=tile,
        cells_per_tile=cells_per_tile,
        verbose=verbose,
        overlap=overlap,
        optimiser=optimiser,
        **kwargs,
    )
    return vmsi_model
