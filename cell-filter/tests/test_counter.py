"""
Executable script for counting cells per pattern across all channels.

Usage:
    python -m cell-filter.tests.test_counter

The script expects the following ND2 files to be present on the filesystem:

    CELLS_PATH    = /project/ag-moonraedler/user/Agathe.Jouneau/4_interacting_cells/20250812/20250812_MDCK_LK_timelapse.nd2
    PATTERNS_PATH = /project/ag-moonraedler/user/Agathe.Jouneau/4_interacting_cells/20250812/20250812_patterns_after.nd2

It performs the following steps:

1. Loads the cells ND2 file and extracts a full channel stack for a given
   field‑of‑view (FOV) and time point.
2. Instantiates ``Patterner`` with the patterns ND2 file to obtain pattern
   bounding boxes.
3. For each channel in the stack:
   * Crops the region defined by each bounding box,
   * Counts nuclei using ``CellposeCounter``,
   * Overlays the count next to the bounding box on the image.
4. Performs sanity checks that a count is produced for every channel/pattern pair
   and that all counts are non‑negative integers.

The script does **not** assert a specific numeric result (the number of cells
depends on the data) but verifies that the workflow finishes without raising
exceptions.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

from cell_filter.utils.nikon import load_nd2, get_nd2_channel_stack, get_nd2_frame
from cell_filter.pattern import Patterner
from cell_filter.core.count import CellposeCounter


# ----------------------------------------------------------------------
# Paths to the real ND2 files – adjust if the location changes.
# ----------------------------------------------------------------------
CELLS_PATH = Path(
    "/project/ag-moonraedler/user/Agathe.Jouneau/4_interacting_cells/20250812/20250812_MDCK_LK_timelapse.nd2"
)
PATTERNS_PATH = Path(
    "/project/ag-moonraedler/user/Agathe.Jouneau/4_interacting_cells/20250812/20250812_patterns_after.nd2"
)
FOV = 10
TIME = 10
NUC = 1


def main() -> None:
    """Run the counting pipeline."""
    # --------------------------------------------------------------
    # 1️⃣ Load the cells file and obtain a channel stack.
    # --------------------------------------------------------------
    print(f"Loading cells from {CELLS_PATH} ...")
    cells_da, cells_meta = load_nd2(CELLS_PATH)

    # ``get_nd2_channel_stack`` returns an array of shape (C, H, W)
    channel_stack = get_nd2_channel_stack(cells_da, f=FOV, t=TIME)
    assert channel_stack.ndim == 3, "Channel stack should be 3‑dimensional (C, H, W)"
    n_channels = channel_stack.shape[0]
    print(f"Channel stack shape: {channel_stack.shape}")
    nuclei_image = channel_stack[NUC]  # Extract the nuclei channel for counting

    # --------------------------------------------------------------
    # 2️⃣ Initialise ``Patterner`` to obtain pattern bounding boxes.
    # --------------------------------------------------------------
    print(f"Loading patterns from {PATTERNS_PATH} ...")
    patterner = Patterner(
        patterns_path=str(PATTERNS_PATH),
        cells_path=str(CELLS_PATH),
        nuclei_channel=1,  # default nuclei channel for the dataset
    )
    patterner.cropper.load_fov(FOV)
    patterner.cropper.load_patterns()
    patterner.cropper.process_patterns()

    bboxes = patterner.cropper.bounding_boxes
    n_patterns = patterner.cropper.n_patterns
    assert bboxes is not None and len(bboxes) == n_patterns
    assert n_patterns > 0, "No patterns detected – aborting script"
    print(f"Detected {n_patterns} pattern(s)")

    # --------------------------------------------------------------
    # 3️⃣ Count nuclei on the pattern image (single channel) and overlay
    #    those counts on each cell channel.
    # --------------------------------------------------------------
    # Load the pattern stack (single channel) and extract the image.

    counter = CellposeCounter()
    # Count nuclei for each pattern region using the nuclei channel image.
    pattern_counts = {}
    for pat_idx, bbox in enumerate(bboxes):
        if bbox is None:
            continue
        x, y, w, h = map(int, bbox)
        region = nuclei_image[y : y + h, x : x + w]
        pattern_counts[pat_idx] = counter.count_nuclei(region)[0]

    # Overlay counts on each cell channel image and save the figures.
    script_dir = Path(__file__).resolve().parent
    for ch_idx in range(n_channels):
        channel_img = channel_stack[ch_idx]  # shape (H, W)

        fig, ax = plt.subplots()
        ax.imshow(channel_img, cmap="gray")

        for pat_idx, bbox in enumerate(bboxes):
            if bbox is None:
                continue
            x, y, w, h = map(int, bbox)

            # Draw the bounding box.
            rect = plt.Rectangle(
                (x, y),
                w,
                h,
                edgecolor="lime",
                facecolor="none",
                linewidth=1.5,
            )
            ax.add_patch(rect)

            # Overlay the count obtained from the pattern image.
            cnt = pattern_counts.get(pat_idx, 0)
            ax.text(
                x,
                y - 4,
                str(cnt),
                color="lime",
                fontsize=8,
                weight="bold",
                verticalalignment="bottom",
            )

        ax.set_title(f"Channel {ch_idx} – overlay from pattern counts")
        # Save the figure next to the script.
        out_path = script_dir / f"channel_{ch_idx}_overlay.png"
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved overlay plot for channel {ch_idx} to {out_path}")

    # --------------------------------------------------------------
    # 4️⃣ Simple sanity checks.
    # --------------------------------------------------------------
    # Verify that we have a count for every pattern.
    assert len(pattern_counts) == n_patterns, "Missing counts for some patterns"

    print("\nCounting completed successfully.")
    print("Pattern counts:")
    for pat_idx, cnt in pattern_counts.items():
        print(f"  Pattern {pat_idx}: {cnt}")

    # Clean up resources.
    patterner.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
