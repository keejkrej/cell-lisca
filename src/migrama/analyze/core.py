"""Cell count analysis for migrama analyze."""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from ..core import CellposeCounter
from ..core.cell_source import CellFovSource
from ..core.pattern import CellCropper

logger = logging.getLogger(__name__)


@dataclass
class AnalysisRecord:
    """Analysis result for a single pattern."""

    cell: int
    fov: int
    x: int
    y: int
    w: int
    h: int
    t0: int
    t1: int


class Analyzer:
    """Analyze cell counts for patterns across frames."""

    def __init__(
        self,
        source: CellFovSource,
        csv_path: str,
        nuclei_channel: int = 1,
        n_cells: int = 4,
        min_size: int = 15,
    ) -> None:
        """Initialize Analyzer.

        Parameters
        ----------
        source : CellFovSource
            Source of cell timelapse data (ND2 or TIFF)
        csv_path : str
            Path to patterns CSV file
        nuclei_channel : int
            Channel index for nuclei
        n_cells : int
            Target number of cells per pattern
        min_size : int
            Minimum object size for Cellpose
        """
        self.source = source
        self.csv_path = Path(csv_path).resolve()
        self.nuclei_channel = nuclei_channel
        self.n_cells = n_cells
        self.min_size = min_size

        self.cropper = CellCropper(
            source=source,
            bboxes_csv=str(self.csv_path),
            nuclei_channel=nuclei_channel,
        )
        self.counter = CellposeCounter()

    def analyze(self, output_path: str) -> list[AnalysisRecord]:
        """Run analysis and write CSV output.

        Parameters
        ----------
        output_path : str
            Output CSV file path

        Returns
        -------
        list[AnalysisRecord]
            Analysis records for each pattern
        """
        records: list[AnalysisRecord] = []

        for fov_idx in sorted(self.cropper.bboxes_by_fov.keys()):
            bboxes = self.cropper.get_bboxes(fov_idx)
            if not bboxes:
                continue

            logger.info(f"Analyzing FOV {fov_idx} with {len(bboxes)} patterns")
            counts_per_pattern = [[] for _ in bboxes]

            for frame_idx in range(self.cropper.n_frames):
                nuclei_list = [
                    self.cropper.extract_nuclei(fov_idx, frame_idx, cell_idx) for cell_idx in range(len(bboxes))
                ]
                counts = self.counter.count_nuclei(nuclei_list, min_size=self.min_size)
                for cell_idx, count in enumerate(counts):
                    counts_per_pattern[cell_idx].append(count)

            for cell_idx, bbox in enumerate(bboxes):
                t0, t1 = self._find_longest_run(counts_per_pattern[cell_idx], self.n_cells)
                records.append(
                    AnalysisRecord(
                        cell=bbox.cell,
                        fov=bbox.fov,
                        x=bbox.x,
                        y=bbox.y,
                        w=bbox.w,
                        h=bbox.h,
                        t0=t0,
                        t1=t1,
                    )
                )

        self._write_csv(output_path, records)
        return records

    @staticmethod
    def _find_longest_run(counts: list[int], target: int) -> tuple[int, int]:
        """Find longest contiguous run of target counts."""
        best_start = -1
        best_end = -1
        best_len = 0
        current_start: int | None = None

        for idx, count in enumerate(counts):
            if count == target:
                if current_start is None:
                    current_start = idx
            elif current_start is not None:
                current_end = idx - 1
                length = current_end - current_start + 1
                if length > best_len:
                    best_len = length
                    best_start = current_start
                    best_end = current_end
                current_start = None

        if current_start is not None:
            current_end = len(counts) - 1
            length = current_end - current_start + 1
            if length > best_len:
                best_len = length
                best_start = current_start
                best_end = current_end

        if best_len == 0:
            return -1, -1

        return best_start, best_end

    @staticmethod
    def _write_csv(output_path: str | Path, records: list[AnalysisRecord]) -> None:
        """Write analysis records to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["cell", "fov", "x", "y", "w", "h", "t0", "t1"])
            for record in records:
                writer.writerow(
                    [
                        record.cell,
                        record.fov,
                        record.x,
                        record.y,
                        record.w,
                        record.h,
                        record.t0,
                        record.t1,
                    ]
                )

        logger.info(f"Saved analysis CSV to {output_path}")
