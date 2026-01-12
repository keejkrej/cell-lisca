"""
Lazy FOV source classes for pattern detection.

Provides a uniform interface for loading pattern images from different sources
(ND2 files, TIFF folders) without coupling the detector to specific formats.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tifffile

from ..io.nikon import get_nd2_frame, load_nd2

logger = logging.getLogger(__name__)


class PatternFovSource(ABC):
    """Abstract base class for lazy FOV pattern sources.

    Subclasses provide a uniform interface for iterating over FOV images,
    regardless of the underlying file format (ND2, TIFF, etc.).
    """

    @property
    @abstractmethod
    def n_fovs(self) -> int:
        """Return the number of FOVs in this source."""
        ...

    @abstractmethod
    def iter_fovs(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over FOVs in order, yielding (fov_id, image) tuples.

        Yields
        ------
        tuple[int, np.ndarray]
            (fov_id, image) for each field of view, in sequential order.
            Image is a 2D numpy array.
        """
        ...

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Convenience: make the source itself iterable."""
        return self.iter_fovs()


class Nd2PatternFovSource(PatternFovSource):
    """Lazy ND2 source for pattern images.

    Reads frames on-demand from an ND2 file without loading the entire
    dataset into memory.

    Parameters
    ----------
    nd2_path : str | Path
        Path to the ND2 file.
    channel : int
        Channel index to use (default: 0).
    frame : int
        Frame index to use (default: 0).
    """

    def __init__(
        self,
        nd2_path: str | Path,
        channel: int = 0,
        frame: int = 0,
    ) -> None:
        """Initialize the ND2 source."""
        self.nd2_path = Path(nd2_path).resolve()
        self.channel = channel
        self.frame = frame

        if not self.nd2_path.exists():
            raise FileNotFoundError(f"ND2 file not found: {self.nd2_path}")

        self._xarr, self._metadata = load_nd2(self.nd2_path)

        if self.channel < 0 or self.channel >= self._metadata.n_channels:
            raise ValueError(f"Channel {self.channel} out of range (0-{self._metadata.n_channels - 1})")
        if self.frame < 0 or self.frame >= self._metadata.n_frames:
            raise ValueError(f"Frame {self.frame} out of range (0-{self._metadata.n_frames - 1})")

        logger.info(f"Nd2PatternFovSource: {self.nd2_path.name}, {self._metadata.n_fovs} FOVs")

    @property
    def n_fovs(self) -> int:
        """Return the number of FOVs."""
        return self._metadata.n_fovs

    def iter_fovs(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over FOVs, yielding (fov_id, image) tuples."""
        for fov_id in range(self._metadata.n_fovs):
            frame = get_nd2_frame(self._xarr, fov_id, self.channel, self.frame)
            yield fov_id, frame


class TiffPatternFovSource(PatternFovSource):
    """Lazy TIFF folder source for pattern images.

    Reads pre-averaged TIFF files on-demand, sorted by FOV index.
    The user provides a path containing a FOV index suffix (e.g., `./folder/xxx_0.tif`),
    and the class automatically discovers all matching files.

    Parameters
    ----------
    tiff_pattern : str | Path
        Path to a TIFF file with FOV index suffix, e.g., `./folder/patterns_0.tif`.
        The FOV index is stripped to create a glob pattern for discovering all FOV files.
    """

    def __init__(self, tiff_pattern: str | Path) -> None:
        """Initialize the TIFF source from a pattern path."""
        tiff_path = Path(tiff_pattern).resolve()

        if not tiff_path.exists():
            raise FileNotFoundError(f"TIFF file not found: {tiff_path}")

        if tiff_path.is_dir():
            raise NotADirectoryError(f"Expected a file path, got a directory: {tiff_path}")

        self._tiff_folder = tiff_path.parent
        self._pattern = self._derive_pattern(tiff_path)
        self._tiff_files = self._discover_tiff_files()
        self._n_fovs = len(self._tiff_files)

        if self._n_fovs == 0:
            raise ValueError(f"No TIFF files found matching pattern: {self._pattern}")

        logger.info(f"TiffPatternFovSource: {self._pattern}, {self._n_fovs} FOVs")

    def _derive_pattern(self, tiff_path: Path) -> str:
        """Derive a glob pattern from a specific file path.

        Strips the numeric FOV suffix from the filename to create a pattern.
        E.g., `./folder/patterns_0.tif` -> `./folder/patterns_*.tif`
        """
        name = tiff_path.name
        suffix = tiff_path.suffix

        if not name.endswith(suffix):
            raise ValueError(f"File does not have expected TIFF extension: {tiff_path}")

        base_without_ext = name[: -len(suffix)]

        if not base_without_ext:
            raise ValueError(f"Invalid TIFF filename: {tiff_path}")

        import re

        numeric_suffix = re.search(r"_(\d+)$", base_without_ext)
        if numeric_suffix:
            prefix = base_without_ext[: numeric_suffix.start()]
            pattern_name = f"{prefix}_*{suffix}"
        else:
            pattern_name = f"*{suffix}"

        return str(tiff_path.parent / pattern_name)

    def _discover_tiff_files(self) -> list[tuple[int, Path]]:
        """Discover TIFF files and extract FOV indices from filenames.

        Returns
        -------
        list[tuple[int, Path]]
            List of (fov_id, path) tuples, sorted by fov_id.
        """
        import re

        files = []
        pattern_str = self._pattern
        folder = self._tiff_folder
        suffix = Path(pattern_str).suffix

        for tiff_path in folder.glob(pattern_str):
            name = tiff_path.name

            base_without_ext = name[: -len(suffix)]
            match = re.search(r"_(\d+)$", base_without_ext)
            if match:
                try:
                    fov_id = int(match.group(1))
                    files.append((fov_id, tiff_path))
                except ValueError:
                    logger.warning(f"Could not parse FOV index from: {name}")
            else:
                logger.warning(f"Filename does not match pattern (missing _<num>): {name}")

        files.sort(key=lambda x: x[0])
        return files

    @property
    def n_fovs(self) -> int:
        """Return the number of FOVs."""
        return self._n_fovs

    def iter_fovs(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over FOVs, yielding (fov_id, image) tuples."""
        for fov_id, tiff_path in self._tiff_files:
            image = tifffile.imread(str(tiff_path))
            yield fov_id, image
