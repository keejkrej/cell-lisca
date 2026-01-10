"""Generic progress signaling for migrama pipelines."""

from dataclasses import dataclass

from psygnal import Signal


@dataclass
class ProgressEvent:
    """Generic progress event with state and iterator context.

    Attributes
    ----------
    state : str
        The current stage or state (e.g., "segmentation", "tracking", "nuclei").
    iterator : str
        The iterator context (e.g., "frame", "file").
    current : int
        Current progress position (0-indexed or 1-indexed, depends on emitter).
    total : int
        Total number of items in the iterator.
    """

    state: str
    iterator: str
    current: int
    total: int


class ProgressEmitter:
    """A simple emitter for progress events using psygnal signals."""

    _signal = Signal(ProgressEvent)

    @property
    def progress(self):
        """Get the progress signal for connecting callbacks."""
        return self._signal

    def emit(self, state: str, iterator: str, current: int, total: int) -> None:
        """Emit a progress event.

        Parameters
        ----------
        state : str
            The current stage or state.
        iterator : str
            The iterator context.
        current : int
            Current progress position.
        total : int
            Total number of items.
        """
        self._signal.emit(ProgressEvent(state=state, iterator=iterator, current=current, total=total))
