"""Tell the user when numba is compiling, because that can take minutes and prints nothing."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from numba.core import event


class CompileNotice(event.Listener):
    """Print one message the first time numba starts compiling a function.

    numba broadcasts a ``"numba:compile"`` event only when it has to compile. A function whose
    compiled code is found in the on-disk cache does not trigger it, so the message appears on
    the first run (or after a numba/Optiland upgrade) and not on later runs.
    """

    MESSAGE = "Compiling the ray-tracing code (numba). Please wait..."

    def __init__(self) -> None:
        self.has_printed = False

    def on_start(self, event_: event.Event) -> None:
        """Print the message, but only for the first compilation."""
        if not self.has_printed:
            self.has_printed = True
            print(self.MESSAGE, flush=True)

    def on_end(self, event_: event.Event) -> None:
        """Nothing to do when a compilation finishes."""


@contextmanager
def announce_numba_compilation() -> Iterator[None]:
    """Within this ``with`` block, print a message if numba starts compiling."""
    with event.install_listener("numba:compile", CompileNotice()):
        yield
