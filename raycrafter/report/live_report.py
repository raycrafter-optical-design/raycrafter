"""Keeping one report up to date with the latest design (#68).

``LiveReport`` decides *when* to build; how a report is built is given to it (in the app,
``raycrafter.report.report_process``), and showing it is the business of
``raycrafter.report.live_server``. It has no Qt code.

One thread of its own does the building, so there is never more than one report in the
making, and they start at least MIN_INTERVAL_S apart. After the first, a report is built
only while a page is watching.

How the latest design is never missed
-------------------------------------
Every design handed to ``set_design`` that differs from the one before gets the next
*revision* number, and every report carries the revision of the design it shows. The report
is out of date exactly when the two numbers differ. There is no "needs building" flag that a
finishing build could clear: if the design changes while a report is being built, that report
is published with its older number, the numbers still differ, and another is built. Each
build takes the newest design at the moment it starts, after any waiting, so a burst of
changes costs one report and ends with the last of them.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace

from raycrafter.core.state import OpticalState, state_to_dict
from raycrafter.report.html_report import message_html

MIN_INTERVAL_S = 10.0  # reports start at least this far apart


class LiveReport:
    """The latest report of a design, rebuilt in the background as the design changes.

    All methods may be called from any thread.

    Parameters
    ----------
    build : callable
        ``build(state_data)`` returns the report (HTML) of a design given as
        ``state_to_dict`` data. It is called on this object's thread, one call at a time.
    min_interval_s : float
        The least time between the starts of two builds.
    """

    def __init__(
        self, build: Callable[[dict], str], min_interval_s: float = MIN_INTERVAL_S
    ) -> None:
        self._build = build
        self._min_interval_s = min_interval_s
        self._changed = threading.Condition()  # guards everything below
        self._design: dict | None = None
        self._design_revision = 0  # of the latest design; 0: none yet
        self._report = message_html("Building the report…")
        self._report_revision = 0  # of the design the report shows; 0: not built yet
        self._watcher_count = 0
        self._stopping = False
        self._thread = threading.Thread(target=self._build_when_wanted, daemon=True)
        self._thread.start()

    def set_design(self, state: OpticalState) -> None:
        """Say what the design now is. A copy is taken, so the caller may go on editing."""
        # The record of an imported file is large and not part of the report.
        design = state_to_dict(replace(state, imported_from=None))
        with self._changed:
            if design != self._design:
                self._design = design
                self._design_revision += 1
                self._changed.notify_all()

    def latest(self) -> tuple[int, str]:
        """The newest report and the revision it shows (0: still being built for the first time)."""
        with self._changed:
            return self._report_revision, self._report

    @property
    def is_watched(self) -> bool:
        """True while someone is inside ``watching``."""
        with self._changed:
            return self._watcher_count > 0

    @contextmanager
    def watching(self) -> Iterator[None]:
        """While inside this, a page wants the report kept up to date."""
        with self._changed:
            self._watcher_count += 1
            self._changed.notify_all()
        try:
            yield
        finally:
            with self._changed:
                self._watcher_count -= 1

    def wait_for_report_other_than(self, revision: int, timeout_s: float) -> bool:
        """Wait until the report's revision is not ``revision``; False if time ran out first."""
        with self._changed:
            return self._changed.wait_for(
                lambda: self._report_revision != revision or self._stopping, timeout=timeout_s
            )

    def stop(self) -> None:
        """Build nothing more and release anyone waiting. A build under way is left to end."""
        with self._changed:
            self._stopping = True
            self._changed.notify_all()

    def _is_wanted(self) -> bool:
        """True if a report should be built (call with the lock held).

        It must be out of date, and either watched or the first one.
        """
        out_of_date = self._report_revision != self._design_revision
        return out_of_date and (self._watcher_count > 0 or self._report_revision == 0)

    def _build_when_wanted(self) -> None:
        """The building thread: wait until a report is wanted and due, build it, publish it."""
        earliest_start = 0.0  # on the time.monotonic clock
        while True:
            with self._changed:
                self._changed.wait_for(lambda: self._stopping or self._is_wanted())
                if self._stopping:
                    return
                too_soon_s = earliest_start - time.monotonic()
                if too_soon_s > 0:
                    self._changed.wait_for(lambda: self._stopping, timeout=too_soon_s)
                    continue  # look again, and only then take the design, which may have moved on
                revision, design = self._design_revision, self._design

            earliest_start = time.monotonic() + self._min_interval_s
            assert design is not None  # a revision above 0 comes with a design
            try:
                report = self._build(design)
            except Exception as error:  # shown on the page; the next change is tried afresh
                report = message_html(f"Could not generate the report: {error}")

            with self._changed:
                if self._stopping:
                    return
                self._report, self._report_revision = report, revision
                self._changed.notify_all()
