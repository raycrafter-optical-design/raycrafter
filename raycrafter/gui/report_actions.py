"""The window's side of "Generate report": a report in the web browser that stays up to date.

The first press starts a ``LiveReport`` (``raycrafter.report.live_report``), which builds
reports in another process, and a server on this computer that shows them
(``raycrafter.report.live_server``), and opens its page in the browser. From then on the
window hands it the design after each change, once nothing is being dragged; all the window
spends on that is a copy of the design. Pressing the button again while the page is open
opens no second tab.
"""

from __future__ import annotations

import webbrowser
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QMessageBox

from raycrafter.report.live_report import LiveReport
from raycrafter.report.live_server import LiveReportServer
from raycrafter.report.report_process import ReportProcess

if TYPE_CHECKING:
    from raycrafter.gui.main_window import MainWindow

STATUS_MESSAGE_MS = 8000


class ReportActions:
    """Opens the window's report in the browser and keeps it supplied with the design.

    Parameters
    ----------
    window : MainWindow
        The window whose design the report shows.
    """

    def __init__(self, window: MainWindow) -> None:
        self.window = window
        # All three are made by the first "Generate report", and are None until then.
        self.report_process: ReportProcess | None = None
        self.live_report: LiveReport | None = None
        self.server: LiveReportServer | None = None

    def open_report(self) -> None:
        """Show the report in the web browser, unless a page of it is already open."""
        if self.live_report is None:
            try:
                self._start()
            except OSError as error:  # e.g. no local port can be opened
                QMessageBox.critical(
                    self.window, "Report failed", f"Could not start the report:\n{error}"
                )
                return
        assert self.live_report is not None and self.server is not None
        self.live_report.set_design(self.window.model.state)
        if self.live_report.is_watched:
            message = "The report open in your web browser updates itself"
        else:
            webbrowser.open(self.server.url)
            message = f"Report opened at {self.server.url}"
        self.window.statusBar().showMessage(message, STATUS_MESSAGE_MS)

    def design_settled(self) -> None:
        """Hand over the design after a change; does nothing if no report was ever opened."""
        if self.live_report is not None:
            self.live_report.set_design(self.window.model.state)

    def stop(self) -> None:
        """Stop serving and building; open pages say that they are no longer updated."""
        if self.server is not None:
            self.server.stop()
        if self.live_report is not None:
            self.live_report.stop()
        if self.report_process is not None:
            self.report_process.close()
        self.server = self.live_report = self.report_process = None

    def _start(self) -> None:
        """Make the report's builder and server."""
        report_process = ReportProcess()  # starts its process when first used
        live_report = LiveReport(report_process)
        server = LiveReportServer(live_report)
        try:
            server.start()
        except OSError:
            live_report.stop()
            raise
        self.report_process, self.live_report, self.server = report_process, live_report, server
