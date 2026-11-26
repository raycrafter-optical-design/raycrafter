"""The live report as a web page that updates itself, served to this computer only (#68).

``LiveReportServer`` is a small HTTP server on 127.0.0.1 that shows what a ``LiveReport``
(``raycrafter.report.live_report``) holds. It never builds a report itself, so every request
is answered at once. It has no Qt code.

Its addresses all start with a random token, so that other web pages cannot read the design:

- ``/<token>/``: the newest report, with a script (``assets/live.js``) that watches for a
  newer one and then reloads the page;
- ``/<token>/revision?after=N``: answers with the report's revision number as soon as it is
  not N, or after WAIT_S; the page asks again at once (a "long poll"). A page waiting here is
  what makes the report worth rebuilding;
- ``/<token>/download``: the report as a file that needs no server, to share.
"""

from __future__ import annotations

import select
import socket
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from secrets import token_urlsafe
from urllib.parse import parse_qs, urlsplit

from raycrafter.report.live_report import MIN_INTERVAL_S, LiveReport
from raycrafter.report.viewer_3d import read_asset

LOCAL_ADDRESS = "127.0.0.1"
WAIT_S = 25.0  # how long a request for the revision is held while there is nothing new
CHECK_S = 0.2  # while holding it, how often to look whether the browser has gone away

LIVE_BAR = (
    '<div class="live-bar"><span id="live-notice">'
    f"Updates as the design changes, at most every {MIN_INTERVAL_S:.0f} s</span>"
    '<a href="download">Save a copy to share</a></div>'
)


def live_page(report: str, revision: int) -> str:
    """The report with the bar and script that keep a served page up to date."""
    script = f'<script data-revision="{revision}">{read_asset("live.js")}</script>'
    before, end_of_body, after = report.rpartition("</body>")
    return f"{before}{LIVE_BAR}\n{script}\n{end_of_body}{after}"


class LiveReportServer:
    """Serves a live report to browsers on this computer, for as long as it is running.

    Parameters
    ----------
    live_report : LiveReport
        What to serve.
    """

    def __init__(self, live_report: LiveReport) -> None:
        self.live_report = live_report
        self.token = token_urlsafe(16)
        self._stopping = threading.Event()
        self._http_server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        """The address of the live report; only while the server is running."""
        if self._http_server is None:
            raise RuntimeError("The live report server is not running")
        port = self._http_server.server_address[1]
        return f"http://{LOCAL_ADDRESS}:{port}/{self.token}/"

    def start(self) -> None:
        """Start listening, in a background thread."""
        live_server = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 (name fixed by http.server)
                try:
                    live_server._answer(self)
                except ConnectionError:
                    pass  # the browser closed the tab before the answer was written

            def log_message(self, format, *args) -> None:  # silence the per-request log lines
                pass

        self._stopping.clear()
        self._http_server = ThreadingHTTPServer((LOCAL_ADDRESS, 0), RequestHandler)
        self._http_server.daemon_threads = True
        threading.Thread(target=self._http_server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        """Stop listening."""
        self._stopping.set()
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None

    def _answer(self, request: BaseHTTPRequestHandler) -> None:
        """Check the token and send what the address asks for."""
        address = urlsplit(request.path)
        prefix = f"/{self.token}/"
        if not address.path.startswith(prefix):
            request.send_error(404)
            return
        page_name = address.path.removeprefix(prefix)

        if page_name == "revision":
            after = parse_qs(address.query).get("after", [""])[0]
            self._wait_for_news(int(after) if after.isdigit() else -1, request.connection)
            _send(request, "text/plain", str(self.live_report.latest()[0]))
            return
        revision, report = self.live_report.latest()
        if page_name == "":
            _send(request, "text/html", live_page(report, revision))
        elif page_name == "download" and revision > 0:
            file_name = f"raycrafter_report_{datetime.now():%Y%m%d_%H%M%S}.html"
            _send(request, "text/html", report, download_name=file_name)
        else:
            request.send_error(404)

    def _wait_for_news(self, shown_revision: int, connection: socket.socket) -> None:
        """Wait for a report other than ``shown_revision``, until the browser goes or WAIT_S."""
        deadline = time.monotonic() + WAIT_S
        with self.live_report.watching():
            while time.monotonic() < deadline and not self._stopping.is_set():
                if self.live_report.wait_for_report_other_than(shown_revision, CHECK_S):
                    return
                if _closed_by_browser(connection):
                    return


def _closed_by_browser(connection: socket.socket) -> bool:
    """True if the other end has closed a connection on which it is waiting for an answer."""
    readable, _, _ = select.select([connection], [], [], 0)
    if not readable:
        return False
    try:
        return connection.recv(1, socket.MSG_PEEK) == b""  # nothing left to read: closed
    except ConnectionError:
        return True


def _send(
    request: BaseHTTPRequestHandler, content_type: str, text: str, download_name: str | None = None
) -> None:
    """Send a successful answer that the browser must not keep a copy of."""
    content = text.encode("utf-8")
    request.send_response(200)
    request.send_header("Content-Type", f"{content_type}; charset=utf-8")
    request.send_header("Content-Length", str(len(content)))
    request.send_header("Cache-Control", "no-store")
    if download_name is not None:
        request.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
    request.end_headers()
    request.wfile.write(content)
