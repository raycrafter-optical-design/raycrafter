"""Lets an AI agent, through the MCP server, control the design shown in the window.

``AgentLink`` runs a ``LinkServer`` (``raycrafter.mcp_server.link``) that receives tool calls on
a background thread. Qt widgets and the model may only be touched from the GUI thread, so each
call is handed to the GUI thread with a signal; the server thread waits there for the answer.
After a call that edits the design, the window redraws, so the user watches the agent work.
"""

from __future__ import annotations

import threading
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QLabel

from raycrafter.mcp_server.link import LinkServer
from raycrafter.mcp_server.tools import EDITING_TOOL_NAMES, TOOL_NAMES, DesignTools

REQUEST_TIMEOUT_S = 100.0  # shorter than the client's wait, so the client gets a clear answer
STATUS_MESSAGE_MS = 3000


class _Request:
    """One tool call waiting for the GUI thread."""

    def __init__(self, tool_name: str, arguments: dict) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.finished = threading.Event()
        self.abandoned = False  # set if the caller gave up, so the GUI thread skips it
        self.result: object = None
        self.error: Exception | None = None


class AgentLink(QObject):
    """Receives agent tool calls and runs them on the window's model.

    Parameters
    ----------
    window : MainWindow
        The window whose design the agent controls.
    endpoint_path : Path, optional
        Where to write the endpoint file; the usual place if omitted.
    """

    _request_received = pyqtSignal(object)

    def __init__(self, window, endpoint_path: Path | None = None) -> None:
        super().__init__(window)
        self.window = window
        self.tools = DesignTools(window.model)
        self.server = LinkServer(self._handle_on_server_thread, endpoint_path)
        self.status_label = QLabel("AI agent link on")
        # Emitted from a server thread and received here (the GUI thread), so Qt queues it.
        self._request_received.connect(self._run_on_gui_thread)

    def start(self) -> None:
        """Start listening and show that the link is on in the status bar."""
        self.server.start()
        self.window.statusBar().addPermanentWidget(self.status_label)

    def stop(self) -> None:
        """Stop listening and remove the endpoint file."""
        self.server.stop()
        self.window.statusBar().removeWidget(self.status_label)

    def _handle_on_server_thread(self, tool_name: str, arguments: dict) -> object:
        """Hand a call to the GUI thread and wait for its result (runs on a server thread)."""
        if tool_name not in TOOL_NAMES:
            raise ValueError(f"Unknown tool {tool_name!r}")
        request = _Request(tool_name, arguments)
        self._request_received.emit(request)
        if not request.finished.wait(REQUEST_TIMEOUT_S):
            request.abandoned = True
            raise ValueError("The RayCrafter window is busy and did not run the request")
        if request.error is not None:
            raise request.error
        return request.result

    def _run_on_gui_thread(self, request: _Request) -> None:
        """Run one tool on the model and, if it edited the design, redraw the window."""
        if request.abandoned:
            return
        try:
            tool = getattr(self.tools, request.tool_name)
            request.result = tool(**request.arguments)
            if request.tool_name in EDITING_TOOL_NAMES:
                self.window.show_new_design()
            self.window.statusBar().showMessage(f"Agent: {request.tool_name}", STATUS_MESSAGE_MS)
        except Exception as error:  # reported to the agent; the window carries on
            request.error = error
        finally:
            request.finished.set()
