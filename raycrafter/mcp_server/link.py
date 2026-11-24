"""A private link between the RayCrafter window and the MCP server (no Qt, no MCP library).

The Claude app starts ``raycrafter-mcp`` as a separate process, so the server cannot reach the
window's design directly. Instead the window runs a ``LinkServer``: a tiny HTTP server on
127.0.0.1 only. It writes its port and a random token to an *endpoint file* in the user's home
folder (readable only by the user). A ``LinkClient`` in the MCP server reads that file and
sends each tool call to the window, which answers with the result.

Protocol: ``POST /call`` with the header ``X-RayCrafter-Token`` and a JSON body
``{"tool": name, "arguments": {...}}``. The reply is ``{"ok": true, "result": ...}`` or
``{"ok": false, "error": message}``. Binary results (PNG pictures) travel as
``{"bytes_base64": "..."}``.
"""

from __future__ import annotations

import base64
import hmac
import http.client
import json
import os
import secrets
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ENDPOINT_FILE_VARIABLE = "RAYCRAFTER_LINK_FILE"  # overrides the endpoint file's location
TOKEN_HEADER = "X-RayCrafter-Token"
LOCAL_ADDRESS = "127.0.0.1"
CONNECT_TIMEOUT_S = 2.0  # a window that is running answers a connection at once
CALL_TIMEOUT_S = 120.0  # a slow analysis (e.g. MTF on a slow machine) may take a while


class LinkUnavailable(Exception):
    """No RayCrafter window is listening (none is running, or its endpoint file is stale)."""


def default_endpoint_path() -> Path:
    """Where the endpoint file lives: ``$RAYCRAFTER_LINK_FILE``, else ~/.raycrafter/agent_link.json."""
    override = os.environ.get(ENDPOINT_FILE_VARIABLE)
    return Path(override) if override else Path.home() / ".raycrafter" / "agent_link.json"


def _encode_result(result: object) -> object:
    """Make a tool's result JSON-safe: bytes become {"bytes_base64": ...}."""
    if isinstance(result, bytes):
        return {"bytes_base64": base64.b64encode(result).decode("ascii")}
    return result


def _decode_result(result: object) -> object:
    """The reverse of ``_encode_result``."""
    if isinstance(result, dict) and set(result) == {"bytes_base64"}:
        return base64.b64decode(result["bytes_base64"])
    return result


class LinkServer:
    """Serves tool calls to a ``handler`` on a local port, for as long as it is running.

    Parameters
    ----------
    handler : callable
        ``handler(tool_name, arguments)`` returns the result (JSON-friendly, or bytes). It may
        raise ValueError for a request that makes no sense; the message goes back to the caller.
        It is called on a server thread, one thread per request.
    endpoint_path : Path, optional
        Where to write the endpoint file; ``default_endpoint_path()`` if omitted.
    """

    def __init__(
        self, handler: Callable[[str, dict], object], endpoint_path: Path | None = None
    ) -> None:
        self.handler = handler
        self.endpoint_path = endpoint_path if endpoint_path is not None else default_endpoint_path()
        self.token = secrets.token_hex(16)
        self._http_server: ThreadingHTTPServer | None = None

    def start(self) -> None:
        """Start listening (in a background thread) and write the endpoint file."""
        link_server = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 (name fixed by http.server)
                link_server._answer(self)

            def log_message(self, format, *args) -> None:  # silence the per-request log lines
                pass

        self._http_server = ThreadingHTTPServer((LOCAL_ADDRESS, 0), RequestHandler)
        self._http_server.daemon_threads = True
        threading.Thread(target=self._http_server.serve_forever, daemon=True).start()

        port = self._http_server.server_address[1]
        self.endpoint_path.parent.mkdir(parents=True, exist_ok=True)
        endpoint = {"port": port, "token": self.token, "pid": os.getpid()}
        # Create the file readable by the user only, before the token is written into it.
        descriptor = os.open(self.endpoint_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as endpoint_file:
            json.dump(endpoint, endpoint_file)

    def stop(self) -> None:
        """Stop listening and remove the endpoint file (if it is still ours)."""
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None
        try:
            if json.loads(self.endpoint_path.read_text()).get("token") == self.token:
                self.endpoint_path.unlink()
        except (OSError, ValueError):
            pass  # already gone, or replaced by a newer window's file

    def _answer(self, request: BaseHTTPRequestHandler) -> None:
        """Check the token, run the requested tool and write the JSON reply."""
        sent_token = request.headers.get(TOKEN_HEADER, "")
        if request.path != "/call" or not hmac.compare_digest(sent_token, self.token):
            request.send_error(403)
            return
        try:
            body = json.loads(request.rfile.read(int(request.headers.get("Content-Length", 0))))
            result = self.handler(body["tool"], body.get("arguments", {}))
            reply = {"ok": True, "result": _encode_result(result)}
        except ValueError as error:  # includes bad JSON; the message is for the agent
            reply = {"ok": False, "error": str(error)}
        except Exception as error:  # a bug: say so, rather than leave the caller waiting
            reply = {"ok": False, "error": f"RayCrafter failed: {type(error).__name__}: {error}"}

        payload = json.dumps(reply).encode()
        request.send_response(200)
        request.send_header("Content-Type", "application/json")
        request.send_header("Content-Length", str(len(payload)))
        request.end_headers()
        request.wfile.write(payload)


class LinkClient:
    """Sends tool calls to a RayCrafter window, if one is running.

    Parameters
    ----------
    endpoint_path : Path, optional
        The window's endpoint file; ``default_endpoint_path()`` if omitted.
    """

    def __init__(self, endpoint_path: Path | None = None) -> None:
        self.endpoint_path = endpoint_path if endpoint_path is not None else default_endpoint_path()

    def call(self, tool_name: str, arguments: dict) -> object:
        """Run a tool in the window and return its result (bytes for a picture).

        Raises
        ------
        LinkUnavailable
            If no window answers. The call has then not been run, so it is safe to run it
            elsewhere.
        ValueError
            If the window ran the tool and it reported a problem.
        """
        try:
            endpoint = json.loads(self.endpoint_path.read_text())
            connection = http.client.HTTPConnection(
                LOCAL_ADDRESS, endpoint["port"], timeout=CONNECT_TIMEOUT_S
            )
            connection.connect()
        except (OSError, ValueError, KeyError) as error:
            raise LinkUnavailable(str(error)) from error

        try:
            connection.sock.settimeout(CALL_TIMEOUT_S)  # connected; now allow for a slow answer
            body = json.dumps({"tool": tool_name, "arguments": arguments})
            connection.request("POST", "/call", body, {TOKEN_HEADER: endpoint["token"]})
            response = connection.getresponse()
            if response.status == 403:  # something else owns the port: the file is stale
                raise LinkUnavailable("the endpoint file is out of date")
            reply = json.loads(response.read())
        except TimeoutError as error:
            raise ValueError(
                "The RayCrafter window did not answer in time (it may be busy); "
                "the request may still be carried out."
            ) from error
        except (ConnectionError, http.client.HTTPException, ValueError) as error:
            # The connection worked but the reply never came whole: the window was closed
            # while it was working. (ValueError here is a reply that is not valid JSON.)
            raise ValueError(
                "The RayCrafter window closed before it answered; the request may not have "
                "been carried out. Check with get_design."
            ) from error
        finally:
            connection.close()

        if not reply["ok"]:
            raise ValueError(reply["error"])
        return _decode_result(reply["result"])
