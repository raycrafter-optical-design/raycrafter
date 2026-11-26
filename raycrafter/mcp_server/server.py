"""The MCP server: ``DesignTools`` published as tools over the Model Context Protocol.

Run ``raycrafter-mcp`` (or ``python -m raycrafter.mcp_server.server``) and an MCP client such
as the Claude desktop app can then start it over stdio and call the tools. If a RayCrafter
window is open, the tools act on its design (see ``raycrafter.mcp_server.link``); otherwise
they act on the server's own headless ``OpticalModel``. Nothing may be printed to standard
output, because that is the channel the protocol itself uses.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image

from raycrafter.core.model import OpticalModel
from raycrafter.mcp_server.link import LinkClient, LinkUnavailable
from raycrafter.mcp_server.tools import TOOL_NAMES, DesignTools

SERVER_INSTRUCTIONS = (
    "RayCrafter designs optical systems from off-the-shelf lenses on an optical axis. Light "
    "enters from the left (small z) along the axis, passes the lenses in order of z, and "
    "reaches the image plane. Start with get_design, find lenses with search_lenses, edit "
    "with the lens and field tools, then read the result with get_summary and the other "
    "analyses. Distances are in mm and angles in degrees; lens and field indices start at 0. "
    "Every result says in connected_to whether it came from the user's open RayCrafter "
    "window (which shows your changes live) or from a headless design with no window."
)


def _report_value_errors(tool: Callable) -> Callable:
    """Wrap a tool so a ValueError reaches the agent as an ordinary tool error with its message.

    Any other exception is a bug in RayCrafter and is left to the MCP library to report.
    """

    @functools.wraps(tool)
    def wrapper(*args, **kwargs):
        try:
            return tool(*args, **kwargs)
        except ValueError as error:
            raise ToolError(str(error)) from error

    return wrapper


def _through_window_if_open(
    tool_name: str, local_tool: Callable, link_client: LinkClient
) -> Callable:
    """Run a tool in the user's RayCrafter window when one is open, else on the local model.

    The choice is made on every call, and a result that is a dictionary says which was used
    (``connected_to``), so an agent notices if the window is closed part-way through.
    """
    signature = inspect.signature(local_tool)

    @functools.wraps(local_tool)
    def wrapper(*args, **kwargs):
        given_arguments = dict(signature.bind(*args, **kwargs).arguments)
        try:
            result = link_client.call(tool_name, given_arguments)
            connected_to = "window"
        except LinkUnavailable:
            result = local_tool(**given_arguments)
            connected_to = "headless"
        if isinstance(result, dict):
            result = {"connected_to": connected_to, **result}
        return result

    return wrapper


def _plot_tool(get_plot_png: Callable[[str], bytes]) -> Callable:
    """The get_plot tool: a function giving PNG bytes, wrapped as the Image MCP wants."""

    def get_plot(kind: str) -> Image:
        """A picture (PNG) of one analysis, as seen in the GUI.

        kind is one of: ray_diagram, spot_diagram, through_focus, mtf, ray_fan,
        field_curvature, grid_distortion, distortion. MTF, ray fan and through-focus show the
        selected field. The distortion plots need a finite object.
        """
        return Image(data=get_plot_png(kind), format="png")

    return get_plot


def build_server(
    model: OpticalModel | None = None, link_client: LinkClient | None = None
) -> MCPServer:
    """An MCP server whose tools act on the open RayCrafter window, or else on ``model``.

    Parameters
    ----------
    model : OpticalModel, optional
        The headless design used when no window is open (a fresh default bench if omitted).
    link_client : LinkClient, optional
        How to reach the window; the usual endpoint file if omitted.
    """
    tools = DesignTools(model if model is not None else OpticalModel())
    link_client = link_client if link_client is not None else LinkClient()
    server = MCPServer("raycrafter", instructions=SERVER_INSTRUCTIONS)
    for tool_name in TOOL_NAMES:
        tool = _through_window_if_open(tool_name, getattr(tools, tool_name), link_client)
        if tool_name == "get_plot":
            tool = _plot_tool(tool)
        server.add_tool(_report_value_errors(tool), name=tool_name)
    return server


def main() -> None:
    """Entry point of ``raycrafter-mcp``: serve over stdio until the client disconnects."""
    build_server().run("stdio")


if __name__ == "__main__":
    main()
