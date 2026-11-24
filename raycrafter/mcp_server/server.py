"""The MCP server: ``DesignTools`` published as tools over the Model Context Protocol.

Run ``raycrafter-mcp`` (or ``python -m raycrafter.mcp_server.server``) and an MCP client such
as the Claude desktop app can then start it over stdio and call the tools. The server owns its
own ``OpticalModel``; there is no window. Nothing may be printed to standard output, because
that is the channel the protocol itself uses.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image

from raycrafter.core.model import OpticalModel
from raycrafter.mcp_server.tools import DesignTools

SERVER_INSTRUCTIONS = (
    "RayCrafter designs optical systems from off-the-shelf lenses on an optical axis. Light "
    "enters from the left (small z) along the axis, passes the lenses in order of z, and "
    "reaches the image plane. Start with get_design, find lenses with search_lenses, edit "
    "with the lens and field tools, then read the result with get_summary and the other "
    "analyses. Distances are in mm and angles in degrees; lens and field indices start at 0."
)

# The DesignTools methods that are offered as tools, in the order they are listed.
TOOL_NAMES = [
    "get_design",
    "search_lenses",
    "get_lens",
    "get_summary",
    "get_mtf",
    "get_ray_fan",
    "get_through_focus",
    "get_field_curvature",
    "get_plot",
    "add_lens",
    "insert_lens",
    "add_paraxial_lens",
    "move_lens",
    "flip_lens",
    "delete_lens",
    "make_plossl",
    "split_plossl",
    "set_field",
    "add_field",
    "remove_fields",
    "select_field",
    "set_image_z",
    "autofocus",
    "new_design",
    "load_zmx",
    "save_zmx",
]


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


def _plot_tool(tools: DesignTools) -> Callable:
    """The get_plot tool: DesignTools gives PNG bytes, MCP wants them wrapped as an Image."""

    def get_plot(kind: str) -> Image:
        """A picture (PNG) of one analysis, as seen in the GUI.

        kind is one of: ray_diagram, spot_diagram, through_focus, mtf, ray_fan,
        field_curvature. MTF, ray fan and through-focus show the selected field.
        """
        return Image(data=tools.get_plot(kind), format="png")

    return get_plot


def build_server(model: OpticalModel | None = None) -> MCPServer:
    """An MCP server whose tools act on ``model`` (a fresh default bench if omitted)."""
    tools = DesignTools(model if model is not None else OpticalModel())
    server = MCPServer("raycrafter", instructions=SERVER_INSTRUCTIONS)
    for tool_name in TOOL_NAMES:
        if tool_name == "get_plot":
            tool = _plot_tool(tools)
        else:
            tool = getattr(tools, tool_name)
        server.add_tool(_report_value_errors(tool), name=tool_name)
    return server


def main() -> None:
    """Entry point of ``raycrafter-mcp``: serve over stdio until the client disconnects."""
    build_server().run("stdio")


if __name__ == "__main__":
    main()
