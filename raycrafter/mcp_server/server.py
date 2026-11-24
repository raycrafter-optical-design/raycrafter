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


def build_server(model: OpticalModel | None = None) -> MCPServer:
    """An MCP server whose tools act on ``model`` (a fresh default bench if omitted)."""
    tools = DesignTools(model if model is not None else OpticalModel())
    server = MCPServer("raycrafter", instructions=SERVER_INSTRUCTIONS)
    for tool_name in TOOL_NAMES:
        server.add_tool(_report_value_errors(getattr(tools, tool_name)), name=tool_name)
    return server


def main() -> None:
    """Entry point of ``raycrafter-mcp``: serve over stdio until the client disconnects."""
    build_server().run("stdio")


if __name__ == "__main__":
    main()
