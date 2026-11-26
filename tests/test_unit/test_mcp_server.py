"""The MCP server publishes the agent tools and reports errors as tool errors."""

import json
from pathlib import Path

import anyio
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from raycrafter.mcp_server.link import LinkClient
from raycrafter.mcp_server.server import TOOL_NAMES, build_server


def test_the_tests_cannot_reach_an_open_window():
    """A server made as these tests make one must not find the user's own window."""
    users_endpoint_path = Path.home() / ".raycrafter" / "agent_link.json"
    assert LinkClient().endpoint_path != users_endpoint_path


def test_every_tool_is_listed_with_a_description():
    tools = anyio.run(build_server().list_tools)
    assert [tool.name for tool in tools] == TOOL_NAMES
    assert all(tool.description for tool in tools)


def test_calling_a_tool_changes_the_servers_model():
    async def add_lens_then_read_design():
        server = build_server()
        await server.call_tool("add_lens", {"part_number": "AC254-050-A"})
        return await server.call_tool("get_design", {})

    result = anyio.run(add_lens_then_read_design)
    design = json.loads(result.content[0].text)
    assert len(design["lenses"]) == 2


def test_a_bad_request_comes_back_as_a_tool_error_with_the_reason():
    async def ask_for_a_missing_field():
        return await build_server().call_tool("select_field", {"field_index": 7})

    with pytest.raises(ToolError, match="There is no field 7"):
        anyio.run(ask_for_a_missing_field)


def test_get_plot_returns_an_image():
    result = anyio.run(build_server().call_tool, "get_plot", {"kind": "spot_diagram"})
    assert result.content[0].type == "image" and result.content[0].mime_type == "image/png"
