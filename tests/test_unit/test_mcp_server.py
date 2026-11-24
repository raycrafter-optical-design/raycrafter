"""The MCP server publishes the agent tools and reports errors as tool errors."""

import json

import anyio
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from raycrafter.mcp_server.server import TOOL_NAMES, build_server


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
