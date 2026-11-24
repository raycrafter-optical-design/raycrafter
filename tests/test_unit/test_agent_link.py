"""The link between a RayCrafter window and the MCP server (no Qt needed to test it)."""

import json

import anyio
import pytest

from raycrafter.core.model import OpticalModel
from raycrafter.mcp_server.link import LinkClient, LinkServer, LinkUnavailable
from raycrafter.mcp_server.server import build_server
from raycrafter.mcp_server.tools import DesignTools


@pytest.fixture
def endpoint_path(tmp_path):
    return tmp_path / "link" / "agent_link.json"


def make_server(handler, endpoint_path):
    server = LinkServer(handler, endpoint_path)
    server.start()
    return server


def test_a_call_gets_the_handlers_result_and_bytes_survive(endpoint_path):
    def handler(tool_name, arguments):
        if tool_name == "picture":
            return b"\x89PNG-bytes"
        return {"tool": tool_name, "arguments": arguments}

    server = make_server(handler, endpoint_path)
    try:
        client = LinkClient(endpoint_path)
        assert client.call("add_lens", {"part_number": "X"}) == {
            "tool": "add_lens",
            "arguments": {"part_number": "X"},
        }
        assert client.call("picture", {}) == b"\x89PNG-bytes"
    finally:
        server.stop()


def test_a_value_error_in_the_handler_reaches_the_caller_with_its_message(endpoint_path):
    def handler(tool_name, arguments):
        raise ValueError("There is no lens 5")

    server = make_server(handler, endpoint_path)
    try:
        with pytest.raises(ValueError, match="There is no lens 5"):
            LinkClient(endpoint_path).call("move_lens", {})
    finally:
        server.stop()


def test_a_bug_in_the_handler_is_reported_not_hidden(endpoint_path):
    def handler(tool_name, arguments):
        raise KeyError("oops")

    server = make_server(handler, endpoint_path)
    try:
        with pytest.raises(ValueError, match="RayCrafter failed: KeyError"):
            LinkClient(endpoint_path).call("anything", {})
    finally:
        server.stop()


def test_the_endpoint_file_is_private_and_removed_on_stop(endpoint_path):
    server = make_server(lambda tool_name, arguments: {}, endpoint_path)
    assert endpoint_path.stat().st_mode & 0o077 == 0  # no access for group or others
    assert json.loads(endpoint_path.read_text())["token"] == server.token
    server.stop()
    assert not endpoint_path.exists()


def test_no_window_means_unavailable(endpoint_path):
    with pytest.raises(LinkUnavailable):
        LinkClient(endpoint_path).call("get_design", {})


def test_a_stale_endpoint_file_means_unavailable(endpoint_path):
    server = make_server(lambda tool_name, arguments: {}, endpoint_path)
    stale_text = endpoint_path.read_text()
    server.stop()
    endpoint_path.write_text(stale_text)  # the window has gone but its file is left behind
    with pytest.raises(LinkUnavailable):
        LinkClient(endpoint_path).call("get_design", {})


def test_a_wrong_token_is_refused(endpoint_path):
    server = make_server(lambda tool_name, arguments: {"secret": 1}, endpoint_path)
    try:
        endpoint = json.loads(endpoint_path.read_text())
        endpoint["token"] = "not-the-token"
        endpoint_path.write_text(json.dumps(endpoint))
        with pytest.raises(LinkUnavailable):
            LinkClient(endpoint_path).call("get_design", {})
    finally:
        server.stop()


def call_tool(server, name, arguments):
    result = anyio.run(server.call_tool, name, arguments)
    return json.loads(result.content[0].text)


def test_mcp_tools_act_on_the_window_when_one_is_open(endpoint_path):
    window_model = OpticalModel()
    window_tools = DesignTools(window_model)
    link = make_server(
        lambda name, arguments: getattr(window_tools, name)(**arguments), endpoint_path
    )
    try:
        headless_model = OpticalModel()
        server = build_server(headless_model, LinkClient(endpoint_path))
        design = call_tool(server, "add_lens", {"part_number": "AC254-050-A"})

        assert design["connected_to"] == "window"
        assert len(window_model.state.lenses) == 2
        assert len(headless_model.state.lenses) == 1  # untouched
    finally:
        link.stop()


def test_mcp_tools_fall_back_to_the_headless_model_without_a_window(endpoint_path):
    headless_model = OpticalModel()
    server = build_server(headless_model, LinkClient(endpoint_path))
    design = call_tool(server, "add_lens", {"part_number": "AC254-050-A"})

    assert design["connected_to"] == "headless"
    assert len(headless_model.state.lenses) == 2


def test_a_picture_comes_back_through_the_window(endpoint_path):
    window_tools = DesignTools(OpticalModel())
    link = make_server(
        lambda name, arguments: getattr(window_tools, name)(**arguments), endpoint_path
    )
    try:
        server = build_server(OpticalModel(), LinkClient(endpoint_path))
        result = anyio.run(server.call_tool, "get_plot", {"kind": "mtf"})
        assert result.content[0].type == "image"
    finally:
        link.stop()


def test_a_window_that_closes_mid_call_gives_a_clear_error(endpoint_path):
    import socket
    import threading

    # A stand-in "window" that accepts the connection, then hangs up without replying.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    endpoint_path.parent.mkdir(parents=True, exist_ok=True)
    endpoint_path.write_text(
        json.dumps({"port": listener.getsockname()[1], "token": "t", "pid": 0})
    )

    def hang_up():
        connection, _address = listener.accept()
        connection.close()

    threading.Thread(target=hang_up, daemon=True).start()
    try:
        with pytest.raises(ValueError, match="closed before it answered"):
            LinkClient(endpoint_path).call("get_design", {})
    finally:
        listener.close()
