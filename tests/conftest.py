"""Shared test fixtures."""

import pytest

from raycrafter.core import OpticalModel
from raycrafter.mcp_server.link import ENDPOINT_FILE_VARIABLE


@pytest.fixture(scope="session", autouse=True)
def hide_open_windows(tmp_path_factory):
    """Keep every test away from a RayCrafter window that happens to be open.

    The MCP server and ``Bench.connect`` look for a window through its endpoint file. Pointing
    them at a file that does not exist stops a test run editing the design someone is working on.
    """
    absent_endpoint_path = tmp_path_factory.mktemp("no_window") / "agent_link.json"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(ENDPOINT_FILE_VARIABLE, str(absent_endpoint_path))
        yield


@pytest.fixture
def focused_model() -> OpticalModel:
    """A bench with one ACT508-200-A (f = 200 mm), a 10 mm collimated beam, autofocused."""
    model = OpticalModel()
    model.autofocus()
    return model
