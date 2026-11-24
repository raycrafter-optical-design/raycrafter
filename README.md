# RayCrafter

[![Python Version](https://img.shields.io/pypi/pyversions/raycrafter.svg)](https://pypi.org/project/raycrafter)
[![PyPI Version](https://img.shields.io/pypi/v/raycrafter.svg)](https://pypi.org/project/raycrafter)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/raycrafter-optical-design/raycrafter/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/raycrafter-optical-design/raycrafter/test_and_deploy.yml?branch=main&label=CI)](https://github.com/raycrafter-optical-design/raycrafter/actions/workflows/test_and_deploy.yml)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![codecov](https://codecov.io/gh/raycrafter-optical-design/raycrafter/branch/main/graph/badge.svg)](https://codecov.io/gh/raycrafter-optical-design/raycrafter)

RayCrafter is an early proof of principle for a lens-level optical design tool. You place
off-the-shelf lenses on an optical axis, drag them around, and see the spot diagram,
through-focus curve, MTF, ray fan and field curvature update as you go. The ray tracing is
done by [Optiland](https://github.com/HarrisonKramer/optiland).

At present it offers the Thorlabs spherical lenses and ideal (paraxial) lenses, and any number
of beams, each with its own width, angle, vergence, offset and wavelength. It can export the
system as a Zemax `.zmx` file and write an HTML report.

**Status:** pre-alpha. Expect anything to change, including the API.


## Install and run

```
pip install raycrafter
raycrafter
```

Or, with [uv](https://docs.astral.sh/uv/): `uvx raycrafter`.


## Use it from an AI agent

RayCrafter includes an [MCP](https://modelcontextprotocol.io) server, so an AI agent such as
Claude can search for lenses, build and edit a design, and read back the optics.

If a RayCrafter window is open, the agent controls that window: you watch its changes happen,
and you can also move lenses yourself and ask the agent about the result. (The window listens
only on your own computer, and writes a private file in `~/.raycrafter` so the server can find
it. Start with `raycrafter --no-agent-link` to turn this off.) If no window is open, the server
works on a design of its own, with no picture. Every result says which one it came from.

**1. Install the MCP extra.**

- With pip: `pip install "raycrafter[mcp]"`. This installs the `raycrafter-mcp` command.
- With [uv](https://docs.astral.sh/uv/): `uv tool install "raycrafter[mcp]"` installs both
  `raycrafter` and `raycrafter-mcp`. Or install nothing and let uv fetch it on demand; see the
  `uvx` setting below.

**2. Tell your MCP client where it is.** In the Claude desktop app, open Settings, then
Developer, then Edit Config. This opens the file `claude_desktop_config.json`, which is at
`~/Library/Application Support/Claude/` on macOS and `%APPDATA%\Claude\` on Windows. Add
RayCrafter under `mcpServers`, then quit and reopen the app:

```json
{
  "mcpServers": {
    "raycrafter": {
      "command": "/full/path/to/raycrafter-mcp"
    }
  }
}
```

The path must be written out in full: the app does not expand `~`, and it may not see the same
`PATH` as your terminal. Find the path with `which raycrafter-mcp` (macOS, Linux) or
`where raycrafter-mcp` (Windows).

If you use uv and installed nothing, run RayCrafter through `uvx` instead (find its path with
`which uvx`):

```json
{
  "mcpServers": {
    "raycrafter": {
      "command": "/full/path/to/uvx",
      "args": ["--from", "raycrafter[mcp]", "raycrafter-mcp"]
    }
  }
}
```

Other MCP clients (for example Claude Code: `claude mcp add raycrafter -- raycrafter-mcp`) take
the same command.


## An agent-generated project

The code in this repository is written by AI coding agents, working with the project's
developers. Due to time constraints, the project is only possible because of agentic
coding.

We nonetheless aim for well-designed, readable code that people who use little or no AI can
understand and contribute to. The rules the agents follow are in [CLAUDE.md](CLAUDE.md), and
significant decisions, with their reasons, are recorded in [log.md](log.md).
