"""An MCP server that lets AI agents design and analyse lens systems with RayCrafter.

``DesignTools`` (``raycrafter.mcp_server.tools``) is the tool layer: plain methods on an
``OpticalModel`` that return JSON-friendly dictionaries, with no dependency on the MCP library.
``raycrafter.mcp_server.server`` wraps them as MCP tools. Needs the ``mcp`` extra.
"""
