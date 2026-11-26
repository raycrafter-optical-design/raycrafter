"""The interactive 3D view in the HTML report (#51).

The page carries everything the view needs, so that the report stays one file that can be
sent to someone and opened without an internet connection:

- the scene (``raycrafter.core.scene``, the same one the app's 3D view draws) as JSON in
  the data format documented there;
- three.js (``assets/three``), as ``data:`` URLs in an "import map", which lets the page's
  scripts import it by name;
- ``assets/viewer3d.js``, which draws the scene.
"""

from __future__ import annotations

import base64
import json
from importlib import resources

from raycrafter.core.scene import STANDARD_VIEWS, Scene, scene_to_dict

# The names the viewer script imports, and the packaged files that provide them.
THREE_MODULES = {
    "three": "three/three.module.min.js",
    "three-orbit-controls": "three/OrbitControls.js",
}


def read_asset(file_name: str) -> str:
    """Text of one of the report's packaged asset files (CSS or JavaScript)."""
    return resources.files("raycrafter.report").joinpath("assets", file_name).read_text("utf-8")


def json_for_script(data: object) -> str:
    """Data as JSON that is safe inside a <script> element."""
    # "</" inside a script block would end it early, so escape it.
    return json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")


def viewer_html() -> str:
    """The view itself: its buttons and the box the scene is drawn in."""
    buttons = "".join(
        f'<button type="button" data-view="{name}">{name}</button>' for name in STANDARD_VIEWS
    )
    return (
        f'<div class="views">{buttons}</div>'
        '<div id="viewer3d"><p id="viewer3d-message">Loading the 3D view…</p></div>'
    )


def scripts_html(scene: Scene) -> str:
    """The data and scripts that fill in the view; they go at the end of the page."""
    imports = {}
    for name, file_name in THREE_MODULES.items():
        content = base64.b64encode(read_asset(file_name).encode("utf-8")).decode("ascii")
        imports[name] = f"data:text/javascript;base64,{content}"
    data = {**scene_to_dict(scene), "views": STANDARD_VIEWS}
    return (
        "<!-- The 3D view uses three.js (MIT licence, Copyright 2010-2023 Three.js Authors). -->\n"
        f'<script type="importmap">{json_for_script({"imports": imports})}</script>\n'
        f'<script id="scene-data" type="application/json">{json_for_script(data)}</script>\n'
        f'<script type="module">{read_asset("viewer3d.js")}</script>'
    )
