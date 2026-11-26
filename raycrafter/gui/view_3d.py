"""The 3D view window: the bench's lenses, rays and planes, which can be turned round (#52).

Opened from View > Show 3D View. It draws the scene that ``raycrafter.core.scene`` builds
(the same one the HTML report shows) with ``pyqtgraph.opengl``, and holds nothing of its
own but the camera. The main window gives it a new scene once changes have settled, never
during a drag; the camera stays where the user put it.

The axis runs left to right on screen, as in the side view: the bench's (x, y, z) are drawn
at OpenGL's (z, x, y), whose third axis is up (``to_view``).
"""

from __future__ import annotations

import itertools

import numpy as np
import pyqtgraph.opengl as gl
from OpenGL import GL
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QVector3D
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from raycrafter.core.scene import (
    AXIS,
    LENS,
    PLANE,
    RAYS,
    STANDARD_VIEWS,
    Scene,
    SceneObject,
)

DEFAULT_VIEW = "3D"
FIELD_OF_VIEW_DEG = 30.0  # horizontal; narrow, so that a long bench is not much distorted
FIT_MARGIN = 1.08  # room left round the scene when a view is chosen
LINE_WIDTH_PX = {LENS: 1.5, RAYS: 1.2, PLANE: 1.5, AXIS: 1.0}
# Lines are drawn first, and the translucent meshes are then blended over them. The meshes
# ignore depth: otherwise whichever surface happened to be drawn first would hide the rays and
# lenses behind it, as glass should not.
LINE_DRAW_ORDER = 0
MESH_DRAW_ORDER = 1
MESH_GL_OPTIONS = {
    GL.GL_DEPTH_TEST: False,
    GL.GL_BLEND: True,
    GL.GL_CULL_FACE: False,
    "glBlendFuncSeparate": (
        GL.GL_SRC_ALPHA,
        GL.GL_ONE_MINUS_SRC_ALPHA,
        GL.GL_ONE,
        GL.GL_ONE_MINUS_SRC_ALPHA,
    ),
}


class View3DWindow(QWidget):
    """A window showing a ``Scene`` in 3D, with buttons for the standard views.

    Parameters
    ----------
    parent : QWidget, optional
        The main window, which owns the 3D view.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("RayCrafter: 3D view")
        self.resize(900, 600)
        self.scene: Scene | None = None
        self.scene_items: list = []

        self.gl_view = gl.GLViewWidget()
        self.gl_view.setBackgroundColor("w")
        self.gl_view.opts["fov"] = FIELD_OF_VIEW_DEG

        buttons = QHBoxLayout()
        self.view_buttons: dict[str, QPushButton] = {}
        for name in STANDARD_VIEWS:
            button = QPushButton(name)
            button.clicked.connect(lambda _checked=False, name=name: self.set_view(name))
            buttons.addWidget(button)
            self.view_buttons[name] = button
        hint = QLabel("Drag to rotate · scroll to zoom · Ctrl-drag to pan")
        hint.setObjectName("hintLabel")
        buttons.addStretch(1)
        buttons.addWidget(hint)

        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        layout.addWidget(self.gl_view, stretch=1)

    def show_scene(self, scene: Scene) -> None:
        """Replace what is drawn. The first scene is fitted to the window."""
        first = self.scene is None
        self.scene = scene
        for item in self.scene_items:
            self.gl_view.removeItem(item)
        self.scene_items = [item for part in scene.objects for item in gl_items(part)]
        for item in self.scene_items:
            self.gl_view.addItem(item)
        if first:
            self.set_view(DEFAULT_VIEW)

    def set_view(self, name: str) -> None:
        """Look at the scene from one of STANDARD_VIEWS, fitted to the window."""
        azimuth, elevation = STANDARD_VIEWS[name]
        if self.scene is None:
            self.gl_view.setCameraPosition(azimuth=azimuth, elevation=elevation)
            return
        low, high = (to_view(corner) for corner in self.scene.bounds())
        centre = 0.5 * (low + high)
        corners = np.array(list(itertools.product(*zip(low, high)))) - centre
        aspect = self.gl_view.width() / max(self.gl_view.height(), 1)
        distance = fitting_distance(corners, azimuth, elevation, aspect)
        self.gl_view.setCameraPosition(
            pos=QVector3D(*centre), distance=distance, azimuth=azimuth, elevation=elevation
        )


def to_view(points: np.ndarray) -> np.ndarray:
    """Bench coordinates (x, y, z) as OpenGL's (z, x, y): the axis across, y up."""
    return np.asarray(points)[..., [2, 0, 1]]


def fitting_distance(
    corners: np.ndarray, azimuth_deg: float, elevation_deg: float, aspect: float
) -> float:
    """How far the camera must be from the centre to see every corner.

    ``corners`` are the corners of the scene's bounding box relative to the point looked at,
    in OpenGL coordinates; ``aspect`` is the window's width over its height.
    """
    azimuth, elevation = np.radians(azimuth_deg), np.radians(elevation_deg)
    to_camera = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    up = np.cross(to_camera, right)
    tan_half_width = np.tan(np.radians(FIELD_OF_VIEW_DEG) / 2.0)
    tan_half_height = tan_half_width / aspect
    # A corner nearer the camera than the centre needs that much more distance.
    needed = np.maximum(
        np.abs(corners @ right) / tan_half_width, np.abs(corners @ up) / tan_half_height
    )
    return float(FIT_MARGIN * np.max(needed + corners @ to_camera))


def gl_items(part: SceneObject) -> list:
    """The OpenGL items that draw one object of the scene: its meshes, then its lines."""
    colour = QColor(part.colour)
    red, green, blue = colour.redF(), colour.greenF(), colour.blueF()
    items: list = []
    for mesh in part.meshes:
        item = gl.GLMeshItem(
            vertexes=to_view(mesh.vertices),
            faces=mesh.faces,
            color=(red, green, blue, part.opacity),
            smooth=False,
            shader=None,  # flat colour: pyqtgraph's lighting darkens faces to near black
            glOptions=MESH_GL_OPTIONS,
        )
        item.setDepthValue(MESH_DRAW_ORDER)
        items.append(item)
    if part.lines:
        # All of an object's polylines as one item of separate segments (pairs of points).
        segments = np.concatenate(
            [np.repeat(to_view(line), 2, axis=0)[1:-1] for line in part.lines]
        )
        item = gl.GLLinePlotItem(
            pos=segments,
            color=(red, green, blue, 1.0),
            width=LINE_WIDTH_PX[part.kind],
            mode="lines",
            antialias=True,
            glOptions="translucent",
        )
        item.setDepthValue(LINE_DRAW_ORDER)
        items.append(item)
    return items
