"""The camera arithmetic of the 3D view (no window or OpenGL context needed)."""

import itertools

import numpy as np
import pytest

from raycrafter.core.scene import STANDARD_VIEWS
from raycrafter.gui.view_3d import FIELD_OF_VIEW_DEG, FIT_MARGIN, fitting_distance, to_view


def test_the_axis_runs_across_the_view_and_y_is_up():
    assert to_view(np.array([1.0, 2.0, 3.0])) == pytest.approx([3.0, 1.0, 2.0])


@pytest.mark.parametrize("view", STANDARD_VIEWS)
@pytest.mark.parametrize("aspect", [0.7, 1.8])
def test_a_fitted_camera_sees_every_corner(view, aspect):
    """Project the corners of a long, thin box: all lie in the window, and one nearly fills it."""
    half_sizes = np.array([200.0, 25.0, 30.0])
    corners = np.array(list(itertools.product(*[(-size, size) for size in half_sizes])))
    azimuth, elevation = np.radians(STANDARD_VIEWS[view])
    distance = fitting_distance(corners, *STANDARD_VIEWS[view], aspect)

    to_camera = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    up = np.cross(to_camera, right)
    depth = distance - corners @ to_camera
    tan_half_width = np.tan(np.radians(FIELD_OF_VIEW_DEG) / 2)
    screen = np.maximum(
        np.abs(corners @ right) / (depth * tan_half_width),
        np.abs(corners @ up) / (depth * tan_half_width / aspect),
    )
    assert screen.max() <= 1.0
    assert screen.max() > 0.8 / FIT_MARGIN
