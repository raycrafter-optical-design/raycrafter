"""Tests of the side view's axis ranges (``raycrafter.core.plot_geometry.axis_ranges``)."""

import pytest

from raycrafter.core import OpticalModel
from raycrafter.core.plot_geometry import (
    DEFAULT_AXIS_Y_RANGE,
    DEFAULT_AXIS_Z_RANGE,
    IMAGE_EDGE_GAP_MM,
    axis_ranges,
)
from raycrafter.core.state import OpticalState


def test_a_design_that_fits_gets_the_default_ranges():
    assert axis_ranges(OpticalModel().state) == (DEFAULT_AXIS_Z_RANGE, DEFAULT_AXIS_Y_RANGE)
    assert axis_ranges(OpticalState()) == (DEFAULT_AXIS_Z_RANGE, DEFAULT_AXIS_Y_RANGE)


def test_a_long_design_is_shown_up_to_beyond_its_image_plane():
    model = OpticalModel()
    model.insert_lens(model.make_paraxial_lens(50.0, z=700.0))
    model.state.image_z = 760.0

    (z_min, z_max), y_range = axis_ranges(model.state)
    assert z_min == DEFAULT_AXIS_Z_RANGE[0]
    assert z_max - IMAGE_EDGE_GAP_MM > 760.0  # the image plane can still be dragged a little
    assert y_range == DEFAULT_AXIS_Y_RANGE


def test_a_tall_lens_widens_the_y_range():
    model = OpticalModel()
    model.state.lenses[0].semi_aperture = 50.0
    _z_range, (y_min, y_max) = axis_ranges(model.state)
    assert y_max > 50.0 and y_min == pytest.approx(-y_max)
