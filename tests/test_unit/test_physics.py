"""Physics checks of the bench against known values for the ACT508-200-A (f = 200 mm).

These are the checks recorded during the pilot. Every change to the optics must keep them
passing (see CLAUDE.md).
"""

import numpy as np
import pytest

from raycrafter.core import OpticalModel
from raycrafter.core.analysis import airy_radius


def test_back_focal_length_matches_the_catalogue(focused_model):
    summary = focused_model.trace_fast().summary
    assert summary.effective_focal_length == pytest.approx(200.0, abs=0.5)
    assert summary.back_focal_length == pytest.approx(190.6, abs=0.1)


def test_autofocus_lands_on_the_back_focal_point(focused_model):
    summary = focused_model.trace_fast().summary
    lens_back_z = focused_model.state.last_back_z()
    assert focused_model.state.image_z == pytest.approx(
        lens_back_z + summary.back_focal_length, abs=0.2
    )


def test_on_axis_spot_is_far_below_the_diffraction_limit(focused_model):
    spot = focused_model.trace_fast().spot
    assert spot.rms_radius * 1000 == pytest.approx(0.07, abs=0.02)  # micrometres
    assert spot.rms_radius < airy_radius(spot.numerical_aperture)


def test_off_axis_image_height_is_f_tan_theta(focused_model):
    focused_model.state.field_angle = 5.0
    spot = focused_model.trace_fast().spot
    assert abs(spot.centroid_y) == pytest.approx(200.0 * np.tan(np.radians(5.0)), rel=0.01)


def test_flipping_the_lens_makes_the_spot_much_worse():
    model = OpticalModel()
    model.state.beam_diameter = 20.0
    model.autofocus()
    correct_way_round = model.trace_fast().spot.rms_radius

    model.flip_lens(0)
    model.autofocus()
    assert model.trace_fast().spot.rms_radius > 10 * correct_way_round


def test_through_focus_minimum_is_at_the_image_plane_after_autofocus(focused_model):
    result = focused_model.compute_through_focus(focused_model.trace_fast())
    assert result.best_offset == pytest.approx(0.0, abs=0.05)


def test_diffraction_limited_mtf_lies_on_the_ideal_curve(focused_model):
    mtf = focused_model.compute_mtf()
    np.testing.assert_allclose(mtf.tangential, mtf.diffraction_limit, atol=0.05)


def test_field_curvature_is_zero_on_axis_at_best_focus(focused_model):
    curvature = focused_model.compute_field_curvature()
    assert curvature.tangential_shift[0] == pytest.approx(0.0, abs=0.05)
    assert curvature.sagittal_shift[0] == pytest.approx(0.0, abs=0.05)
