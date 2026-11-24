"""Physics checks of the bench against known values, mainly for the ACT508-200-A (f = 200 mm).

These are the checks recorded during the pilot. Every change to the optics must keep them
passing (see CLAUDE.md).
"""

import numpy as np
import pytest

from raycrafter.core import OpticalModel, analysis
from raycrafter.core.analysis import airy_radius
from raycrafter.core.plot_geometry import f_number_text


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
    assert spot.rms_radius < airy_radius(spot)


def test_off_axis_image_height_is_f_tan_theta(focused_model):
    focused_model.state.selected_field.angle_deg = 5.0
    spot = focused_model.trace_fast().spot
    assert abs(spot.centroid_y) == pytest.approx(200.0 * np.tan(np.radians(5.0)), rel=0.01)


def test_flipping_the_lens_makes_the_spot_much_worse():
    model = OpticalModel()
    model.state.selected_field.diameter_mm = 20.0
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


# ------------------------------------------------- beams wider than a lens (issue #3)


def bench_with(part_number: str, beam_diameter_mm: float) -> OpticalModel:
    """One lens on the bench, a collimated beam of the given diameter, autofocused."""
    model = OpticalModel(add_default_lens=False)
    model.add_lens(part_number)
    model.state.selected_field.diameter_mm = beam_diameter_mm
    model.autofocus()
    return model


def test_a_lens_rim_limits_the_beam_to_the_lens_diameter():
    # LA1116 is Ø6 mm and convex side first, so rays converge inside it and the front rim
    # is what limits the beam: a 10 mm beam is cut down to 6 mm (to within one probe ray).
    model = bench_with("LA1116", beam_diameter_mm=10.0)
    probe_step_mm = 10.0 * 2 / (analysis.PUPIL_PROBE_RAYS - 1)
    assert analysis.transmitted_beam_diameter(model.state) == pytest.approx(6.0, abs=probe_step_mm)

    # A beam narrower than the lens passes whole.
    assert analysis.transmitted_beam_diameter(bench_with("ACT508-200-A", 10.0).state) == 10.0


def test_analyses_use_only_the_rays_that_get_through_a_small_lens():
    # A 10 mm beam on a Ø2 mm lens used to give NaN everywhere and crash the MTF.
    model = bench_with("AC020-004-A", beam_diameter_mm=10.0)
    fast_trace = model.trace_fast()
    assert fast_trace.spot.has_light and np.isfinite(fast_trace.spot.rms_radius)
    assert np.isfinite(model.compute_through_focus(fast_trace).rms_current).all()
    assert model.compute_mtf().frequency.size > 0
    assert np.isfinite(model.compute_field_curvature().tangential_shift[0])

    # In the ray fan, the rays beyond the rim are blocked (NaN) and the central ones are not.
    fan = model.compute_ray_fan()
    assert (
        np.isnan(fan.tangential_error[0]) and fan.tangential_error[analysis.RAY_FAN_RAYS // 2] == 0
    )


def test_no_light_at_all_is_reported_not_an_error():
    # Two Ø2 mm lenses 300 mm apart: a 10 degree beam through the first misses the second.
    model = OpticalModel(add_default_lens=False)
    first = model.add_lens("AC020-004-A")
    model.insert_lens(model.make_lens("AC020-004-A", front_z=first.back_z + 300.0))
    model.state.selected_field.angle_deg = 10.0
    image_z = model.state.image_z

    fast_trace = model.trace_fast()
    assert not fast_trace.spot.has_light
    assert np.isnan(model.compute_through_focus(fast_trace).rms_current).all()
    assert model.compute_mtf().frequency.size == 0
    model.autofocus()
    assert model.state.image_z == image_z  # nothing to focus, so the image plane stays put


def test_f_number_uses_the_transmitted_beam_when_a_lens_clips_it():
    # LA1116 (Ø6 mm) under a 10 mm beam: the rim passes 6 mm, so f/# = EFL / 6 mm, not EFL / 10.
    model = bench_with("LA1116", beam_diameter_mm=10.0)
    effective_focal_length = model.trace_fast().summary.effective_focal_length
    result = model.compute_f_number()
    assert result.beam_is_clipped
    assert result.f_number == pytest.approx(effective_focal_length / 6.0, rel=0.01)
    assert f_number_text(result).endswith("(beam clipped to 6.0 mm)")


def test_f_number_of_an_unclipped_beam_is_unchanged(focused_model):
    result = focused_model.compute_f_number()  # ACT508-200-A, 10 mm beam
    assert not result.beam_is_clipped
    assert result.f_number == pytest.approx(200.0 / 10.0, abs=0.1)
    assert f_number_text(result) == "f/20.0"
