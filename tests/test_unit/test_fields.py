"""Tests of fields (beams): editing them, launching their rays and their physics."""

import numpy as np
import pytest

from raycrafter.core import OpticalModel
from raycrafter.core.state import AXIS_START_Z
from raycrafter.palette import wavelength_colour

# Schott N-BK7 refractive indices at the F, d and C lines (486.13, 587.56, 656.27 nm).
N_BK7_INDEX = {486.13: 1.52238, 587.56: 1.51680, 656.27: 1.51432}
LA1509_RADIUS_MM = 51.51  # the curved face of the LA1509 plano-convex lens (N-BK7, f = 100 mm)


@pytest.fixture
def ideal_lens_bench() -> OpticalModel:
    """A paraxial lens of f = 100 mm with its front at z = 100 mm, and one 10 mm beam."""
    model = OpticalModel(add_default_lens=False)
    model.insert_lens(model.make_paraxial_lens(100.0, z=100.0))
    return model


# ------------------------------------------------------------------ editing


def test_a_bench_starts_with_one_collimated_on_axis_field():
    beam = OpticalModel().state.selected_field
    assert (beam.diameter_mm, beam.angle_deg, beam.vergence_dioptres, beam.offset_mm) == (
        10.0,
        0.0,
        0.0,
        0.0,
    )
    assert beam.wavelength_nm == 555.0 and beam.colour is None


def test_a_new_field_copies_the_selected_one_and_is_selected():
    model = OpticalModel()
    model.state.selected_field.angle_deg = 3.0
    model.state.selected_field.colour = "#123456"
    new_field = model.add_field()
    assert len(model.state.fields) == 2 and model.state.selected_field_index == 1
    assert new_field.angle_deg == 3.0
    assert new_field.colour is None  # back to the colour of its wavelength
    assert model.state.fields[0].colour == "#123456"  # the original is unchanged


def test_removing_fields_always_leaves_one():
    model = OpticalModel()
    model.add_field()
    model.add_field()
    model.remove_fields([1, 2])
    assert len(model.state.fields) == 1 and model.state.selected_field_index == 0

    model.remove_fields([0])  # the last field cannot be removed
    assert len(model.state.fields) == 1


def test_removing_a_field_that_does_not_exist_changes_nothing():
    model = OpticalModel()
    model.add_field()
    with pytest.raises(IndexError, match="no field 5"):
        model.remove_fields([1, 5])
    assert len(model.state.fields) == 2


def test_wavelength_colours_follow_the_spectrum():
    def rgb(wavelength_nm: float) -> tuple[int, ...]:
        colour = wavelength_colour(wavelength_nm)
        return tuple(int(colour[index : index + 2], 16) for index in (1, 3, 5))

    red, green, blue = rgb(450)
    assert blue > red and blue > green
    red, green, blue = rgb(530)
    assert green > red and green > blue
    red, green, blue = rgb(650)
    assert red > green and red > blue
    assert wavelength_colour(1064) == wavelength_colour(780)  # infrared: the deep red end


# ------------------------------------------------------------------ ray launching


def test_side_view_rays_start_at_the_axis_start_and_meet_the_lens_at_the_offset(
    ideal_lens_bench,
):
    ideal_lens_bench.state.selected_field.offset_mm = 4.0
    paths = ideal_lens_bench.trace_fast().paths[0]
    assert np.all(paths.z[0] == AXIS_START_Z)
    # Row 1 is the lens. The middle ray of the fan is the centre of the beam.
    assert paths.z[1, 2] == pytest.approx(100.0)
    assert paths.y[1, 2] == pytest.approx(4.0)
    assert paths.y[1].max() - paths.y[1].min() == pytest.approx(10.0)  # the beam's width


def test_an_offset_collimated_beam_still_focuses_on_the_axis(ideal_lens_bench):
    ideal_lens_bench.state.selected_field.offset_mm = 4.0
    ideal_lens_bench.autofocus()
    assert ideal_lens_bench.state.image_z == pytest.approx(200.0, abs=0.01)
    assert ideal_lens_bench.trace_fast().spot.centroid_y == pytest.approx(0.0, abs=1e-6)


def test_a_diverging_beam_is_imaged_by_the_lens_equation(ideal_lens_bench):
    # -5 D is a point source 200 mm before the lens; 1/v = 1/100 - 1/200 puts the image 200 mm
    # behind it.
    ideal_lens_bench.state.selected_field.vergence_dioptres = -5.0
    ideal_lens_bench.autofocus()
    assert ideal_lens_bench.state.image_z == pytest.approx(300.0, abs=0.05)
    assert ideal_lens_bench.compute_mtf().frequency.size > 0  # Optiland's own object model


def test_each_field_is_traced_and_reported_separately(ideal_lens_bench):
    model = ideal_lens_bench
    model.add_field().angle_deg = 2.0
    model.state.image_z = 200.0
    fast_trace = model.trace_fast()
    assert len(fast_trace.spots) == len(fast_trace.paths) == 2
    assert fast_trace.spot is fast_trace.spots[1]  # the new field is the selected one
    assert fast_trace.spots[0].centroid_y == pytest.approx(0.0, abs=1e-9)
    assert fast_trace.spots[1].centroid_y == pytest.approx(100.0 * np.tan(np.radians(2.0)))


# ------------------------------------------------------------------ wavelength


def test_focal_length_follows_the_glass_index_at_each_wavelength():
    # For a plano-convex lens, EFL = R / (n - 1) exactly, so N-BK7's dispersion moves the
    # focus about 1.5 mm between blue (F) and red (C) light.
    model = OpticalModel(add_default_lens=False)
    model.add_lens("LA1509")
    for wavelength_nm, index in N_BK7_INDEX.items():
        model.state.selected_field.wavelength_nm = wavelength_nm
        expected_efl = LA1509_RADIUS_MM / (index - 1.0)
        assert model.trace_fast().summary.effective_focal_length == pytest.approx(
            expected_efl, rel=2e-4
        )


def test_blue_and_red_fields_focus_apart_by_the_chromatic_shift():
    model = OpticalModel(add_default_lens=False)
    model.add_lens("LA1509")
    model.state.selected_field.diameter_mm = 2.0  # thin beam: spherical aberration is negligible
    model.state.selected_field.wavelength_nm = 486.13
    blue_focus_z = model.autofocus()
    model.add_field().wavelength_nm = 656.27
    red_focus_z = model.autofocus()

    shift_mm = LA1509_RADIUS_MM / (N_BK7_INDEX[656.27] - 1) - LA1509_RADIUS_MM / (
        N_BK7_INDEX[486.13] - 1
    )
    assert red_focus_z - blue_focus_z == pytest.approx(shift_mm, abs=0.02)


def test_every_wavelength_is_exported_to_zemax(ideal_lens_bench, tmp_path):
    ideal_lens_bench.add_field().wavelength_nm = 656.0
    zmx_path = tmp_path / "two_wavelengths.zmx"
    ideal_lens_bench.save_zmx(str(zmx_path))
    assert zmx_path.read_text().count("WAVM") == 2
