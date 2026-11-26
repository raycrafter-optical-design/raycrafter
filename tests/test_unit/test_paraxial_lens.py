"""Tests of paraxial (ideal thin) lenses: their optics, the editing rules and the export."""

import numpy as np
import pytest

from raycrafter.core import OpticalModel


@pytest.fixture
def paraxial_model() -> OpticalModel:
    """A bench with one paraxial lens of f = 100 mm at z = 50 mm and a 10 mm collimated beam."""
    model = OpticalModel(add_default_lens=False)
    model.insert_lens(model.make_paraxial_lens(100.0, z=50.0))
    return model


def test_paraxial_lens_focuses_a_collimated_beam_at_f(paraxial_model):
    paraxial_model.autofocus()
    assert paraxial_model.state.image_z == pytest.approx(150.0, abs=0.01)

    summary = paraxial_model.trace_fast().summary
    assert summary.effective_focal_length == pytest.approx(100.0)
    assert summary.back_focal_length == pytest.approx(100.0, abs=0.01)
    # An ideal lens has no aberration: what is left (below 1 nm) is the autofocus tolerance.
    assert paraxial_model.trace_fast().spot.rms_radius < 1e-6


def test_off_axis_image_height_is_f_tan_theta(paraxial_model):
    paraxial_model.state.selected_field.angle_deg = 5.0
    paraxial_model.state.image_z = 150.0
    spot = paraxial_model.trace_fast().spot
    assert abs(spot.centroid_y) == pytest.approx(100.0 * np.tan(np.radians(5.0)), rel=1e-6)


def test_two_thin_lenses_combine_as_the_thin_lens_formula_says():
    # 1/f = 1/f1 + 1/f2 - d / (f1 f2), with f1 = 100, f2 = -50, d = 60 mm gives f = 500 mm.
    model = OpticalModel(add_default_lens=False)
    model.insert_lens(model.make_paraxial_lens(100.0, z=50.0))
    model.insert_lens(model.make_paraxial_lens(-50.0, z=110.0))
    model.state.image_z = 300.0
    assert model.trace_fast().summary.effective_focal_length == pytest.approx(500.0)


def test_paraxial_lens_is_thin_and_cannot_be_flipped_or_paired():
    model = OpticalModel(add_default_lens=False)
    model.insert_lens(model.make_paraxial_lens(100.0, z=50.0))
    model.insert_lens(model.make_paraxial_lens(100.0, z=51.0))
    first = model.state.lenses[0]
    assert first.is_paraxial and first.thickness == 0.0 and first.back_z == first.z

    model.flip_lens(0)
    assert not first.flipped
    assert "paraxial" in model.reason_no_plossl(0)


def test_negative_paraxial_lens_is_marked_negative():
    assert OpticalModel.make_paraxial_lens(-75.0).is_negative
    assert not OpticalModel.make_paraxial_lens(75.0).is_negative


def test_zero_focal_length_is_rejected():
    with pytest.raises(ValueError):
        OpticalModel.make_paraxial_lens(0.0)


def test_paraxial_lens_is_exported_to_zemax(paraxial_model, tmp_path):
    zmx_path = tmp_path / "paraxial.zmx"
    paraxial_model.save_zmx(str(zmx_path))
    zmx_text = zmx_path.read_text()
    assert "TYPE PARAXIAL" in zmx_text
    assert "PARM 1 1.0000000000000000E+02" in zmx_text  # the focal length, 100 mm


def test_a_paraxial_lens_takes_a_new_focal_length_and_diameter(paraxial_model):
    """#70: the context bar edits a placed paraxial lens; it stays put and refocuses as f says."""
    paraxial_model.record_undo_step()
    paraxial_model.set_paraxial_lens(0, -40.0, 12.5)
    lens = paraxial_model.state.lenses[0]
    assert lens.paraxial_focal_length_mm == -40.0 and lens.is_negative
    assert lens.semi_aperture == pytest.approx(6.25) and lens.z == 50.0
    assert lens.part_number == "Paraxial f=-40 mm"

    paraxial_model.set_paraxial_lens(0, 200.0, 12.5)
    paraxial_model.autofocus()
    assert paraxial_model.state.image_z == pytest.approx(250.0, abs=0.01)

    # Undo goes back to the lens as it was.
    paraxial_model.record_undo_step()
    assert paraxial_model.undo()
    assert paraxial_model.state.lenses[0].paraxial_focal_length_mm == 100.0
    assert paraxial_model.state.lenses[0].semi_aperture == pytest.approx(25.4)


def test_a_paraxial_lens_keeps_its_diameter_in_a_project_file(paraxial_model, tmp_path):
    paraxial_model.set_paraxial_lens(0, 75.0, 8.0)
    project_path = paraxial_model.save_project(tmp_path / "thin.raycrafter")
    reopened = OpticalModel(add_default_lens=False)
    reopened.load_project(project_path)
    lens = reopened.state.lenses[0]
    assert lens.paraxial_focal_length_mm == 75.0 and lens.semi_aperture == pytest.approx(4.0)


def test_a_paraxial_lens_needs_a_focal_length_and_a_positive_diameter(paraxial_model):
    with pytest.raises(ValueError):
        paraxial_model.set_paraxial_lens(0, 0.0, 10.0)
    with pytest.raises(ValueError):
        paraxial_model.set_paraxial_lens(0, 50.0, 0.0)
    with pytest.raises(ValueError):
        OpticalModel.make_paraxial_lens(50.0, diameter_mm=-1.0)
    catalogue_model = OpticalModel()  # one catalogue lens
    with pytest.raises(ValueError):
        catalogue_model.set_paraxial_lens(0, 50.0, 10.0)
