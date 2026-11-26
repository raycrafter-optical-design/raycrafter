"""The shortcuts that keep each update fast (#27) must not change what it shows."""

import numpy as np
import pytest

from raycrafter.core import OpticalModel, analysis, optic_builder


@pytest.fixture
def mixed_fields_bench() -> OpticalModel:
    """Three unlike fields through two lenses; the last field is clipped by the small one."""
    model = OpticalModel()
    model.insert_lens(model.make_lens("AC127-050-A", front_z=model.state.last_back_z() + 40.0))
    blue = model.add_field()
    blue.wavelength_nm, blue.angle_deg = 450.0, 2.0
    wide = model.add_field()
    wide.wavelength_nm, wide.diameter_mm, wide.vergence_dioptres = 700.0, 30.0, -1.0
    return model


def test_fields_traced_together_match_each_traced_alone(mixed_fields_bench):
    state = mixed_fields_bench.state
    together = mixed_fields_bench.trace_fast()
    assert np.isnan(together.paths[2].y).any()  # some rays are blocked, so that case is covered

    all_fields = list(state.fields)
    for field_number, beam in enumerate(all_fields):
        state.fields, state.selected_field_index = [beam], 0
        alone = analysis.trace_fast(state)

        spot, expected_spot = together.spots[field_number], alone.spots[0]
        np.testing.assert_allclose(spot.position_x, expected_spot.position_x, atol=1e-12)
        np.testing.assert_allclose(spot.position_y, expected_spot.position_y, atol=1e-12)
        assert spot.rms_radius == pytest.approx(expected_spot.rms_radius)
        assert spot.wavelength_um == beam.wavelength_um

        paths, expected_paths = together.paths[field_number], alone.paths[0]
        np.testing.assert_allclose(paths.z, expected_paths.z, atol=1e-12)
        np.testing.assert_allclose(paths.y, expected_paths.y, atol=1e-12)


def test_each_glass_is_loaded_once_and_shared(focused_model):
    first = optic_builder.build_optic(focused_model.state)
    second = optic_builder.build_optic(focused_model.state)
    assert second.surfaces[1].material_post is first.surfaces[1].material_post


def test_a_shared_glass_does_not_remember_every_wavelength_for_ever(focused_model):
    wavelengths_nm = np.linspace(450.0, 700.0, 2 * optic_builder.MAX_REMEMBERED_INDICES)
    for wavelength_nm in wavelengths_nm:
        focused_model.state.selected_field.wavelength_nm = float(wavelength_nm)
        focused_model.trace_fast()
    material = optic_builder.build_optic(focused_model.state).surfaces[1].material_post
    # A few more than the limit can be added by one trace before the next emptying.
    assert len(material._n_cache) < optic_builder.MAX_REMEMBERED_INDICES + 10
