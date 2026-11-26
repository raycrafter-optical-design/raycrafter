"""The source plane, where every field is defined, and its gap to the first lens (issue #41)."""

import copy
import math

import numpy as np
import pytest

from raycrafter import Bench
from raycrafter.core import analysis
from raycrafter.core.model import FIRST_LENS_MIN_Z, OpticalModel
from raycrafter.core.project_file import ProjectFileError, project_from_dict, project_to_dict
from raycrafter.core.ray_launch import beam_at_first_lens


@pytest.fixture
def bench() -> OpticalModel:
    """One paraxial f = 100 mm lens at z = 100 mm and a 3 degree field."""
    model = OpticalModel(add_default_lens=False)
    model.insert_lens(model.make_paraxial_lens(100.0, z=100.0))
    model.state.image_z = 200.0
    model.state.selected_field.angle_deg = 3.0
    return model


def centre_height_at_first_lens(model: OpticalModel) -> float:
    """Where the centre of the selected field's beam crosses the first lens (mm)."""
    paths = model.trace_fast().paths[model.state.selected_field_index]
    middle_ray = paths.y.shape[1] // 2  # the fan is odd, so its middle ray is the centre
    return float(paths.y[1, middle_ray])


def test_by_default_the_source_plane_is_at_the_first_lens_and_moves_with_it(bench):
    assert bench.state.source_z == 100.0 and bench.state.source_gap_locked
    landing_before = centre_height_at_first_lens(bench)
    bench.move_lens(0, 150.0)
    assert bench.state.source_z == 150.0
    assert centre_height_at_first_lens(bench) == pytest.approx(landing_before)


def test_changing_the_source_gap_moves_an_angled_beam_by_the_change_times_tan_angle(bench):
    bench.set_source_gap_locked(False)
    landing_before = centre_height_at_first_lens(bench)
    bench.set_source_gap(40.0)
    assert bench.state.lenses[0].z == pytest.approx(140.0)  # the source plane stays put
    expected_rise = 40.0 * math.tan(math.radians(3.0))
    assert centre_height_at_first_lens(bench) - landing_before == pytest.approx(expected_rise)


def test_an_unlocked_source_plane_stays_put_and_stops_the_first_lens(bench):
    with pytest.raises(ValueError, match="locked"):
        bench.set_source_gap(10.0)
    bench.set_source_gap_locked(False)
    bench.set_source_z(60.0)
    bench.move_lens(0, 150.0)
    assert bench.state.source_z == 60.0
    bench.move_lens(0, 30.0)  # dragged in front of the source plane: it stops there
    assert bench.state.lenses[0].z == pytest.approx(60.0)
    with pytest.raises(ValueError, match="between"):
        bench.set_source_z(70.0)  # behind the first lens
    with pytest.raises(ValueError, match="between"):
        bench.set_source_z(-5.0)  # in front of the start of the axis
    assert bench.free_front_z(0.0, 30.0) == pytest.approx(60.0)


def test_a_locked_source_gap_follows_whichever_lens_is_first(bench):
    bench.set_source_gap_locked(False)
    bench.set_source_gap(30.0)  # source at 100, lens at 130
    bench.set_source_gap_locked(True)

    new_index = bench.insert_lens(bench.make_paraxial_lens(50.0, z=60.0))
    assert new_index == 0 and bench.state.source_z == pytest.approx(30.0)
    # The new lens cannot go so far forward that the source plane leaves the axis.
    assert bench.free_front_z(0.0, 0.0) == pytest.approx(max(FIRST_LENS_MIN_Z, 30.0))

    bench.delete_lens(0)
    assert bench.state.source_z == pytest.approx(100.0)  # 30 mm in front of the lens left
    assert bench.state.source_gap_locked


def test_the_beam_carried_to_the_lens_traces_the_same(bench):
    """``beam_at_first_lens`` (used by the MTF and ZMX export) matches the launched rays."""
    bench.state.selected_field.vergence_dioptres = 4.0  # converging on a point 250 mm away
    bench.state.selected_field.offset_mm = 1.0
    bench.set_source_gap_locked(False)
    bench.set_source_gap(50.0)
    spot_from_source = bench.trace_fast().spot

    at_lens = copy.deepcopy(bench.state)
    at_lens.fields[0] = beam_at_first_lens(bench.state, bench.state.selected_field)
    at_lens.source_z = at_lens.lenses[0].z
    assert at_lens.fields[0].diameter_mm == pytest.approx(10.0 * 0.8)
    assert at_lens.fields[0].vergence_dioptres == pytest.approx(4.0 / 0.8)
    spot_from_lens = analysis.trace_fast(at_lens).spot
    np.testing.assert_allclose(spot_from_lens.position_y, spot_from_source.position_y, atol=1e-9)
    np.testing.assert_allclose(spot_from_lens.position_x, spot_from_source.position_x, atol=1e-9)


def test_the_source_plane_is_saved_and_checked_in_project_files(bench):
    bench.set_source_gap_locked(False)
    bench.set_source_z(40.0)
    data = project_to_dict(bench.state)
    loaded = project_from_dict(data)
    assert loaded.source_z == 40.0 and not loaded.source_gap_locked
    data["source_z_mm"] = 120.0  # behind the first lens
    with pytest.raises(ProjectFileError, match="source plane"):
        project_from_dict(data)


def test_scripts_and_agents_set_the_source_gap():
    bench = Bench()
    assert bench.source_gap_mm == 0.0 and bench.source_gap_locked
    with pytest.raises(ValueError, match="locked"):
        bench.set_source_gap(20.0)
    bench.lock_source_gap(False)
    bench.set_source_gap(20.0)
    assert bench.source_gap_mm == pytest.approx(20.0)
    assert bench.lenses[0].front_z_mm == pytest.approx(bench.source_z_mm + 20.0)
    bench.set_source_z(bench.source_z_mm - 5.0)
    assert bench.source_gap_mm == pytest.approx(25.0)


def test_the_mtf_allows_for_the_source_gap(bench):
    bench.state.selected_field.vergence_dioptres = -2.0
    bench.set_source_gap_locked(False)
    bench.set_source_gap(50.0)
    at_lens = copy.deepcopy(bench.state)
    at_lens.fields[0] = beam_at_first_lens(bench.state, bench.state.selected_field)
    at_lens.source_z = at_lens.lenses[0].z
    from_source, from_lens = analysis.compute_mtf(bench.state), analysis.compute_mtf(at_lens)
    assert from_source.frequency.size > 0
    np.testing.assert_allclose(from_source.frequency, from_lens.frequency)
    np.testing.assert_allclose(from_source.tangential, from_lens.tangential)
