"""Tests of what is known about each lens as a part: vendor, mount type and surface range."""

import pytest

from raycrafter.api import Bench
from raycrafter.core.model import OpticalModel
from raycrafter.core.optic_builder import build_optic
from raycrafter.core.state import state_from_dict, state_to_dict
from raycrafter.core.zmx_import import load_zmx


@pytest.fixture
def model() -> OpticalModel:
    """A doublet, a paraxial lens and a triplet whose description says it is unmounted."""
    bench_model = OpticalModel()
    bench_model.insert_lens(bench_model.make_paraxial_lens(150.0, z=200.0))
    bench_model.add_lens("TRH254-040-A")
    return bench_model


def metadata(lenses) -> list[tuple]:
    return [(lens.part_number, lens.vendor, lens.mount_type) for lens in lenses]


EXPECTED = [
    ("ACT508-200-A", "Thorlabs", None),
    ("Paraxial f=+150 mm", None, None),
    ("TRH254-040-A", "Thorlabs", "Unmounted"),
]


def test_catalogue_lenses_report_vendor_and_mount_where_known(model):
    assert metadata(model.state.lenses) == EXPECTED
    assert model.catalog["AC080-030-B"].mount_type == "Mounted"


def test_metadata_survives_copying_the_state(model):
    copied = state_from_dict(state_to_dict(model.state))
    assert metadata(copied.lenses) == EXPECTED


def test_metadata_survives_a_zmx_save_and_load(model, tmp_path):
    zmx_path = str(tmp_path / "design.zmx")
    model.save_zmx(zmx_path)
    assert metadata(load_zmx(zmx_path, model.catalog).state.lenses) == EXPECTED


def test_a_lens_not_in_the_catalogue_has_no_vendor(model, tmp_path):
    zmx_path = str(tmp_path / "design.zmx")
    model.save_zmx(zmx_path)
    without_catalogue = load_zmx(zmx_path, {}).state.lenses[0]
    assert without_catalogue.vendor is None and without_catalogue.mount_type is None


def test_surface_ranges_match_the_optical_system(model):
    state = model.state
    ranges = [state.surface_range(index) for index in range(len(state.lenses))]
    assert ranges == [(1, 3), (4, 4), (5, 8)]  # doublet, paraxial lens, triplet

    surfaces = build_optic(state).surfaces.surfaces
    assert len(surfaces) == ranges[-1][1] + 2  # plus the object and image surfaces
    for lens, (first, _last) in zip(state.lenses, ranges):
        assert surfaces[first].geometry.cs.z - surfaces[1].geometry.cs.z == pytest.approx(
            lens.z - state.lenses[0].z
        )


def test_the_script_api_reports_the_metadata():
    lens = Bench().lenses[0]
    assert (lens.vendor, lens.mount_type, lens.surface_range) == ("Thorlabs", None, (1, 3))
    assert Bench().search_lenses("TRH254-040-A")[0].vendor == "Thorlabs"
