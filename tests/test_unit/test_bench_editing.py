"""Tests of editing the bench: adding, moving, swapping and inserting lenses."""

import pytest

from raycrafter.core import OpticalModel


@pytest.fixture
def two_identical_lenses() -> OpticalModel:
    """Two ACT508-200-A lenses next to each other."""
    model = OpticalModel()
    model.add_lens("ACT508-200-A")
    return model


def test_new_lens_goes_to_the_right_without_overlap(two_identical_lenses):
    first, second = two_identical_lenses.state.lenses
    assert second.z > first.back_z
    assert two_identical_lenses.state.image_z > second.back_z


def test_moving_a_lens_cannot_overlap_its_neighbour(two_identical_lenses):
    model = two_identical_lenses
    first, second = model.state.lenses
    model.move_lens(0, requested_front_z=first.z + 1.0)  # tiny step to the right
    model.move_lens(0, requested_front_z=second.z - 1.0)  # would overlap, but not pass the centre
    assert first.back_z <= second.z + 1e-9


def test_dragging_past_a_neighbours_centre_swaps_them():
    model = OpticalModel()
    model.add_lens("AC508-100-A")
    first, second = model.state.lenses
    span = (first.z, second.back_z)

    new_order = model.move_lens(0, requested_front_z=second.back_z)
    assert new_order == [1, 0]
    assert [lens.part_number for lens in model.state.lenses] == ["AC508-100-A", "ACT508-200-A"]
    assert (model.state.lenses[0].z, model.state.lenses[1].back_z) == pytest.approx(span)


def test_zmx_export_writes_a_file(tmp_path, two_identical_lenses):
    zmx_path = tmp_path / "system.zmx"
    two_identical_lenses.save_zmx(str(zmx_path))
    assert zmx_path.stat().st_size > 0


def test_free_front_z_snaps_out_of_an_occupied_stretch(two_identical_lenses):
    model = two_identical_lenses
    first, second = model.state.lenses
    thickness = 5.0

    # Asking for a spot inside the first lens gives the nearest free spot beside it.
    requested = first.z + 0.4 * first.thickness
    front_z = model.free_front_z(thickness, requested)
    assert front_z + thickness <= first.z + 1e-9 or front_z >= first.back_z - 1e-9

    # A free spot is returned unchanged.
    assert model.free_front_z(thickness, 40.0) == pytest.approx(40.0)
    # A lens thicker than every gap (the space after the last lens being capped) fits nowhere.
    assert model.free_front_z(150.0, 40.0, max_back_z=second.back_z + 1.0) is None


def test_inserted_lens_goes_into_axis_order(two_identical_lenses):
    model = two_identical_lenses
    first, second = model.state.lenses
    gap_middle = 0.5 * (first.back_z + second.z)
    new_lens = model.make_lens("LA1050")
    new_lens.z = model.free_front_z(new_lens.thickness, gap_middle - new_lens.thickness / 2)

    index = model.insert_lens(new_lens)
    assert index == 1
    assert [lens.part_number for lens in model.state.lenses] == [
        "ACT508-200-A",
        "LA1050",
        "ACT508-200-A",
    ]


def test_inserting_behind_the_image_plane_pushes_it_back_within_the_limit():
    model = OpticalModel()
    new_lens = model.make_lens("LA1050", front_z=model.state.image_z + 20.0)
    model.insert_lens(new_lens, max_image_z=new_lens.back_z + 10.0)
    assert model.state.image_z == pytest.approx(new_lens.back_z + 10.0)
