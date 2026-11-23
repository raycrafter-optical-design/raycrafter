"""Tests of editing the bench: adding, moving, swapping and pairing lenses."""

import pytest

from raycrafter.core import OpticalModel
from raycrafter.core.model import PLOSSL_GAP_MM


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


def test_make_plossl_pairs_two_identical_lenses(two_identical_lenses):
    model = two_identical_lenses
    assert model.plossl_partner(0) == 1
    model.make_plossl(0)

    left, right = model.state.lenses
    assert left.group_id is not None and left.group_id == right.group_id
    # Crown elements face each other: the left lens's stronger element is at its back.
    assert left.flipped == left.positive_element_is_first()
    assert right.flipped != right.positive_element_is_first()
    assert right.z > left.back_z - PLOSSL_GAP_MM  # surfaces may curve towards each other


def test_a_plossl_pair_moves_as_one_and_splits(two_identical_lenses):
    model = two_identical_lenses
    model.make_plossl(0)
    left, right = model.state.lenses
    spacing = right.z - left.z

    model.move_lens(0, requested_front_z=left.z - 5.0)
    assert right.z - left.z == pytest.approx(spacing)

    model.split_plossl(0)
    assert left.group_id is None and right.group_id is None


def test_negative_lenses_cannot_form_a_plossl_pair():
    model = OpticalModel(add_default_lens=False)
    model.add_lens("ACN254-050-A")
    model.add_lens("ACN254-050-A")
    assert model.plossl_partner(0) is None


def test_a_lens_in_a_plossl_pair_cannot_be_flipped(two_identical_lenses):
    model = two_identical_lenses
    model.make_plossl(0)
    flipped_before = model.state.lenses[0].flipped
    model.flip_lens(0)
    assert model.state.lenses[0].flipped == flipped_before


def test_deleting_one_lens_of_a_pair_releases_the_other(two_identical_lenses):
    model = two_identical_lenses
    model.make_plossl(0)
    model.delete_lens(0)
    assert len(model.state.lenses) == 1
    assert model.state.lenses[0].group_id is None


def test_zmx_export_writes_a_file(tmp_path, two_identical_lenses):
    zmx_path = tmp_path / "system.zmx"
    two_identical_lenses.save_zmx(str(zmx_path))
    assert zmx_path.stat().st_size > 0
