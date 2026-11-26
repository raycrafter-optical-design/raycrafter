"""Tests of Plossl pairs made from one lens (issue #67)."""

import copy
import math

import pytest

from raycrafter.core import OpticalModel
from raycrafter.core.assemblies import PLOSSL_NAME
from raycrafter.core.gaps import closest_sag_difference_mm
from raycrafter.core.plossl import PLOSSL_GAP_MM
from raycrafter.core.replacement import principal_planes_z


def optical_centre_z(lenses) -> float:
    """Midway between the principal planes of a run of lenses."""
    front_z, rear_z = principal_planes_z(lenses, 555.0)
    return 0.5 * (front_z + rear_z)


def bench_with(*placements: tuple[str, float], image_z: float = 400.0) -> OpticalModel:
    """A bench with the given (part number, front z) lenses."""
    model = OpticalModel(add_default_lens=False)
    for part_number, front_z in placements:
        model.insert_lens(model.make_lens(part_number, front_z))
    model.state.image_z = image_z
    return model


@pytest.mark.parametrize("part_number", ["ACT508-200-A", "AC254-050-A", "LA1509-A"])
def test_a_lens_becomes_a_pair_centred_on_its_optical_centre(part_number):
    model = bench_with((part_number, 100.0))
    centre_z = optical_centre_z(model.state.lenses)
    model.make_plossl(0)

    left, right = model.state.lenses
    assert left.part_number == right.part_number == part_number
    assert model.assembly_indices(0) == [0, 1]
    assert model.assembly_name(0) == PLOSSL_NAME
    assert optical_centre_z(model.state.lenses) == pytest.approx(centre_z, abs=1e-6)
    closest_mm = right.z - left.back_z + closest_sag_difference_mm(left, right)
    assert closest_mm == pytest.approx(PLOSSL_GAP_MM)


def test_a_pair_of_doublets_has_its_crown_elements_inside():
    model = bench_with(("AC254-050-A", 100.0))
    model.make_plossl(0)
    left, right = model.state.lenses
    # These doublets are catalogued crown first, with the crown's face the strongly curved one.
    assert left.surfaces_in_use()[-1].radius == pytest.approx(-33.34)
    assert right.surfaces_in_use()[0].radius == pytest.approx(33.34)


def test_a_pair_of_plano_convex_singlets_has_its_curved_faces_inside():
    model = bench_with(("LA1509-A", 100.0))
    model.make_plossl(0)
    left, right = model.state.lenses
    assert math.isinf(left.surfaces_in_use()[0].radius)
    assert left.surfaces_in_use()[-1].radius == pytest.approx(-51.51)
    assert right.surfaces_in_use()[0].radius == pytest.approx(51.51)
    assert math.isinf(right.surfaces_in_use()[-1].radius)


def test_making_a_pair_of_a_flipped_lens_gives_the_same_pair():
    upright = bench_with(("AC254-050-A", 100.0))
    flipped = bench_with(("AC254-050-A", 100.0))
    flipped.flip_lens(0)
    upright.make_plossl(0)
    flipped.make_plossl(0)
    assert [lens.flipped for lens in flipped.state.lenses] == [
        lens.flipped for lens in upright.state.lenses
    ]


def test_the_pair_has_a_shorter_focal_length_than_one_lens():
    model = bench_with(("AC254-100-A", 100.0))
    single_mm = model.trace_fast().summary.effective_focal_length
    model.make_plossl(0)
    pair_mm = model.trace_fast().summary.effective_focal_length
    assert 0.45 * single_mm < pair_mm < 0.6 * single_mm


def test_neighbours_stay_put_and_the_new_indices_follow_the_pair():
    model = bench_with(("LA1050", 40.0), ("AC254-050-A", 100.0), ("LA1509-A", 150.0))
    before = [lens.z for lens in model.state.lenses]
    model.make_plossl(1)
    lenses = model.state.lenses
    assert [lens.part_number for lens in lenses] == [
        "LA1050",
        "AC254-050-A",
        "AC254-050-A",
        "LA1509-A",
    ]
    assert [lenses[0].z, lenses[3].z] == [before[0], before[2]]
    assert model.state.image_z == 400.0


def test_locked_gaps_keep_their_length_and_move_their_lenses():
    model = bench_with(("LA1050", 60.0), ("AC254-050-A", 100.0), ("LA1509-A", 120.0))
    model.set_gap_locked(0, True)
    model.set_gap_locked(1, True)
    gaps_before = [gap.width_mm for gap in model.air_gaps()]
    model.make_plossl(1)
    gaps_after = model.air_gaps()
    assert gaps_after[0].width_mm == pytest.approx(gaps_before[0])
    assert gaps_after[2].width_mm == pytest.approx(gaps_before[1])
    assert [gap.locked for gap in gaps_after] == [True, True, True, False]


def test_a_locked_image_gap_moves_the_image_plane_with_the_pair():
    model = bench_with(("AC254-050-A", 100.0), image_z=200.0)
    model.set_gap_locked(0, True)
    gap_mm = model.air_gaps()[0].width_mm
    model.make_plossl(0)
    assert model.air_gaps()[-1].width_mm == pytest.approx(gap_mm)
    assert model.air_gaps()[-1].locked


def test_a_locked_source_gap_keeps_its_length():
    model = bench_with(("AC254-050-A", 100.0))
    model.set_source_gap_locked(False)
    model.set_source_z(70.0)
    model.set_source_gap_locked(True)
    model.make_plossl(0)
    assert model.source_gap().width_mm == pytest.approx(30.0)


@pytest.mark.parametrize(
    "placements, lens_index",
    [
        ((("LA1050", 85.0), ("AC254-050-A", 100.0)), 1),  # too close in front
        ((("AC254-050-A", 100.0), ("LA1050", 112.0)), 0),  # too close behind
        ((("AC254-050-A", 21.0),), 0),  # at the start of the bench
    ],
)
def test_a_pair_that_does_not_fit_is_refused_and_nothing_changes(placements, lens_index):
    model = bench_with(*placements)
    state_before = copy.deepcopy(model.state)
    with pytest.raises(ValueError, match="does not fit"):
        model.make_plossl(lens_index)
    assert model.state == state_before


def test_only_a_real_positive_lens_on_its_own_can_become_a_pair():
    model = bench_with(("ACN254-050-A", 100.0), ("AC254-050-A", 160.0))
    model.insert_lens(model.make_paraxial_lens(100.0, 250.0))
    model.make_plossl(1)
    for lens_index, reason in [(0, "positive"), (1, "assembly"), (3, "paraxial")]:
        assert reason in model.reason_no_plossl(lens_index)
        with pytest.raises(ValueError, match=reason):
            model.make_plossl(lens_index)


def test_a_pair_moves_as_one_and_splits():
    model = bench_with(("AC254-050-A", 100.0))
    model.make_plossl(0)
    left, right = model.state.lenses
    spacing = right.z - left.z

    model.move_lens(0, requested_front_z=left.z - 5.0)
    assert right.z - left.z == pytest.approx(spacing)

    model.split_assembly(0)
    assert left.assembly_id is None and right.assembly_id is None


def test_a_lens_in_a_pair_cannot_be_flipped_and_deleting_one_releases_the_other():
    model = bench_with(("AC254-050-A", 100.0))
    model.make_plossl(0)
    flipped_before = model.state.lenses[0].flipped
    model.flip_lens(0)
    assert model.state.lenses[0].flipped == flipped_before

    model.delete_lens(0)
    assert len(model.state.lenses) == 1
    assert model.state.lenses[0].assembly_id is None


def test_a_lens_is_never_dropped_inside_a_pair():
    model = bench_with(("AC254-050-A", 100.0))
    model.make_plossl(0)
    left, right = model.state.lenses
    gap_centre = 0.5 * (left.back_z + right.z)
    front_z = model.free_front_z(0.1, gap_centre)
    assert not left.z < front_z < right.back_z
