"""Optimising one gap for a goal (issue #38): collimate the output, or focus on the image plane."""

import copy

import pytest

from raycrafter.core import analysis
from raycrafter.core.gap_optimisation import COLLIMATE, FOCUS, optimise_gap
from raycrafter.core.gaps import SOURCE_GAP_INDEX
from raycrafter.core.model import OpticalModel
from raycrafter.mcp_server.tools import DesignTools


@pytest.fixture(scope="module")
def catalog():
    from raycrafter.lenscatalog import thorlabs

    return thorlabs.load_lenses()


def make_bench(catalog, lenses, beam_diameter_mm=10.0, image_z=600.0) -> OpticalModel:
    """A bench of (part number or paraxial focal length, front z) pairs."""
    model = OpticalModel(catalog, add_default_lens=False)
    for part, front_z in lenses:
        if isinstance(part, float):
            model.insert_lens(model.make_paraxial_lens(part, front_z))
        else:
            model.insert_lens(model.make_lens(part, front_z))
    model.state.image_z = image_z
    model.state.selected_field.diameter_mm = beam_diameter_mm
    return model


@pytest.mark.parametrize(
    ("second_focal_length_mm", "expected_gap_mm"), [(50.0, 150.0), (-50.0, 50.0)]
)
def test_a_paraxial_pair_collimates_at_the_sum_of_focal_lengths(
    catalog, second_focal_length_mm, expected_gap_mm
):
    model = make_bench(catalog, [(100.0, 100.0), (second_focal_length_mm, 180.0)])
    result = model.optimise_gap(0, COLLIMATE)
    assert result.gap_after_mm == pytest.approx(expected_gap_mm, abs=1e-3)
    assert result.merit_after == pytest.approx(0.0, abs=1e-3)
    assert not result.at_bound


@pytest.mark.parametrize(("flip_second", "expected_gap_mm"), [(False, 146.84), (True, 140.49)])
def test_a_real_doublet_pair_collimates_at_the_stored_gap(catalog, flip_second, expected_gap_mm):
    """AC254-100-A then AC254-050-A, 10 mm beam at 555 nm; matches the trial in issue #38,
    which used Optiland's own optimiser (146.83 and 140.49 mm)."""
    model = make_bench(catalog, [("AC254-100-A", 100.0), ("AC254-050-A", 220.0)])
    if flip_second:
        model.flip_lens(1)
    result = model.optimise_gap(0, COLLIMATE)
    assert result.gap_after_mm == pytest.approx(expected_gap_mm, abs=0.02)
    assert result.merit_after < result.merit_before / 100


def test_optimising_does_not_change_the_bench(catalog):
    model = make_bench(catalog, [(100.0, 100.0), (50.0, 180.0)])
    state_before = copy.deepcopy(model.state)
    model.optimise_gap(0, COLLIMATE)
    assert model.state == state_before


def test_collimating_moves_the_image_plane_with_the_lenses(catalog):
    model = make_bench(catalog, [(100.0, 100.0), (50.0, 180.0)])
    image_gap_mm = model.air_gaps()[1].width_mm
    model.apply_gap_optimisation(model.optimise_gap(0, COLLIMATE), lock=True)
    assert model.air_gaps()[0].width_mm == pytest.approx(150.0, abs=1e-3)
    assert model.air_gaps()[0].locked
    assert model.air_gaps()[1].width_mm == pytest.approx(image_gap_mm)


def test_focusing_keeps_the_image_plane_where_it_is(catalog):
    model = make_bench(catalog, [("AC254-100-A", 100.0), ("AC254-050-A", 180.0)], image_z=300.0)
    result = model.optimise_gap(0, FOCUS)
    model.apply_gap_optimisation(result)
    assert model.state.image_z == 300.0
    assert model.air_gaps()[0].width_mm == pytest.approx(result.gap_after_mm)
    # The merit is the RMS spot radius on the image plane, as the spot diagram shows it.
    assert analysis.trace_spot(model.state).rms_radius * 1000 == pytest.approx(result.merit_after)
    assert result.merit_after < result.merit_before


def test_the_source_gap_collimates_a_point_source(catalog):
    """A beam diverging from a point 50 mm in front of the source plane is collimated by a
    f = 100 mm lens 100 mm from the point: 50 mm from the source plane."""
    model = make_bench(catalog, [(100.0, 150.0)])
    model.set_source_gap_locked(False)
    model.state.source_z = 50.0
    model.state.selected_field.vergence_dioptres = -20.0
    result = model.optimise_gap(SOURCE_GAP_INDEX, COLLIMATE)
    assert result.gap_after_mm == pytest.approx(50.0, abs=1e-3)


def test_a_solution_out_of_range_is_reported_at_the_bound(catalog):
    """f = 100 then f = -500 would collimate at a gap of -400 mm: the best is the shortest."""
    model = make_bench(catalog, [(100.0, 100.0), (-500.0, 150.0)])
    result = model.optimise_gap(0, COLLIMATE)
    assert result.at_bound
    assert result.gap_after_mm == pytest.approx(result.shortest_gap_mm)


def test_a_gap_that_gains_by_clipping_rays_is_not_chosen(catalog):
    """The second lens is only 2 mm across. Collimated, the beam would be 10 mm wide there,
    so only a few central rays would get through (a near-perfect merit, by clipping). With
    every ray through, the beam must be squeezed through it near the focus of the first lens."""
    model = make_bench(catalog, [(100.0, 100.0), (50.0, 180.0)], beam_diameter_mm=20.0)
    model.state.lenses[1].semi_aperture = 1.0
    result = model.optimise_gap(0, COLLIMATE)
    assert result.rays_through_after == result.rays_launched
    assert result.gap_after_mm < 110.0  # nowhere near 150 mm


@pytest.mark.parametrize(
    ("lock", "lens_index", "goal", "message"),
    [
        ("gap", 0, COLLIMATE, "locked"),
        (None, 1, COLLIMATE, "image plane"),  # the gap to the image plane
        ("image", 0, FOCUS, "image plane is locked"),
        (None, 0, "sharpen", "goal"),
        (None, SOURCE_GAP_INDEX, COLLIMATE, "source plane is locked"),
    ],
)
def test_gaps_that_cannot_be_optimised_are_refused(catalog, lock, lens_index, goal, message):
    model = make_bench(catalog, [(100.0, 100.0), (50.0, 180.0)])
    if lock == "gap":
        model.set_gap_locked(0, True)
    elif lock == "image":
        model.set_gap_locked(1, True)
    with pytest.raises(ValueError, match=message):
        optimise_gap(model.state, lens_index, goal)


def test_agents_can_ask_without_applying(catalog):
    model = make_bench(catalog, [(100.0, 100.0), (50.0, 180.0)])
    tools = DesignTools(model)
    result = tools.optimise_gap(0, "collimate", apply=False)
    assert result["optimisation"]["gap_after_mm"] == pytest.approx(150.0, abs=1e-3)
    assert result["lenses"][0]["gap_after_mm"] == pytest.approx(80.0)

    result = tools.optimise_gap(0, "collimate", lock=True)
    assert result["lenses"][0]["gap_after_mm"] == pytest.approx(150.0, abs=1e-3)
    assert result["lenses"][0]["gap_after_locked"]


# ------------------------------------------------------------------ suggesting a flip (#56)


def doublet_pair(catalog, flip_second: bool = False) -> OpticalModel:
    """AC254-100-A then AC254-050-A with a 20 mm beam: the example in issue #56."""
    model = make_bench(
        catalog, [("AC254-100-A", 100.0), ("AC254-050-A", 220.0)], beam_diameter_mm=20.0
    )
    if flip_second:
        model.flip_lens(1)
    return model


def test_flipping_the_second_doublet_is_suggested(catalog):
    """In catalogue orientation the residual is about 600 µrad; flipped, about 85 µrad."""
    model = doublet_pair(catalog)
    result = model.optimise_gap(0, COLLIMATE)
    assert result.merit_after == pytest.approx(601, abs=5)
    assert result.flip_lens_index == 1
    assert result.flipped_merit == pytest.approx(85, abs=2)
    assert result.flipped_gap_mm == pytest.approx(140.35, abs=0.02)
    assert not model.state.lenses[1].flipped  # suggested, not done


def test_no_flip_is_suggested_once_the_lens_is_the_right_way_round(catalog):
    result = doublet_pair(catalog, flip_second=True).optimise_gap(0, COLLIMATE)
    assert result.flip_lens_index is None and result.flipped_merit is None


def test_paraxial_lenses_are_never_suggested_for_flipping(catalog):
    model = make_bench(catalog, [(100.0, 100.0), (50.0, 180.0)])
    assert model.optimise_gap(0, COLLIMATE).flip_lens_index is None


def test_applying_the_suggested_flip(catalog):
    model = doublet_pair(catalog)
    result = model.optimise_gap(0, COLLIMATE)
    model.apply_gap_optimisation(result, flip=True)
    assert model.state.lenses[1].flipped
    assert model.air_gaps()[0].width_mm == pytest.approx(result.flipped_gap_mm)


def test_a_flip_that_was_not_suggested_is_refused(catalog):
    model = doublet_pair(catalog, flip_second=True)
    with pytest.raises(ValueError, match="would not do clearly better"):
        model.apply_gap_optimisation(model.optimise_gap(0, COLLIMATE), flip=True)


def test_agents_flip_only_when_they_ask(catalog):
    tools = DesignTools(doublet_pair(catalog))
    result = tools.optimise_gap(0, "collimate")
    assert result["optimisation"]["flip_lens_index"] == 1
    assert not result["lenses"][1]["flipped"]

    tools = DesignTools(doublet_pair(catalog))
    result = tools.optimise_gap(0, "collimate", flip_if_better=True)
    assert result["lenses"][1]["flipped"]
    assert result["lenses"][0]["gap_after_mm"] == pytest.approx(140.35, abs=0.02)
