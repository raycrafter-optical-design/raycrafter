"""Air gaps between lenses (raycrafter.core.gaps) and their use by scripts and agents."""

import math

import numpy as np
import pytest

from raycrafter import Bench
from raycrafter.core.gaps import MIN_IMAGE_GAP_MM, closest_sag_difference_mm, minimum_gap_mm
from raycrafter.core.model import OpticalModel
from raycrafter.mcp_server.tools import EDITING_TOOL_NAMES, DesignTools


@pytest.fixture
def three_lenses() -> OpticalModel:
    model = OpticalModel()
    model.add_lens("LA1509-A")
    model.add_lens("LA1509-A")
    return model


def widths(model: OpticalModel) -> list[float]:
    return [gap.width_mm for gap in model.air_gaps()]


def test_gaps_run_vertex_to_vertex_and_to_the_image(three_lenses):
    state = three_lenses.state
    gaps = three_lenses.air_gaps()
    assert [gap.to_image for gap in gaps] == [False, False, True]
    assert gaps[0].start_z == state.lenses[0].back_z
    assert gaps[0].end_z == state.lenses[1].z
    assert gaps[2].end_z == state.image_z
    assert OpticalModel(add_default_lens=False).air_gaps() == []


@pytest.mark.parametrize("lens_index", [0, 1, 2])
def test_setting_a_gap_leaves_every_other_gap_unchanged(three_lenses, lens_index):
    before = widths(three_lenses)
    first_z = three_lenses.state.lenses[0].z

    three_lenses.set_gap(lens_index, 37.5)

    expected = list(before)
    expected[lens_index] = 37.5
    assert widths(three_lenses) == pytest.approx(expected, abs=1e-9)
    assert three_lenses.state.lenses[0].z == first_z


def test_a_paraxial_pair_set_to_the_sum_of_focal_lengths_is_afocal():
    model = OpticalModel(add_default_lens=False)
    for focal_length_mm in (100.0, 50.0):
        lens_count = len(model.state.lenses)
        model.insert_lens(model.make_paraxial_lens(focal_length_mm, z=100.0 + 20.0 * lens_count))
    model.set_gap(0, 150.0)

    # A 10 mm collimated beam leaves 5 mm wide, wherever the image plane is.
    for image_gap_mm in (20.0, 200.0):
        model.set_gap(1, image_gap_mm)
        heights = model.trace_fast().spot.position_y
        assert np.ptp(heights) == pytest.approx(5.0, rel=1e-3)


def test_concave_facing_surfaces_have_a_minimum_gap_above_zero():
    model = OpticalModel(add_default_lens=False)
    model.add_lens("LC1715-A")  # plano-concave
    model.add_lens("LC1715-A")
    model.flip_lens(1)
    left, right = model.state.lenses
    if closest_sag_difference_mm(left, right) == 0.0:  # concave faces turned outwards
        model.flip_lens(0)
        model.flip_lens(1)
    minimum_mm = minimum_gap_mm(model.state, 0)
    assert minimum_mm > 1.0

    before = widths(model)
    with pytest.raises(ValueError, match="overlap"):
        model.set_gap(0, minimum_mm - 0.01)
    assert widths(model) == before
    model.set_gap(0, minimum_mm)  # touching at the rim is allowed

    # The message gives a number that is itself accepted.
    with pytest.raises(ValueError) as refusal:
        model.set_gap(0, 0.0)
    shown_mm = float(str(refusal.value).split("at least ")[1].split(" mm")[0])
    model.set_gap(0, shown_mm)


@pytest.mark.parametrize("gap_mm", [-1.0, math.nan, math.inf])
def test_impossible_gaps_are_refused(three_lenses, gap_mm):
    with pytest.raises(ValueError):
        three_lenses.set_gap(0, gap_mm)


def test_the_image_plane_stays_behind_the_last_lens(three_lenses):
    with pytest.raises(ValueError, match="image plane"):
        three_lenses.set_gap(2, MIN_IMAGE_GAP_MM / 2)
    three_lenses.set_gap(2, MIN_IMAGE_GAP_MM)


def test_the_gap_inside_a_plossl_pair_cannot_be_set_but_others_move_the_pair(three_lenses):
    three_lenses.set_gap(1, 5.0)  # close enough to pair
    three_lenses.make_plossl(1)  # the two LA1509-A
    assert [gap.in_assembly for gap in three_lenses.air_gaps()] == [False, True, False]
    with pytest.raises(ValueError, match="assembly"):
        three_lenses.set_gap(1, 20.0)

    pair_gap = widths(three_lenses)[1]
    three_lenses.set_gap(0, 60.0)
    assert widths(three_lenses)[:2] == pytest.approx([60.0, pair_gap])


def test_there_is_no_gap_behind_a_missing_lens(three_lenses):
    for lens_index in (-1, 3):
        with pytest.raises(IndexError):
            three_lenses.set_gap(lens_index, 10.0)


def test_agent_tool_and_bench_set_and_report_gaps():
    assert "set_gap" in EDITING_TOOL_NAMES
    tools = DesignTools(OpticalModel())
    design = tools.set_gap(0, 123.0)
    assert design["lenses"][0]["gap_after_mm"] == 123.0
    with pytest.raises(ValueError, match="no lens 4"):
        tools.set_gap(4, 10.0)

    bench = Bench()
    assert bench.set_gap(0, 80.0).gap_after_mm == pytest.approx(80.0)
    assert bench.image_z_mm - bench.lenses[0].back_z_mm == pytest.approx(80.0)
