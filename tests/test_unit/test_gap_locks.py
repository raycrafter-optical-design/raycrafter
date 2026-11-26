"""Locked gaps (issue #36): lenses joined by locked gaps keep their spacing and move as one."""

import pytest

from raycrafter import Bench
from raycrafter.core.model import OpticalModel
from raycrafter.mcp_server.tools import DesignTools


@pytest.fixture
def three_lenses() -> OpticalModel:
    """Three paraxial lenses at z = 100, 150 and 200 mm, the image plane at 300 mm."""
    model = OpticalModel(add_default_lens=False)
    for front_z in (100.0, 150.0, 200.0):
        model.insert_lens(model.make_paraxial_lens(100.0, z=front_z))
    model.state.image_z = 300.0
    return model


def widths(model: OpticalModel) -> list[float]:
    return [gap.width_mm for gap in model.air_gaps()]


def front_positions(model: OpticalModel) -> list[float]:
    return [lens.z for lens in model.state.lenses]


@pytest.mark.parametrize("dragged_index", [0, 1])
def test_dragging_either_lens_of_a_locked_pair_keeps_their_gap(three_lenses, dragged_index):
    three_lenses.set_gap_locked(0, True)
    three_lenses.move_lens(dragged_index, three_lenses.state.lenses[dragged_index].z + 20.0)
    assert front_positions(three_lenses) == pytest.approx([120.0, 170.0, 200.0])

    # Pushed further, the pair stops against the third lens rather than overlapping it.
    three_lenses.move_lens(0, 160.0)
    assert front_positions(three_lenses) == pytest.approx([150.0, 200.0, 200.0])
    assert widths(three_lenses)[0] == pytest.approx(50.0)


def test_lock_state_survives_swapping_blocks(three_lenses):
    three_lenses.set_gap_locked(0, True)  # lenses 0 and 1 form a block
    new_order = three_lenses.move_lens(2, 100.0)  # past the middle of the block
    assert new_order == [2, 0, 1]
    lenses = three_lenses.state.lenses
    assert [lens.locked_to_next for lens in lenses] == [False, True, False]
    assert lenses[2].z - lenses[1].z == pytest.approx(50.0)
    assert three_lenses.block_indices(2) == [1, 2]


def test_a_locked_gap_cannot_be_typed_until_unlocked(three_lenses):
    three_lenses.set_gap_locked(1, True)
    assert [gap.locked for gap in three_lenses.air_gaps()] == [False, True, False]
    with pytest.raises(ValueError, match="locked"):
        three_lenses.set_gap(1, 30.0)
    three_lenses.set_gap(0, 30.0)  # other gaps still can be, carrying the locked pair along
    assert widths(three_lenses)[1] == pytest.approx(50.0)
    three_lenses.set_gap_locked(1, False)
    three_lenses.set_gap(1, 30.0)
    assert widths(three_lenses)[1] == pytest.approx(30.0)


def test_a_new_lens_cannot_split_a_locked_gap(three_lenses):
    three_lenses.set_gap_locked(0, True)
    # The free stretch nearest 128 mm is now the one between lenses 1 and 2.
    assert three_lenses.free_front_z(0.0, 128.0) == pytest.approx(150.0)
    with pytest.raises(ValueError, match="locked"):
        three_lenses.insert_lens(three_lenses.make_paraxial_lens(50.0, z=128.0))
    assert len(three_lenses.state.lenses) == 3


def test_a_locked_image_gap_moves_the_image_with_the_last_lens(three_lenses):
    three_lenses.set_gap_locked(2, True)
    three_lenses.move_lens(2, 250.0)
    assert three_lenses.state.image_z == pytest.approx(350.0)

    with pytest.raises(ValueError, match="image plane is locked"):
        three_lenses.autofocus()
    with pytest.raises(ValueError, match="image plane is locked"):
        DesignTools(three_lenses).set_image_z(400.0)
    with pytest.raises(ValueError, match="image plane is locked"):
        three_lenses.add_lens("AC254-050-A")
    # Nowhere behind the last lens is free for a new one.
    assert three_lenses.free_front_z(0.0, 320.0) <= three_lenses.state.lenses[-1].z


def test_deleting_a_lens_removes_the_locks_either_side(three_lenses):
    for lens_index in range(3):
        three_lenses.set_gap_locked(lens_index, True)
    three_lenses.delete_lens(1)
    assert [gap.locked for gap in three_lenses.air_gaps()] == [False, True]
    three_lenses.delete_lens(1)  # the last lens: its image-gap lock goes too
    assert [gap.locked for gap in three_lenses.air_gaps()] == [False]


def test_a_plossl_pair_is_locked_and_splitting_it_unlocks_it():
    model = OpticalModel()
    model.make_plossl(0)
    assert model.air_gaps()[0].locked and model.air_gaps()[0].in_assembly
    with pytest.raises(ValueError, match="always locked"):
        model.set_gap_locked(0, False)
    model.split_assembly(0)
    assert not model.air_gaps()[0].locked
    assert model.block_indices(0) == [0]


def test_locks_count_as_unsaved_changes_and_can_be_undone(three_lenses):
    three_lenses.mark_saved()
    three_lenses.clear_undo_history()
    three_lenses.set_gap_locked(0, True)
    three_lenses.record_undo_step()
    assert three_lenses.has_unsaved_changes()
    three_lenses.undo()
    assert not three_lenses.air_gaps()[0].locked
    assert not three_lenses.has_unsaved_changes()


def test_scripts_and_agents_lock_gaps():
    bench = Bench(empty=True)
    bench.add_paraxial_lens(100.0)
    bench.add_paraxial_lens(50.0)
    assert bench.lock_gap(1).gap_after_locked  # the gap to the image plane
    assert not bench.lenses[0].gap_after_locked
    with pytest.raises(ValueError, match="locked"):
        bench.autofocus()
    bench.lock_gap(1, locked=False)
    bench.autofocus()

    tools = DesignTools(OpticalModel())
    assert tools.lock_gap(0)["lenses"][0]["gap_after_locked"] is True
