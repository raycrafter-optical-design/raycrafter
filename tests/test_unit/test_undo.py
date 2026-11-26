"""Undo and redo (raycrafter.core.history and the model's use of it)."""

import copy

import pytest

from raycrafter.core.history import MAX_UNDO_STEPS, MERGE_WINDOW_S, UndoHistory
from raycrafter.core.model import OpticalModel
from raycrafter.core.state import OpticalState


class FakeClock:
    """A clock that moves only when told to."""

    def __init__(self) -> None:
        self.now_s = 100.0

    def __call__(self) -> float:
        return self.now_s


@pytest.fixture
def model() -> OpticalModel:
    """Two identical doublets, close enough to pair, with nothing to undo yet."""
    model = OpticalModel()
    model.add_lens("ACT508-200-A")
    model.clear_undo_history()
    return model


EDITS = {
    "add a lens": lambda model: model.add_lens("LA1509-A"),
    "delete a lens": lambda model: model.delete_lens(0),
    "flip a lens": lambda model: model.flip_lens(0),
    "move a lens": lambda model: model.move_lens(0, 60.0),
    "swap lenses": lambda model: (model.flip_lens(0), model.move_lens(0, 300.0)),
    "pair lenses": lambda model: model.make_plossl(0),
    "add a field": lambda model: model.add_field(),
    "edit a field": lambda model: setattr(model.state.selected_field, "angle_deg", 2.0),
    "move the image": lambda model: setattr(model.state, "image_z", 400.0),
}


@pytest.mark.parametrize("edit", EDITS.values(), ids=EDITS.keys())
def test_undo_then_redo_returns_identical_states(model, edit):
    before = copy.deepcopy(model.state)
    edit(model)
    model.record_undo_step()
    after = copy.deepcopy(model.state)
    assert after != before

    assert model.undo()
    assert model.state == before
    assert model.redo()
    assert model.state == after


def test_many_movements_recorded_once_are_one_step(model):
    start_z = model.state.lenses[0].z
    for requested_z in (90.0, 80.0, 70.0, 60.0):  # a drag
        model.move_lens(0, requested_z)
    model.record_undo_step()  # the release

    model.undo()
    assert model.state.lenses[0].z == start_z
    assert not model.history.can_undo


def test_nothing_changed_makes_no_step_and_keeps_redo(model):
    model.flip_lens(0)
    model.record_undo_step()
    model.undo()

    model.record_undo_step()
    model.state.image_z += 1e-6  # autofocus repeated on an unchanged bench
    model.record_undo_step()
    assert model.history.can_redo
    assert not model.history.can_undo


def test_a_new_change_discards_redo(model):
    model.flip_lens(0)
    model.record_undo_step()
    model.undo()
    model.flip_lens(1)
    model.record_undo_step()
    assert not model.history.can_redo
    assert not model.redo()


def test_selecting_a_field_is_not_a_step_and_undo_shows_the_edited_field(model):
    model.add_field()
    model.record_undo_step()

    model.select_field(0)
    model.state.image_z += 5.0  # as "Lock focus" would, for the newly selected field
    model.record_undo_step()
    model.state.selected_field.angle_deg = 3.0
    model.record_undo_step()

    model.undo()
    assert model.state.selected_field_index == 0
    assert model.state.selected_field.angle_deg == 0.0
    model.undo()  # the field being added: the selection and refocus were no step
    assert len(model.state.fields) == 1


def test_changes_with_one_merge_key_in_quick_succession_are_one_step():
    clock = FakeClock()
    state = OpticalState()
    history = UndoHistory(state, clock=clock)

    for angle in (0.1, 0.2, 0.3):
        state.fields[0].angle_deg = angle
        history.record(state, merge_key="angle")
        clock.now_s += 0.5 * MERGE_WINDOW_S
    clock.now_s += MERGE_WINDOW_S  # a pause
    state.fields[0].angle_deg = 0.4
    history.record(state, merge_key="angle")
    state.fields[0].diameter_mm = 4.0  # another value, straight away
    history.record(state, merge_key="diameter")

    assert history.undo().fields[0].diameter_mm == 10.0
    assert history.undo().fields[0].angle_deg == 0.3
    assert history.undo().fields[0].angle_deg == 0.0
    assert history.undo() is None


def test_a_merged_change_that_ends_where_it_began_leaves_no_step():
    state = OpticalState()
    history = UndoHistory(state, clock=FakeClock())
    for angle in (0.1, 0.0):
        state.fields[0].angle_deg = angle
        history.record(state, merge_key="angle")
    assert not history.can_undo


def test_only_the_latest_steps_are_kept():
    state = OpticalState()
    history = UndoHistory(state)
    for step in range(MAX_UNDO_STEPS + 5):
        state.image_z += 1.0
        history.record(state)
    undone = 0
    while history.undo() is not None:
        undone += 1
    assert undone == MAX_UNDO_STEPS


def test_undoing_back_to_the_saved_design_leaves_nothing_unsaved(model):
    model.mark_saved()
    model.flip_lens(0)
    model.record_undo_step()
    assert model.has_unsaved_changes()
    model.undo()
    assert not model.has_unsaved_changes()


def test_new_and_load_clear_the_history(model, tmp_path):
    file_path = str(tmp_path / "design.zmx")
    model.save_zmx(file_path)
    for start_again in (model.new_design, lambda: model.load_zmx(file_path)):
        model.state.image_z += 10.0
        model.record_undo_step()
        start_again()
        assert not model.history.can_undo
        assert not model.history.can_redo
