"""Undo and redo: snapshots of the bench, taken after each user action. It has no Qt code.

``UndoHistory`` does not watch the bench. A front end tells it when an action is complete
(``record``), e.g. when a drag ends rather than on every movement, so that one action is one
step. Because whole ``OpticalState`` copies are kept, anything stored in the state can be
undone without code of its own here.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Hashable
from dataclasses import replace

from raycrafter.core.state import OpticalState

MAX_UNDO_STEPS = 200  # older steps are forgotten
MERGE_WINDOW_S = 1.0  # changes with the same merge key this close together make one step

# Autofocus repeated on an unchanged bench lands within a tiny distance of its last answer,
# not exactly on it. An image plane that has moved less than this has not moved.
IMAGE_Z_TOLERANCE_MM = 1e-3


def _same_apart_from_image(first: OpticalState, second: OpticalState) -> bool:
    """True if two states differ at most in the image plane and the selected field.

    Everything else in the state counts (lenses, fields, locks, the source plane, ...), so
    that what is added to ``OpticalState`` later is undoable without changes here.
    """

    def comparable(state: OpticalState) -> OpticalState:
        return replace(state, image_z=0.0, selected_field_index=0)

    return comparable(first) == comparable(second)


def same_design(first: OpticalState, second: OpticalState) -> bool:
    """True if there is nothing to undo between two states (the selected field is ignored)."""
    image_shift_mm = abs(first.image_z - second.image_z)
    return _same_apart_from_image(first, second) and image_shift_mm <= IMAGE_Z_TOLERANCE_MM


class UndoHistory:
    """The states that can be returned to by undo and redo.

    Parameters
    ----------
    state : OpticalState
        The starting point: there is nothing to undo before it.
    clock : callable, optional
        Returns the time in seconds; used to merge changes made in quick succession.
    """

    def __init__(self, state: OpticalState, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self.reset(state)

    def reset(self, state: OpticalState) -> None:
        """Forget every step and start again from ``state``."""
        self._current = copy.deepcopy(state)
        self._undo_states: list[OpticalState] = []
        self._redo_states: list[OpticalState] = []
        self._merge_key: Hashable | None = None  # of the last step, while it can still grow
        self._merge_time_s = 0.0

    @property
    def can_undo(self) -> bool:
        """True if there is a step to undo."""
        return bool(self._undo_states)

    @property
    def can_redo(self) -> bool:
        """True if there is an undone step to redo."""
        return bool(self._redo_states)

    def record(self, state: OpticalState, merge_key: Hashable | None = None) -> None:
        """Note the bench as it is after an action. Nothing happens if nothing changed.

        Selecting another field is not a step of its own, even though "Lock focus" may then
        move the image plane to that field's focus.

        Parameters
        ----------
        state : OpticalState
            The bench now.
        merge_key : hashable, optional
            Names what was changed, e.g. one value in a spin box. Changes with the same key
            less than MERGE_WINDOW_S apart become a single step. Leave it out for actions
            that are always a step of their own.
        """
        selection_changed = state.selected_field_index != self._current.selected_field_index
        if selection_changed and _same_apart_from_image(state, self._current):
            self.amend(state)
            return
        if same_design(state, self._current):
            self._current = replace(self._current, selected_field_index=state.selected_field_index)
            return

        now = self._clock()
        merging = (
            merge_key is not None
            and merge_key == self._merge_key
            and now - self._merge_time_s <= MERGE_WINDOW_S
        )
        if not merging:
            self._undo_states.append(self._current)
            del self._undo_states[:-MAX_UNDO_STEPS]
        self._current = copy.deepcopy(state)
        self._redo_states.clear()
        self._merge_key, self._merge_time_s = merge_key, now

        # A merged change that ends where it started (a value stepped up and back down)
        # leaves nothing to undo.
        if merging and same_design(self._current, self._undo_states[-1]):
            self._current = self._undo_states.pop()
            self._stop_merging()

    def amend(self, state: OpticalState) -> None:
        """Replace the current snapshot without making a step; redo remains available.

        For changes that are not user actions, e.g. "Lock focus" refocusing after an undo.
        """
        self._current = copy.deepcopy(state)
        self._stop_merging()

    def forget_imported_source(self) -> None:
        """Drop the record of the imported file from every snapshot, so undo cannot bring it
        back (see ``OpticalModel.save_project(forget_import=True)``)."""
        for snapshots in (self._undo_states, self._redo_states):
            snapshots[:] = [replace(state, imported_from=None) for state in snapshots]
        self._current = replace(self._current, imported_from=None)

    def undo(self) -> OpticalState | None:
        """Step back, and return a copy of the state to show (None if there is none)."""
        return self._step(self._undo_states, self._redo_states)

    def redo(self) -> OpticalState | None:
        """Step forward again, and return a copy of the state to show (None if there is none)."""
        return self._step(self._redo_states, self._undo_states)

    def _step(
        self, source: list[OpticalState], destination: list[OpticalState]
    ) -> OpticalState | None:
        """Move the current snapshot onto ``destination`` and take the next from ``source``."""
        if not source:
            return None
        destination.append(self._current)
        self._current = source.pop()
        self._stop_merging()
        return copy.deepcopy(self._current)

    def _stop_merging(self) -> None:
        """The next change starts a new step."""
        self._merge_key = None
