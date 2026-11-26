"""The gap menu's optimise actions: the "Collimate…" entries and "Focus on image plane…" (#38).

Collimating is measured either on the output of the whole system ("Collimate system
output…") or on the light leaving the element behind the gap, a lens or its assembly
("Collimate after …", #66). Where those are the same, because that element is the last, there
is one entry, "Collimate output…".

Each one finds the best length for the gap with ``raycrafter.core.gap_optimisation``, then
shows the gap and the merit before and after. The user can apply it, and lock the gap
(ticked by default, as the usual next step is to keep the gap that was found). If turning
a lens next to the gap round does clearly better (#56), the dialog says so and offers
"Flip and apply"; a lens is never flipped without that. Applying is one undo step. Nothing
is calculated until the user asks, and never during a drag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QCheckBox, QMenu, QMessageBox

from raycrafter.core import gap_optimisation
from raycrafter.core.gap_optimisation import (
    COLLIMATE,
    FOCUS,
    GapOptimisation,
    element_end_behind_gap,
)
from raycrafter.core.gaps import SOURCE_GAP_INDEX

if TYPE_CHECKING:
    from raycrafter.gui.main_window import MainWindow

STATUS_MESSAGE_MS = 5000
MERIT_NAMES = {
    COLLIMATE: "RMS angle of the output rays",
    FOCUS: "RMS spot radius on the image plane",
}


@dataclass(frozen=True)
class Choice:
    """What the user chose in the result dialog: whether to flip the suggested lens and lock."""

    flip: bool
    lock: bool


class GapOptimiser:
    """Carries out the gap menu's optimise actions on the window's model.

    Parameters
    ----------
    window : MainWindow
        The window whose model and status bar are used.
    """

    def __init__(self, window: MainWindow) -> None:
        self.window = window

    def add_menu_entries(self, menu: QMenu, lens_index: int) -> None:
        """Add the "Collimate…" entries and "Focus on image plane…" to a gap's menu.

        An entry that cannot be used is greyed out, with the reason as its tooltip.
        """
        menu.setToolTipsVisible(True)
        entries = [(text, COLLIMATE, after) for text, after in self.collimate_entries(lens_index)]
        entries.append(("Focus on image plane…", FOCUS, None))
        for text, goal, measure_after in entries:
            action = menu.addAction(text)
            reason = self.reason_not_possible(lens_index, goal)
            action.setEnabled(reason is None)
            if reason is not None:
                action.setToolTip(reason)
            action.triggered.connect(
                lambda _checked=False, goal=goal, measure_after=measure_after: self.optimise(
                    lens_index, goal, measure_after
                )
            )

    def collimate_entries(self, lens_index: int) -> list[tuple[str, int | None]]:
        """The text of each collimate entry, with the lens it measures after (None: system)."""
        state = self.window.model.state
        element_end = element_end_behind_gap(state, lens_index)
        if element_end == len(state.lenses) - 1:
            return [("Collimate output…", None)]
        # "&" would mark a keyboard mnemonic in a menu, so it is doubled to show as itself.
        name = self.element_name(element_end).replace("&", "&&")
        return [
            (f"Collimate after {name}…", element_end),
            ("Collimate system output…", None),
        ]

    def element_name(self, lens_index: int) -> str:
        """What the element holding a lens is called: its assembly, else its part number."""
        model = self.window.model
        return model.assembly_name(lens_index) or model.state.lenses[lens_index].part_number

    def reason_not_possible(self, lens_index: int, goal: str) -> str | None:
        """Why this gap cannot be optimised for this goal, or None if it can."""
        if goal == FOCUS and self.window.lock_focus_button.isChecked():
            return 'Turn off "Lock focus" first: it would move the image plane'
        try:
            gap_optimisation.check_can_optimise(self.window.model.state, lens_index, goal)
        except ValueError as error:
            return str(error)
        return None

    def optimise(
        self, lens_index: int, goal: str, measure_after_lens_index: int | None = None
    ) -> None:
        """Find the best gap, show it, and apply it if the user agrees."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = self.window.model.optimise_gap(lens_index, goal, measure_after_lens_index)
        except ValueError as error:
            self.window.statusBar().showMessage(str(error), STATUS_MESSAGE_MS)
            return
        finally:
            QApplication.restoreOverrideCursor()

        choice = self.ask_to_apply(result)
        if choice is None:
            return
        self.window.model.apply_gap_optimisation(result, lock=choice.lock, flip=choice.flip)
        gap_mm = result.flipped_gap_mm if choice.flip else result.gap_after_mm
        self.window.request_update()
        self.window.statusBar().showMessage(f"Gap set to {gap_mm:.3f} mm", STATUS_MESSAGE_MS)

    def ask_to_apply(self, result: GapOptimisation) -> Choice | None:
        """Show the result and ask what to do; None if the user cancels."""
        box = QMessageBox(self.window)
        box.setWindowTitle("Focus" if result.goal == FOCUS else "Collimate")
        box.setText(self.describe(result))
        lock_box = QCheckBox("Lock the gap")
        lock_box.setChecked(True)
        box.setCheckBox(lock_box)
        flip_button = None
        if result.flip_lens_index is not None:
            flip_button = box.addButton("Flip and apply", QMessageBox.ButtonRole.AcceptRole)
        apply_button = box.addButton("Apply", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(flip_button or apply_button)
        box.exec()
        clicked = box.clickedButton()
        if clicked is None or clicked not in (flip_button, apply_button):
            return None
        return Choice(flip=clicked is flip_button, lock=lock_box.isChecked())

    def describe(self, result: GapOptimisation) -> str:
        """The result in words: which gap, the gap and merit before and after, any warning."""
        lenses = self.window.model.state.lenses
        if result.lens_index == SOURCE_GAP_INDEX:
            gap_name = f"the gap from the source plane to {lenses[0].part_number}"
        else:
            before, after = lenses[result.lens_index], lenses[result.lens_index + 1]
            gap_name = f"the gap between {before.part_number} and {after.part_number}"
        unit = result.merit_unit
        merit_name = MERIT_NAMES[result.goal]
        measured_after = result.measured_after_lens_index
        if measured_after is not None and measured_after < len(lenses) - 1:
            merit_name = f"RMS angle of the rays leaving {self.element_name(measured_after)}"
        lines = [
            f"Best length for {gap_name}:",
            "",
            f"Gap: {result.gap_before_mm:.3f} mm  →  {result.gap_after_mm:.3f} mm",
            f"{merit_name}: {format_merit(result.merit_before)} {unit}  →  "
            f"{format_merit(result.merit_after)} {unit}",
            f"Rays of the selected field that get through: {result.rays_through_after} of "
            f"{result.rays_launched}",
        ]
        if result.flip_lens_index is not None and result.flipped_merit is not None:
            flipped_part = lenses[result.flip_lens_index].part_number
            lines += [
                "",
                f"{flipped_part} turned round does clearly better: gap "
                f"{result.flipped_gap_mm:.3f} mm, {merit_name[0].lower() + merit_name[1:]} "
                f"{format_merit(result.flipped_merit)} {unit}.",
            ]
        if result.at_bound:
            lines += [
                "",
                f"The best gap is at the end of the allowed range ({result.shortest_gap_mm:.1f} "
                f"to {result.longest_gap_mm:.1f} mm), so there may be no true solution in it.",
            ]
        return "\n".join(lines)


def format_merit(value: float) -> str:
    """A merit with sensible precision: 0.012, 3.4, 601, 25457; "no light" for NaN."""
    if value != value:  # NaN: no light got through
        return "no light"
    if value < 1.0:
        return f"{value:.3f}"
    if value < 100.0:
        return f"{value:.1f}"
    return f"{value:.0f}"
