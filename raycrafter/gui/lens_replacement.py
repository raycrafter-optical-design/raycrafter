"""The lens menu's replace actions: swap a lens on the bench for another, in place (#63).

- "Replace with same f…" opens the lens picker showing only lenses of about the same focal
  length (same sign, within ``replacement.SAME_FOCAL_LENGTH_TOLERANCE``).
- "Swap lens…" opens the lens picker with no focal length filter, to swap for any lens.
- "Replace with paraxial" puts an ideal thin lens of the same focal length in its place.

In both pickers the coating filter starts at "suits the beams". Where the new lens goes is
decided by ``raycrafter.core.replacement``. If the lens is one of a Plossl pair, the user is
asked whether to replace both lenses (it stays a Plossl pair) or to split the pair and
replace only this one.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QMenu, QMessageBox

from raycrafter.gui.lens_picker import LensPickerDialog

if TYPE_CHECKING:
    from raycrafter.gui.main_window import MainWindow

STATUS_MESSAGE_MS = 5000
REPLACE_BOTH = "both"  # the Plossl question's answers
SPLIT_PAIR = "split"


class LensReplacer:
    """Carries out the replace actions of the lens menu on the window's model.

    Parameters
    ----------
    window : MainWindow
        The window whose model, side view and status bar are used.
    """

    def __init__(self, window: MainWindow) -> None:
        self.window = window
        # Made on first use, then kept, so its filters stay as the user left them. It is not
        # the "Add catalogue lens" picker, whose focal length filter it would otherwise reset.
        self.picker: LensPickerDialog | None = None

    def add_menu_entries(self, menu: QMenu, lens_index: int) -> None:
        """Add the three replace entries to the lens menu."""
        same_action = menu.addAction("Replace with same f…")
        same_action.triggered.connect(lambda: self.replace_with_catalogue_lens(lens_index, True))
        swap_action = menu.addAction("Swap lens…")
        swap_action.triggered.connect(lambda: self.replace_with_catalogue_lens(lens_index, False))

        paraxial_action = menu.addAction("Replace with paraxial")
        lens = self.window.model.state.lenses[lens_index]
        paraxial_action.setEnabled(not lens.is_paraxial)
        paraxial_action.triggered.connect(lambda: self.replace_with_paraxial(lens_index))

    def replace_with_catalogue_lens(self, lens_index: int, same_focal_length: bool) -> None:
        """Let the user pick a catalogue lens, then put it in place of this one.

        Parameters
        ----------
        lens_index : int
            The lens to replace.
        same_focal_length : bool
            If True, the picker shows only lenses of about the same focal length.
        """
        model = self.window.model
        lens = model.state.lenses[lens_index]
        focal_length_mm = model.lens_focal_length_mm(lens_index) if same_focal_length else None
        if focal_length_mm is not None and not math.isfinite(focal_length_mm):
            focal_length_mm = None  # a lens with no power: there is no focal length to match

        if self.picker is None:
            self.picker = LensPickerDialog(model.catalog, self.window, action_text="Replace")
        self.picker.set_design_wavelengths([beam.wavelength_nm for beam in model.state.fields])
        self.picker.prepare_for_replacement(lens.part_number, focal_length_mm)
        if self.picker.exec() != LensPickerDialog.DialogCode.Accepted:
            return
        part_number = self.picker.selected_part_number()
        if part_number is None:
            return

        choice = None
        if model.in_plossl_pair(lens_index):
            choice = self.ask_about_plossl_pair(
                can_replace_both=not model.catalog[part_number].is_negative
            )
            if choice is None:
                return
        if choice == REPLACE_BOTH:
            self.apply(
                lambda: model.replace_plossl_pair(lens_index, part_number),
                f"Replaced the Plossl pair with two {part_number}",
            )
            return
        new_lens = model.make_lens(part_number)
        self.apply(
            lambda: model.replace_lens(
                lens_index, new_lens, split_plossl_pair=choice == SPLIT_PAIR
            ),
            f"Replaced {lens.part_number} with {part_number}",
        )

    def replace_with_paraxial(self, lens_index: int) -> None:
        """Put an ideal thin lens of the same focal length in place of this lens."""
        model = self.window.model
        lens = model.state.lenses[lens_index]
        # Rounded to 0.1 mm, so the new lens is labelled "Paraxial f=+200 mm", not +199.948.
        focal_length_mm = round(model.lens_focal_length_mm(lens_index), 1)
        if not math.isfinite(focal_length_mm):
            self.window.statusBar().showMessage(
                f"{lens.part_number} has no power, so it has no paraxial equivalent",
                STATUS_MESSAGE_MS,
            )
            return
        in_pair = model.in_plossl_pair(lens_index)
        if in_pair and self.ask_about_plossl_pair(can_replace_both=False) is None:
            return
        new_lens = model.make_paraxial_lens(focal_length_mm)
        self.apply(
            lambda: model.replace_lens(lens_index, new_lens, split_plossl_pair=in_pair),
            f"Replaced {lens.part_number} with a paraxial lens of f = {focal_length_mm:.1f} mm",
        )

    def ask_about_plossl_pair(self, can_replace_both: bool) -> str | None:
        """Ask what to do about a Plossl pair one of whose lenses is being replaced.

        Returns REPLACE_BOTH, SPLIT_PAIR, or None if the user cancelled. Replacing both is
        offered only if ``can_replace_both`` (a Plossl pair needs a positive catalogue lens).
        """
        box = QMessageBox(self.window)
        box.setWindowTitle("Replace a lens of a Plossl pair")
        text = "This lens is one of a Plossl pair."
        if can_replace_both:
            text += " Replace both lenses, so that it stays a Plossl pair, or split the pair?"
        else:
            text += " Split the pair and replace only this lens?"
        box.setText(text)
        both_button = None
        if can_replace_both:
            both_button = box.addButton("Replace both", QMessageBox.ButtonRole.AcceptRole)
        split_button = box.addButton("Split pair", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(both_button or split_button)
        box.exec()
        if both_button is not None and box.clickedButton() is both_button:
            return REPLACE_BOTH
        if box.clickedButton() is split_button:
            return SPLIT_PAIR
        return None

    def apply(self, replace: Callable[[], None], done_message: str) -> None:
        """Run a replacement, then redraw; if it is refused, say why (nothing has changed)."""
        try:
            replace()
        except ValueError as error:
            self.window.statusBar().showMessage(str(error), STATUS_MESSAGE_MS)
            return
        self.window.side_view.rebuild_lens_graphics()
        self.window.request_update()
        self.window.statusBar().showMessage(done_message, STATUS_MESSAGE_MS)
