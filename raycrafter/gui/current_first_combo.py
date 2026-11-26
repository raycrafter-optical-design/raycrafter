"""A drop-down whose list always opens with the current choice at the top."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox


class CurrentFirstComboBox(QComboBox):
    """A drop-down whose list opens with the current choice in its top row.

    The list opens under the pointer, so its top row is the one highlighted first. With
    the current choice anywhere else, the highlight jumped from it to the top row as the
    list opened. The other choices keep their order below it.

    Read the choice with ``currentData()`` and set it with ``findData``: an item's position
    changes each time the list opens, its data does not.
    """

    def showPopup(self) -> None:
        """Move the current choice to the top (without counting as a change), then open."""
        current_row = self.currentIndex()
        if current_row > 0:
            self.blockSignals(True)  # reordering does not change the choice
            text, data = self.itemText(current_row), self.itemData(current_row)
            tooltip = self.itemData(current_row, Qt.ItemDataRole.ToolTipRole)
            self.removeItem(current_row)
            self.insertItem(0, text, data)
            if tooltip is not None:
                self.setItemData(0, tooltip, Qt.ItemDataRole.ToolTipRole)
            self.setCurrentIndex(0)
            self.blockSignals(False)
        self.view().setMinimumWidth(self.width())  # the list is as wide as the box
        super().showPopup()
