"""Makes Undo and Redo work while a number box has the keyboard."""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtGui import QKeyEvent, QKeySequence
from PyQt6.QtWidgets import QAbstractSpinBox


def is_undo_or_redo(event: QKeyEvent) -> bool:
    """True for the keys of Edit > Undo or Redo."""
    return (
        event.matches(QKeySequence.StandardKey.Undo)
        or event.matches(QKeySequence.StandardKey.Redo)
        or QKeySequence(event.keyCombination()) == QKeySequence("Ctrl+Shift+Z")
    )


class _LeaveUndoKeys(QObject):
    """Event filter that stops a spin box claiming the undo and redo keys.

    Qt asks the focused widget first whether it wants a shortcut key (``ShortcutOverride``).
    A spin box's text editor says yes to undo, but has nothing useful to undo: its history is
    lost each time the value is stepped. Refusing here lets the window's Edit menu act.
    """

    def eventFilter(self, watched, event) -> bool:
        """Refuse a ShortcutOverride for undo or redo; pass everything else on."""
        if event.type() == QEvent.Type.ShortcutOverride and is_undo_or_redo(event):
            event.ignore()
            return True  # the spin box never sees it, so cannot accept it
        return False


def leave_undo_keys_to_window(spin_box: QAbstractSpinBox) -> None:
    """Have the undo and redo keys undo the design even while ``spin_box`` has the keyboard."""
    spin_box.installEventFilter(_LeaveUndoKeys(spin_box))  # the box owns, and so keeps, it
