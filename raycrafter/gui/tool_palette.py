"""The tool palette: the narrow strip of tool icons down the left of the main window.

One tool is active at a time, as in a graphics editor. Move is the default; the add tools
stay active after placing something, so several can be placed in a row, until another tool
is chosen or V or Escape goes back to Move. Below a separator, Autofocus is an action, not
a tool: it runs once and leaves the active tool as it was.

What each tool does is up to the main window (see ``raycrafter.gui.tool_actions``); this
module only shows the tools and reports the user's choice.
"""

from __future__ import annotations

from enum import Enum

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QIcon, QKeySequence
from PyQt6.QtWidgets import QToolBar, QWidget

from raycrafter.gui.tool_icons import (
    autofocus_icon,
    catalogue_lens_icon,
    field_icon,
    move_icon,
    paraxial_lens_icon,
    user_part_icon,
)

PALETTE_WIDTH_PX = 50  # a little wider than the buttons, for room either side (as in Affinity)
TOOL_ICON_SIZE_PX = 24


class Tool(Enum):
    """The tools of the palette, in the order they are shown."""

    MOVE = "move"
    ADD_PARAXIAL_LENS = "add paraxial lens"
    ADD_CATALOGUE_LENS = "add catalogue lens"
    ADD_USER_PART = "add user part"
    ADD_FIELD = "add field"


# Per tool: icon maker, name for the tooltip, and keyboard shortcut. F is not used: it flips
# the lens under the pointer.
TOOL_DETAILS: dict[Tool, tuple] = {
    Tool.MOVE: (move_icon, "Move", "V"),
    Tool.ADD_PARAXIAL_LENS: (paraxial_lens_icon, "Add paraxial lens", "P"),
    Tool.ADD_CATALOGUE_LENS: (catalogue_lens_icon, "Add catalogue lens", "L"),
    Tool.ADD_USER_PART: (user_part_icon, "Add user part", "U"),
    Tool.ADD_FIELD: (field_icon, "Add field", "R"),
}

# Where the per-lens actions are, since they are not buttons.
MOVE_HINT = (
    "\nDrag a lens to move it; drag it past its neighbour to swap them."
    "\nRight-click a lens to flip it (F), delete it (Del), make a Plossl pair or an assembly."
    "\nClick a gap to type it; right-click it to lock it."
)


class ToolPalette(QToolBar):
    """The tool buttons (one checked at a time) and the Autofocus button.

    Parameters
    ----------
    parent : QWidget, optional
        The main window; the tools' keyboard shortcuts work anywhere in it.

    Signals
    -------
    tool_chosen(Tool)
        The user chose a tool, by clicking it or with its shortcut. Not emitted by
        ``set_tool``.
    autofocus_requested
        The Autofocus button was pressed.
    """

    tool_chosen = pyqtSignal(object)
    autofocus_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Tools", parent)
        self.setObjectName("toolPalette")  # styled in raycrafter.gui.style
        self.setMovable(False)
        self.setOrientation(Qt.Orientation.Vertical)
        self.setFixedWidth(PALETTE_WIDTH_PX)
        self.setIconSize(QSize(TOOL_ICON_SIZE_PX, TOOL_ICON_SIZE_PX))
        self.toggleViewAction().setVisible(False)  # no right-click menu offering to hide it

        self.tool_group = QActionGroup(self)  # exclusive: checking one unchecks the others
        self.tool_actions: dict[Tool, QAction] = {}
        for tool, (make_icon, name, shortcut) in TOOL_DETAILS.items():
            tooltip = f"{name} ({shortcut})"
            if tool == Tool.MOVE:
                tooltip += MOVE_HINT
            action = self._add_button(make_icon(), tooltip)
            action.setCheckable(True)
            action.setShortcut(QKeySequence(shortcut))
            # Shortcuts work wherever the focus is in the window, except in a text box, which
            # takes the letter keys first.
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            action.triggered.connect(lambda _checked=False, tool=tool: self.tool_chosen.emit(tool))
            self.tool_group.addAction(action)
            self.tool_actions[tool] = action
        self.tool_actions[Tool.MOVE].setChecked(True)

        self.addSeparator()
        self.autofocus_action = self._add_button(
            autofocus_icon(), "Autofocus: move the image plane to best focus"
        )
        self.autofocus_action.triggered.connect(self.autofocus_requested.emit)

    def _add_button(self, icon: QIcon, tooltip: str) -> QAction:
        action = QAction(icon, tooltip, self)
        action.setToolTip(tooltip)
        self.addAction(action)
        return action

    def current_tool(self) -> Tool:
        """The tool whose button is checked."""
        for tool, action in self.tool_actions.items():
            if action.isChecked():
                return tool
        return Tool.MOVE

    def set_tool(self, tool: Tool) -> None:
        """Check a tool's button without emitting ``tool_chosen`` (the caller acts on it)."""
        self.tool_actions[tool].setChecked(True)
