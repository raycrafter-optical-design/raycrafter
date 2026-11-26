"""The context bar: the strip across the top of the main window whose controls follow the tool.

As in a graphics editor, the bar shows the settings of the active tool (``show_page``), e.g.
the focal length of the next paraxial lens. Whatever the tool, the focus-lock button sits at
the far right, after a separator.

On macOS the bar is drawn in the window's title bar, as Affinity Photo does
(``put_in_title_bar``); a press on an empty part of it moves the window, as the title bar
would. The bar keeps clear of the window buttons (red, yellow, green) at its left end.
"""

from __future__ import annotations

import sys

from PyQt6.QtCore import QEvent, QObject, QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QMainWindow,
    QSizePolicy,
    QStackedWidget,
    QToolBar,
    QToolButton,
    QWidget,
)

from raycrafter.gui.tool_icons import ITALIC_F, focus_lock_icon
from raycrafter.gui.tool_palette import Tool
from raycrafter.gui.undo_keys import leave_undo_keys_to_window

BAR_HEIGHT_PX = 44
WINDOW_BUTTONS_WIDTH_PX = 78  # room left at the bar's left end for the macOS window buttons
FOCAL_LENGTH_LIMIT_MM = 9999.9  # the box takes any focal length up to this, either sign
FOCAL_LENGTH_BOX_WIDTH_PX = 96
USER_PART_BOX_WIDTH_PX = 200
LOCK_ICON_SIZE = QSize(36, 24)
NO_USER_PARTS = "No user parts saved yet"

# Qt 6.9 added Qt::ExpandedClientAreaHint, which lets the window's content extend under the
# title bar. PyQt6 has no name for it yet: it reuses the bit of the obsolete
# MaximizeUsingFullscreenGeometryHint, so that name is used for it here.
EXPANDED_CLIENT_AREA_HINT = Qt.WindowType.MaximizeUsingFullscreenGeometryHint


def put_in_title_bar(window: QMainWindow) -> None:
    """On macOS, extend the window's content under its title bar, so the bar sits there.

    Call before the window is shown. Elsewhere this does nothing and the bar is an ordinary
    toolbar under the title bar.
    """
    if sys.platform != "darwin":
        return
    window.setWindowFlag(EXPANDED_CLIENT_AREA_HINT, True)
    window.setWindowFlag(Qt.WindowType.NoTitleBarBackgroundHint, True)
    # Qt would otherwise push the content back down out of the title bar (its "safe area").
    window.setAttribute(Qt.WidgetAttribute.WA_ContentsMarginsRespectsSafeArea, False)


class TitleBarDragger(QObject):
    """Event filter: a press on the bar's empty space moves the window, as a title bar does.

    With the content under the title bar, Qt receives every click there, so macOS never sees
    one with which to move the window. ``startSystemMove`` hands the drag to the system. A
    double-click zooms the window, as double-clicking a title bar does.
    """

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Start a window move on a left press; zoom on a double-click; pass on the rest."""
        if not isinstance(watched, QWidget):
            return False
        window = watched.window()
        if (
            event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
        ):
            window.windowHandle().startSystemMove()
            return True
        if event.type() == QEvent.Type.MouseButtonDblClick:
            if window.isMaximized():
                window.showNormal()
            else:
                window.showMaximized()
            return True
        return False


class ContextBar(QToolBar):
    """The tool's settings on the left, the focus-lock button on the right.

    Parameters
    ----------
    parent : QWidget, optional
        The main window.

    Signals
    -------
    paraxial_focal_length_changed(float)
        The focal length for the next paraxial lens was changed (mm; may be 0).
    user_part_activated(str)
        A user part was chosen from the drop-down (even the one already shown).
    """

    paraxial_focal_length_changed = pyqtSignal(float)
    user_part_activated = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Context", parent)
        self.setObjectName("contextBar")  # styled in raycrafter.gui.style
        self.setMovable(False)
        self.setFixedHeight(BAR_HEIGHT_PX)
        self.toggleViewAction().setVisible(False)  # no right-click menu offering to hide it

        self.dragger = TitleBarDragger(self)
        self.installEventFilter(self.dragger)  # the gaps between the bar's widgets
        if sys.platform == "darwin":
            window_buttons_room = QWidget()
            window_buttons_room.setFixedWidth(WINDOW_BUTTONS_WIDTH_PX)
            window_buttons_room.installEventFilter(self.dragger)
            self.addWidget(window_buttons_room)

        # One page of controls per tool; tools without settings share the empty page.
        self.pages = QStackedWidget()
        self.empty_page = self._new_page()
        self.paraxial_page = self._build_paraxial_page()
        self.user_part_page = self._build_user_part_page()
        self.page_of_tool = {
            Tool.ADD_PARAXIAL_LENS: self.paraxial_page,
            Tool.ADD_USER_PART: self.user_part_page,
        }
        self.addWidget(self.pages)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        spacer.installEventFilter(self.dragger)
        self.addWidget(spacer)
        self.addSeparator()

        self.lock_focus_button = QToolButton()
        self.lock_focus_button.setObjectName("lockFocusButton")
        self.lock_focus_button.setCheckable(True)  # off by default
        self.lock_focus_button.setIconSize(LOCK_ICON_SIZE)
        self.lock_focus_button.setToolTip(
            "Lock focus: keep the image plane at best focus as lenses or the beams change, "
            "whenever that focus is in view. Not available while the gap to the image plane "
            "is locked."
        )
        self.lock_focus_button.toggled.connect(self._show_lock_state)
        self._show_lock_state(False)
        self.addWidget(self.lock_focus_button)

    # ------------------------------------------------------------------ pages

    def _new_page(self) -> QHBoxLayout:
        """Add an empty page to the stack; return its layout, to fill with controls."""
        page = QWidget()
        page.installEventFilter(self.dragger)  # its empty space moves the window too
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.pages.addWidget(page)
        return layout

    def _build_paraxial_page(self) -> QHBoxLayout:
        """A sign button and the focal length of the next paraxial lens."""
        layout = self._new_page()
        self.sign_button = QToolButton()
        self.sign_button.setObjectName("signButton")
        self.sign_button.setText("±")
        self.sign_button.setToolTip("Change the sign of the focal length")
        layout.addWidget(self.sign_button)

        focal_length_label = QLabel(f"{ITALIC_F} =")
        focal_length_label.setObjectName("italicF")
        layout.addWidget(focal_length_label)

        self.focal_length_box = QDoubleSpinBox()
        self.focal_length_box.setRange(-FOCAL_LENGTH_LIMIT_MM, FOCAL_LENGTH_LIMIT_MM)
        self.focal_length_box.setDecimals(1)
        self.focal_length_box.setSuffix(" mm")
        self.focal_length_box.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.focal_length_box.setFixedWidth(FOCAL_LENGTH_BOX_WIDTH_PX)
        self.focal_length_box.setToolTip(
            "Focal length of the next paraxial (ideal thin) lens; negative for a diverging lens"
        )
        leave_undo_keys_to_window(self.focal_length_box)
        layout.addWidget(self.focal_length_box)

        self.sign_button.clicked.connect(
            lambda: self.focal_length_box.setValue(-self.focal_length_box.value())
        )
        self.focal_length_box.valueChanged.connect(self.paraxial_focal_length_changed.emit)
        return layout

    def _build_user_part_page(self) -> QHBoxLayout:
        """A drop-down of the saved user parts (see ``set_user_parts``)."""
        layout = self._new_page()
        part_label = QLabel("Part")
        layout.addWidget(part_label)
        self.user_part_box = QComboBox()
        # A plain list view: the native macOS popup ignores the style sheet (see FieldEditor).
        self.user_part_box.setView(QListView())
        self.user_part_box.setFixedWidth(USER_PART_BOX_WIDTH_PX)
        self.user_part_box.setToolTip(
            'Assemblies saved with "Save as user part" (right-click an assembly)'
        )
        self.user_part_box.activated.connect(
            lambda row: self.user_part_activated.emit(self.user_part_box.itemData(row))
        )
        # The bar's boxes are all one height: a combo box is otherwise taller than a spin box.
        self.user_part_box.setFixedHeight(self.focal_length_box.sizeHint().height())
        layout.addWidget(self.user_part_box)
        return layout

    def show_page(self, tool: Tool) -> None:
        """Show the settings of a tool (an empty bar for a tool without any)."""
        layout = self.page_of_tool.get(tool, self.empty_page)
        self.pages.setCurrentWidget(layout.parentWidget())

    # ------------------------------------------------------------------ values

    def paraxial_focal_length_mm(self) -> float:
        """The focal length set for the next paraxial lens, in mm (0 is not a lens)."""
        return self.focal_length_box.value()

    def set_paraxial_focal_length_mm(self, focal_length_mm: float) -> None:
        """Set the focal length box (emits ``paraxial_focal_length_changed``)."""
        self.focal_length_box.setValue(focal_length_mm)

    def set_user_parts(self, part_labels: list[tuple[str, str]]) -> None:
        """Fill the user part drop-down with (label, part name) pairs, keeping the choice."""
        previous_name = self.user_part_box.currentData()
        self.user_part_box.clear()
        for label, name in part_labels:
            self.user_part_box.addItem(label, name)
        if not part_labels:
            self.user_part_box.addItem(NO_USER_PARTS, None)
        self.user_part_box.setEnabled(bool(part_labels))
        row = self.user_part_box.findData(previous_name)
        self.user_part_box.setCurrentIndex(max(row, 0))

    def chosen_user_part(self) -> str | None:
        """The name of the user part shown in the drop-down, or None if there are none."""
        return self.user_part_box.currentData()

    def _show_lock_state(self, locked: bool) -> None:
        """A closed red padlock while focus is locked, an open grey one otherwise."""
        self.lock_focus_button.setIcon(focus_lock_icon(locked))
