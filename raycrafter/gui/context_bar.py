"""The context bar: the strip across the top of the main window whose controls follow the tool.

As in a graphics editor, the bar shows the settings of the active tool (``show_page``), e.g.
the focal length and diameter of the next paraxial lens, or of the selected paraxial lens.
Whatever the tool, the focus-lock button sits at the far right, after a separator.

On macOS the bar is drawn in the window's title bar, as Affinity Photo does
(``put_in_title_bar``); a press on an empty part of it moves the window, as the title bar
would. The bar keeps clear of the window buttons (red, yellow, green) at its left end. macOS
would draw the window title over the bar's controls, so the title is hidden there
(``hide_title_text``) and the bar shows the document's name itself, left of the focus lock.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys

from PyQt6.QtCore import QEvent, QObject, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QComboBox,
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

from raycrafter.gui.scaled_slider import ScaledSlider
from raycrafter.gui.tool_icons import ITALIC_F, focus_lock_icon
from raycrafter.gui.tool_palette import Tool

BAR_HEIGHT_PX = 44
WINDOW_BUTTONS_WIDTH_PX = 78  # room left at the bar's left end for the macOS window buttons
FOCAL_LENGTH_LIMIT_MM = 9999.9  # the box takes any focal length up to this, either sign
DIAMETER_LIMIT_MM = 9999.9
# The sliders' ends and power laws: 0 to 250 mm takes 40% of the focal length slider, and
# 0 to 50 mm takes 59% of the diameter slider (see raycrafter.gui.scaled_slider).
FOCAL_LENGTH_SLIDER_MAXIMUM_MM = 1000.0
FOCAL_LENGTH_SLIDER_EXPONENT = 1.5
DIAMETER_SLIDER_MAXIMUM_MM = 100.0
DIAMETER_SLIDER_EXPONENT = 1.3
USER_PART_BOX_WIDTH_PX = 200
LOCK_ICON_SIZE = QSize(36, 24)
NO_USER_PARTS = "No user parts saved yet"

# Qt 6.9 added Qt::ExpandedClientAreaHint, which lets the window's content extend under the
# title bar. PyQt6 has no name for it yet: it reuses the bit of the obsolete
# MaximizeUsingFullscreenGeometryHint, so that name is used for it here.
EXPANDED_CLIENT_AREA_HINT = Qt.WindowType.MaximizeUsingFullscreenGeometryHint


NS_WINDOW_TITLE_HIDDEN = 1  # NSWindowTitleVisibility.hidden


def hide_title_text(window: QMainWindow) -> None:
    """On macOS, stop the system drawing the window title (it would cover the bar's controls).

    The title itself is kept, for the Dock, the Window menu and Mission Control. Qt has no
    setting for this, so the native window's ``titleVisibility`` is set through the
    Objective-C runtime. Call once the native window exists (e.g. on the first show).
    """
    # Only on the native macOS platform: elsewhere (e.g. the off-screen platform the tests
    # use) winId() is not an NSView, and sending it a message would crash.
    if QGuiApplication.platformName() != "cocoa":
        return
    objc_library_path = ctypes.util.find_library("objc")
    if objc_library_path is None:
        return
    objc = ctypes.cdll.LoadLibrary(objc_library_path)
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    # objc_msgSend is called through prototypes of the exact argument types each message
    # takes: on Apple silicon a call with the wrong prototype passes its arguments wrongly.
    message_send_address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
    if message_send_address is None:
        return
    send_no_argument = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
        message_send_address
    )
    send_integer = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long)(
        message_send_address
    )
    native_view = int(window.winId())  # on macOS, the window's NSView
    native_window = send_no_argument(native_view, objc.sel_registerName(b"window"))
    if native_window:
        send_integer(
            native_window, objc.sel_registerName(b"setTitleVisibility:"), NS_WINDOW_TITLE_HIDDEN
        )


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
    paraxial_changed(float, float)
        The paraxial focal length or diameter was changed by the user (mm; f may be 0).
    user_part_activated(str)
        A user part was chosen from the drop-down (even the one already shown).
    """

    paraxial_changed = pyqtSignal(float, float)
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

        # The document's name (see set_document_title); on macOS the window title is hidden.
        self.document_title = QLabel()
        self.document_title.setObjectName("documentTitle")
        self.document_title.installEventFilter(self.dragger)  # it is part of the title bar
        self.addWidget(self.document_title)
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
        """A sign button, then the focal length and the diameter, each a slider and a box."""
        layout = self._new_page()
        self.sign_button = QToolButton()
        self.sign_button.setObjectName("signButton")
        self.sign_button.setText("±")
        self.sign_button.setToolTip("Change the sign of the focal length")
        layout.addWidget(self.sign_button)

        focal_length_label = QLabel(f"{ITALIC_F} =")
        focal_length_label.setObjectName("italicF")
        layout.addWidget(focal_length_label)
        self.focal_length = ScaledSlider(
            FOCAL_LENGTH_SLIDER_MAXIMUM_MM,
            FOCAL_LENGTH_SLIDER_EXPONENT,
            (-FOCAL_LENGTH_LIMIT_MM, FOCAL_LENGTH_LIMIT_MM),
            signed=True,
        )
        self.focal_length.setToolTip("Focal length of the paraxial (ideal thin) lens")
        layout.addWidget(self.focal_length)

        layout.addSpacing(10)
        diameter_label = QLabel("⌀ =")  # the diameter sign
        diameter_label.setObjectName("diameterSign")
        layout.addWidget(diameter_label)
        self.diameter = ScaledSlider(
            DIAMETER_SLIDER_MAXIMUM_MM, DIAMETER_SLIDER_EXPONENT, (0.1, DIAMETER_LIMIT_MM)
        )
        self.diameter.setToolTip("Clear diameter of the paraxial lens")
        layout.addWidget(self.diameter)

        self.sign_button.clicked.connect(self._change_focal_length_sign)
        self.focal_length.value_changed.connect(lambda _value: self._emit_paraxial_changed())
        self.diameter.value_changed.connect(lambda _value: self._emit_paraxial_changed())
        return layout

    def _change_focal_length_sign(self) -> None:
        self.focal_length.set_value(-self.focal_length.value())
        self._emit_paraxial_changed()

    def _emit_paraxial_changed(self) -> None:
        self.paraxial_changed.emit(*self.paraxial_values_mm())

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
        self.user_part_box.setFixedHeight(self.focal_length.box.sizeHint().height())
        layout.addWidget(self.user_part_box)
        return layout

    def show_page(self, tool: Tool) -> None:
        """Show the settings of a tool (an empty bar for a tool without any)."""
        layout = self.page_of_tool.get(tool, self.empty_page)
        self.pages.setCurrentWidget(layout.parentWidget())

    # ------------------------------------------------------------------ values

    def paraxial_values_mm(self) -> tuple[float, float]:
        """The paraxial focal length and diameter shown, in mm (a focal length of 0 is no lens)."""
        return self.focal_length.value(), self.diameter.value()

    def set_paraxial_values_mm(self, focal_length_mm: float, diameter_mm: float) -> None:
        """Show a focal length and diameter, without emitting ``paraxial_changed``."""
        self.focal_length.set_value(focal_length_mm)
        self.diameter.set_value(diameter_mm)

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

    def set_document_title(self, title: str, modified: bool) -> None:
        """Show the document's name, with " (edited)" while it has unsaved changes."""
        self.document_title.setText(f"{title} (edited)" if modified else title)

    def _show_lock_state(self, locked: bool) -> None:
        """A closed red padlock while focus is locked, an open grey one otherwise."""
        self.lock_focus_button.setIcon(focus_lock_icon(locked))
