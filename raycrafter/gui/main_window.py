"""The RayCrafter main window: side view on top, controls and analysis plots below.

- "Add catalogue lens" opens the lens picker (``raycrafter.gui.lens_picker``). After Insert,
  a ghost of the chosen lens follows the pointer along the axis; a click places it, Escape
  cancels. "Add paraxial lens" does the same for an ideal thin lens of the focal length in
  the box beside it.
- Hover a lens and it turns a darker shade; that is the lens the actions apply to. Drag a
  lens to slide it along the axis; drag it past a neighbour's centre to swap them. Drag the
  green line to move the image plane (not while "Lock focus" is on).
- Right-click a lens for Flip (F), Delete (Del or Backspace) and Make / Split Plossl.
- The RayCrafter menu has About and Quit; the File menu has New, Load ZMX and Save ZMX.
  New, Load and Quit ask first if the design has changed since it was last saved or loaded.
- "Add field" opens the field window (``raycrafter.gui.field_editor``), where beams are
  added, removed and edited while the bench updates live.

Updates come in two speeds, to keep the window responsive (see CLAUDE.md). The side view,
spot, through-focus plot and status bar are recomputed on every change (coalesced, so a
burst of drag events costs one recompute). Everything else waits until nothing is being
dragged (a lens or the image plane):

- The MTF, ray fan and field curvature are faded while a drag goes on, then recomputed
  once the mouse is released (or SLOW_PLOT_DELAY_MS after a change that is not a drag),
  and only for the tab that is showing.
- The f-number needs an extra trace, so during a drag the status bar keeps showing the
  last value, and it is recomputed on release.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable

import pyqtgraph as pg
from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QAction, QCloseEvent, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from raycrafter import palette
from raycrafter.core.analysis import FastTrace, airy_radius
from raycrafter.core.model import OpticalModel
from raycrafter.core.plot_geometry import f_number_text, rms_radius_text
from raycrafter.core.state import OpticalState, PlacedLens
from raycrafter.core.zmx_import import ZmxImportError
from raycrafter.gui.about_dialog import AboutDialog
from raycrafter.gui.analysis_panels import (
    FieldCurvaturePanel,
    MtfPanel,
    RayFanPanel,
    SpotPanel,
    ThroughFocusPanel,
    make_tabs,
)
from raycrafter.gui.field_editor import FieldEditor
from raycrafter.gui.lens_picker import LensPickerDialog
from raycrafter.gui.side_view import IMAGE_MAX_Z, MIN_IMAGE_GAP_MM, SideView
from raycrafter.gui.style import style_sheet
from raycrafter.report import write_and_open_report

SLOW_PLOT_DELAY_MS = 120  # MTF, ray fan and field curvature wait this long after the last change
STALE_PLOT_OPACITY = 0.4  # the slower plots are this faded while a drag makes them out of date
DEFAULT_PARAXIAL_FOCAL_LENGTH_MM = 100
PARAXIAL_FOCAL_LENGTH_RANGE_MM = (-2000, 2000)
FOCAL_LENGTH_BOX_WIDTH_PX = 82  # room for "-2000 mm" and the arrows, leaving the rest to buttons
CONTROL_PANEL_WIDTH_PX = 385
HINT_MIN_HEIGHT_PX = 52  # height reserved for the help text at the foot of the control panel


class MainWindow(QMainWindow):
    """Side view, analysis plots and controls in one window.

    Parameters
    ----------
    model : OpticalModel, optional
        The bench to show. A new one (with one default lens) is made if not given.
    """

    def __init__(self, model: OpticalModel | None = None) -> None:
        super().__init__()
        self.setWindowTitle("RayCrafter")
        self.resize(1450, 900)
        self.setStyleSheet(style_sheet())
        pg.setConfigOptions(background="w", foreground=palette.TEXT, antialias=True)

        self.model = model if model is not None else OpticalModel()
        self.lens_picker: LensPickerDialog | None = None  # made on first use, then kept
        self.field_editor: FieldEditor | None = None  # made on first use, then kept
        self.f_number_readout: str | None = None  # last f-number text, kept during a drag
        self._build_widgets()
        self._build_menus()
        self._connect_signals()

        # Updates are coalesced: many drag events trigger one recompute per event-loop pass.
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.recompute_and_redraw)

        # The slower plots are redrawn only after changes have stopped for a moment.
        self.slow_timer = QTimer(self)
        self.slow_timer.setSingleShot(True)
        self.slow_timer.timeout.connect(self.update_slow_plots)

        self.model.autofocus()
        self.side_view.rebuild_lens_graphics()
        self.recompute_and_redraw()

        # The design as last saved or loaded (at first, the start-up bench): New, Load and
        # Quit ask before throwing away a design that differs from it.
        self.saved_state: OpticalState = copy.deepcopy(self.model.state)

    # ------------------------------------------------------------------ setup

    def _build_widgets(self) -> None:
        """Create the side view, the analysis plots and the control panel, and lay them out."""
        self.side_view = SideView(self.model)
        self.spot_panel = SpotPanel()
        self.through_focus_panel = ThroughFocusPanel()
        self.mtf_panel = MtfPanel()
        self.ray_fan_panel = RayFanPanel()
        self.field_curvature_panel = FieldCurvaturePanel()

        # The slower plots fade while a drag leaves them out of date. The effect is switched
        # off (not set to full opacity) when they are current, so it costs nothing then.
        self.slow_plot_fades: list[QGraphicsOpacityEffect] = []
        for panel in (self.mtf_panel, self.ray_fan_panel, self.field_curvature_panel):
            fade = QGraphicsOpacityEffect(panel)
            fade.setOpacity(STALE_PLOT_OPACITY)
            fade.setEnabled(False)
            panel.setGraphicsEffect(fade)
            self.slow_plot_fades.append(fade)

        # Three tab widgets of equal width. Field curvature gets a one-tab widget so that all
        # three plots line up.
        self.spot_tabs = make_tabs(
            [("Spot", self.spot_panel), ("Through-focus", self.through_focus_panel)]
        )
        self.mtf_tabs = make_tabs([("MTF", self.mtf_panel), ("Ray fan", self.ray_fan_panel)])
        self.field_tabs = make_tabs([("Field curvature", self.field_curvature_panel)])

        bottom_row = QHBoxLayout()
        self.control_panel = self._build_control_panel()
        bottom_row.addWidget(self.control_panel)
        for tab_widget in (self.spot_tabs, self.mtf_tabs, self.field_tabs):
            bottom_row.addWidget(tab_widget, stretch=1)

        central = QWidget()
        layout = QVBoxLayout(central)
        # Plot widgets report a large preferred height, which skewed the 2:3 split between the
        # side view and the bottom row; ignoring it lets the stretch factors alone decide.
        for plot_area in (self.side_view, self.spot_tabs, self.mtf_tabs, self.field_tabs):
            plot_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        layout.addWidget(self.side_view, stretch=2)
        layout.addLayout(bottom_row, stretch=3)
        self.setCentralWidget(central)

        self.status_label = QLabel()
        self.statusBar().addWidget(self.status_label)

    def _build_control_panel(self) -> QFrame:
        """The framed panel of buttons."""
        self.add_catalogue_lens_button = QPushButton("Add catalogue lens")
        self.add_paraxial_lens_button = QPushButton("Add paraxial lens")
        self.paraxial_focal_length_box = QSpinBox()
        self.paraxial_focal_length_box.setRange(*PARAXIAL_FOCAL_LENGTH_RANGE_MM)
        self.paraxial_focal_length_box.setValue(DEFAULT_PARAXIAL_FOCAL_LENGTH_MM)
        self.paraxial_focal_length_box.setSuffix(" mm")
        self.paraxial_focal_length_box.setMaximumWidth(FOCAL_LENGTH_BOX_WIDTH_PX)
        self.paraxial_focal_length_box.setToolTip(
            "Focal length of the next paraxial (ideal thin) lens; negative for a diverging lens"
        )
        self.autofocus_button = QPushButton("Autofocus")
        self.add_field_button = QPushButton("Add field")
        self.add_field_button.setToolTip("Open the field window to add, remove and edit beams")
        self.report_button = QPushButton("Generate report")
        self.lock_focus_checkbox = QCheckBox("Lock focus")  # off by default
        self.lock_focus_checkbox.setToolTip(
            "Keep the image plane at best focus as lenses or the beams change, "
            "whenever that focus is in view"
        )

        # Reminder of where the per-lens actions are, since they are not buttons.
        hint_label = QLabel(
            "Right-click a lens to flip it (F), delete it (Del) or make a Plossl pair. "
            "Drag a lens past its neighbour to swap their order."
        )
        hint_label.setObjectName("hintLabel")
        hint_label.setWordWrap(True)
        # A wrapped label reports a large height-for-width size hint, which swelled the bottom
        # row and squeezed the side view to a sliver. Ignoring its vertical hint keeps the
        # panel's size set by the controls; the label just uses the spare space at the bottom.
        hint_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored)
        hint_label.setMinimumHeight(HINT_MIN_HEIGHT_PX)
        hint_label.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)

        # Top row: the two ways of adding a lens; the paraxial one has its focal length beside it.
        add_lens_row = QHBoxLayout()
        add_lens_row.setSpacing(6)
        for button in (self.add_catalogue_lens_button, self.add_paraxial_lens_button):
            button.setStyleSheet("padding: 6px 6px;")  # narrower than usual, so the row fits
            add_lens_row.addWidget(button)
        add_lens_row.addWidget(self.paraxial_focal_length_box)

        controls = QGridLayout()
        controls.setSpacing(8)
        controls.addLayout(add_lens_row, 0, 0, 1, 2)
        controls.addWidget(self.autofocus_button, 1, 0)
        controls.addWidget(self.lock_focus_checkbox, 1, 1)
        controls.addWidget(self.add_field_button, 2, 0)
        controls.addWidget(self.report_button, 2, 1)
        controls.setRowStretch(3, 1)  # spare space goes above the hint, which sits at the foot
        controls.addWidget(hint_label, 4, 0, 1, 2, Qt.AlignmentFlag.AlignBottom)
        controls.setContentsMargins(14, 14, 14, 14)

        # A framed card (QFrame#controls in the style sheet), so the controls read as one panel.
        control_panel = QFrame()
        control_panel.setObjectName("controls")
        control_panel.setLayout(controls)
        control_panel.setFixedWidth(CONTROL_PANEL_WIDTH_PX)
        return control_panel

    def _build_menus(self) -> None:
        """The RayCrafter menu (About, Quit) and the File menu (New, Load ZMX, Save ZMX).

        On macOS, Qt moves About and Quit into the application menu, which is named after
        the app, so they appear where Mac users expect them; the menu roles tell it which.
        """
        menu_bar = self.menuBar()

        def add_action(menu, text: str, slot, shortcut=None, role=None) -> QAction:
            action = menu.addAction(text)
            action.triggered.connect(slot)
            if shortcut is not None:
                action.setShortcut(QKeySequence(shortcut))
            if role is not None:
                action.setMenuRole(role)
            return action

        app_menu = menu_bar.addMenu("RayCrafter")
        add_action(app_menu, "About", self.show_about, role=QAction.MenuRole.AboutRole)
        add_action(
            app_menu,
            "Quit RayCrafter",
            self.close,  # closeEvent asks first if there are unsaved changes
            shortcut=QKeySequence.StandardKey.Quit,
            role=QAction.MenuRole.QuitRole,
        )

        file_menu = menu_bar.addMenu("File")
        add_action(file_menu, "New", self.new_design, shortcut=QKeySequence.StandardKey.New)
        add_action(file_menu, "Load ZMX...", self.load_zmx, shortcut=QKeySequence.StandardKey.Open)
        add_action(file_menu, "Save ZMX...", self.save_zmx, shortcut=QKeySequence.StandardKey.Save)

    def _connect_signals(self) -> None:
        """Wire the buttons, side view and keyboard shortcuts to their handlers."""
        self.add_catalogue_lens_button.clicked.connect(self.open_lens_picker)
        self.add_paraxial_lens_button.clicked.connect(self.start_paraxial_lens_placement)
        # A focal length of zero is not a lens, so the button is off while the box shows 0.
        self.paraxial_focal_length_box.valueChanged.connect(
            lambda focal_length_mm: self.add_paraxial_lens_button.setEnabled(focal_length_mm != 0)
        )
        self.autofocus_button.clicked.connect(self.run_autofocus)
        self.add_field_button.clicked.connect(self.open_field_editor)
        self.report_button.clicked.connect(self.generate_report)
        self.lock_focus_checkbox.toggled.connect(self.on_lock_focus_toggled)

        self.side_view.changed.connect(self.request_update)
        self.side_view.lens_menu_requested.connect(self.show_lens_menu)
        self.side_view.placement_finished.connect(self.on_placement_finished)

        # Releasing a dragged item brings back the readouts that are skipped during a drag.
        self.side_view.drag_finished.connect(self.request_update)

        # F and Delete/Backspace act on the lens under the pointer.
        QShortcut(
            QKeySequence("F"), self, activated=lambda: self.act_on_hovered_lens(self.flip_lens)
        )
        for delete_key in ("Delete", "Backspace"):  # Backspace is "delete" on Mac keyboards
            QShortcut(
                QKeySequence(delete_key),
                self,
                activated=lambda: self.act_on_hovered_lens(self.delete_lens),
            )
        QShortcut(QKeySequence("Escape"), self, activated=self.side_view.cancel_placement)

        # Only the visible tab of each pair is computed, so switching tab must trigger a redraw.
        self.mtf_tabs.currentChanged.connect(lambda _index: self.update_mtf_tab())

    # --------------------------------------------------------------- controls

    def request_update(self) -> None:
        """Schedule one recompute; repeated requests before it runs collapse into one."""
        if not self.update_timer.isActive():
            self.update_timer.start(0)

    def open_lens_picker(self) -> None:
        """Show the lens picker; after Insert, let the user place the chosen lens on the axis."""
        if self.lens_picker is None:
            self.lens_picker = LensPickerDialog(self.model.catalog, parent=self)
        if self.lens_picker.exec() != LensPickerDialog.DialogCode.Accepted:
            return
        part_number = self.lens_picker.selected_part_number()
        if part_number is None:
            return
        self.start_lens_placement(self.model.make_lens(part_number))

    def start_paraxial_lens_placement(self) -> None:
        """Let the user place a paraxial lens with the focal length shown beside the button."""
        focal_length_mm = float(self.paraxial_focal_length_box.value())
        if focal_length_mm != 0.0:
            self.start_lens_placement(self.model.make_paraxial_lens(focal_length_mm))

    def start_lens_placement(self, new_lens: PlacedLens) -> None:
        """Make a new lens follow the pointer along the axis until a click places it."""
        self.side_view.start_placement(new_lens)
        self.statusBar().showMessage(
            f"Click on the axis to place {new_lens.part_number}. Esc or right-click to cancel."
        )

    def on_placement_finished(self, placed: bool) -> None:
        """Clear the placement instructions; the side view has already updated the model."""
        self.statusBar().clearMessage()
        if not placed:
            self.statusBar().showMessage("Lens not added", 3000)

    def act_on_hovered_lens(self, action: Callable[[int], None]) -> None:
        """Keyboard shortcuts: run ``action(index)`` for the lens under the pointer, if any."""
        if self.side_view.hovered_index is not None:
            action(self.side_view.hovered_index)

    def delete_lens(self, lens_index: int) -> None:
        """Remove one lens."""
        self.model.delete_lens(lens_index)
        self.side_view.rebuild_lens_graphics()
        self.request_update()

    def flip_lens(self, lens_index: int) -> None:
        """Turn one lens around (not available inside a Plossl pair)."""
        self.model.flip_lens(lens_index)
        self.request_update()

    def make_plossl(self, lens_index: int) -> None:
        """Snap a lens and its identical neighbour into a Plossl pair, then refocus."""
        if self.model.plossl_partner(lens_index) is not None:
            self.model.make_plossl(lens_index)
            self.model.autofocus()  # the pair has a new focal length, so the old focus is stale
            self.request_update()

    def split_plossl(self, lens_index: int) -> None:
        """Release a lens's Plossl pair."""
        self.model.split_plossl(lens_index)
        self.request_update()

    def show_lens_menu(self, lens_index: int, global_position: QPoint) -> None:
        """Pop up the actions for one lens at the pointer."""
        self.build_lens_menu(lens_index).exec(global_position)

    def build_lens_menu(self, lens_index: int) -> QMenu:
        """The context menu for one lens. Entries that do not apply are greyed out.

        A tab in an entry's text puts the keyboard shortcut in a right-hand column.
        """
        lens = self.model.state.lenses[lens_index]
        in_pair = lens.group_id is not None
        menu = QMenu(self)

        flip_action = menu.addAction("Flip\tF")
        flip_action.setEnabled(not in_pair and not lens.is_paraxial)
        flip_action.triggered.connect(lambda: self.flip_lens(lens_index))

        delete_action = menu.addAction("Delete\tDel")
        delete_action.triggered.connect(lambda: self.delete_lens(lens_index))

        menu.addSeparator()
        if in_pair:
            split_action = menu.addAction("Split Plossl")
            split_action.triggered.connect(lambda: self.split_plossl(lens_index))
        else:
            plossl_action = menu.addAction("Make Plossl")
            plossl_action.setEnabled(self.model.plossl_partner(lens_index) is not None)
            plossl_action.triggered.connect(lambda: self.make_plossl(lens_index))

            # While "Make Plossl" is highlighted, outline the two lenses that would be paired.
            def preview_if_plossl(hovered_action) -> None:
                if hovered_action is plossl_action and plossl_action.isEnabled():
                    self.side_view.show_plossl_preview(lens_index)
                else:
                    self.side_view.hide_plossl_preview()

            menu.hovered.connect(preview_if_plossl)
        menu.aboutToHide.connect(self.side_view.hide_plossl_preview)
        return menu

    def run_autofocus(self) -> None:
        """Move the image plane to the minimum-RMS-spot position."""
        self.model.autofocus()
        self.request_update()

    def open_field_editor(self) -> None:
        """Show the field window (made on first use) and bring it to the front.

        The first time, it is put over the control panel, so it covers neither the side
        view nor the plots the user is watching.
        """
        if self.field_editor is None:
            self.field_editor = FieldEditor(self.model, parent=self)
            self.field_editor.fields_changed.connect(self.request_update)
            self.field_editor.adjustSize()
            self.field_editor.move(self.control_panel.mapToGlobal(QPoint(0, 0)))
        self.field_editor.refresh()  # the bench may have changed since it was last open
        self.field_editor.show()
        self.field_editor.raise_()
        self.field_editor.activateWindow()

    # ------------------------------------------------------------------ files

    def has_unsaved_changes(self) -> bool:
        """True if the design differs from the one last saved or loaded."""
        return self.model.state != self.saved_state

    def mark_saved(self) -> None:
        """Remember the current design as saved (after Save, Load or New)."""
        self.saved_state = copy.deepcopy(self.model.state)

    def confirm_discard_changes(self, action_text: str) -> bool:
        """If there are unsaved changes, ask whether to save them first.

        Parameters
        ----------
        action_text : str
            What is about to happen, e.g. "starting a new design".

        Returns
        -------
        bool
            True to go ahead (nothing to lose, saved, or discarded); False to stop.
        """
        if not self.has_unsaved_changes():
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            f"The design has changed since it was last saved. Save it before {action_text}?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save_zmx()
        return answer == QMessageBox.StandardButton.Discard

    def show_new_design(self) -> None:
        """Redraw everything after the model's state was replaced (New or Load)."""
        self.side_view.cancel_placement()
        self.side_view.rebuild_lens_graphics()
        if self.field_editor is not None:
            self.field_editor.refresh()
        self.f_number_readout = None
        self.request_update()

    def new_design(self) -> None:
        """File > New: an empty bench with one default field."""
        if not self.confirm_discard_changes("starting a new design"):
            return
        self.model.new_design()
        self.mark_saved()
        self.show_new_design()

    def load_zmx(self) -> None:
        """File > Load ZMX: replace the design with the system in a Zemax file."""
        if not self.confirm_discard_changes("loading another design"):
            return
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self, "Load Zemax file", "", "Zemax files (*.zmx)"
        )
        if not file_path:
            return  # dialog cancelled
        try:
            notes = self.model.load_zmx(file_path)
        except ZmxImportError as error:
            QMessageBox.critical(self, "Load failed", f"Could not load {file_path}:\n{error}")
            return
        self.mark_saved()
        self.show_new_design()
        if notes:
            QMessageBox.information(
                self, "Loaded with changes", "\n\n".join(["Some details were simplified:"] + notes)
            )
        self.statusBar().showMessage(f"Loaded {file_path}", 5000)

    def save_zmx(self) -> bool:
        """File > Save ZMX: ask where to save, then export the system as a Zemax .zmx file.

        Returns
        -------
        bool
            True if the file was written.
        """
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self, "Save Zemax file", "lens_system.zmx", "Zemax files (*.zmx)"
        )
        if not file_path:
            return False  # dialog cancelled
        if not file_path.lower().endswith(".zmx"):
            file_path += ".zmx"
        try:
            self.model.save_zmx(file_path)
        except Exception as error:  # show any export problem instead of crashing the GUI
            QMessageBox.critical(self, "Save failed", f"Could not save {file_path}:\n{error}")
            return False
        self.mark_saved()
        self.statusBar().showMessage(f"Saved {file_path}", 5000)
        return True

    def show_about(self) -> None:
        """RayCrafter > About: the icon, name, version and project link."""
        AboutDialog(self).exec()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Closing the window (or Quit) asks first if there are unsaved changes."""
        if self.confirm_discard_changes("quitting"):
            event.accept()
        else:
            event.ignore()

    def generate_report(self) -> None:
        """Write an HTML report of the current system and open it in the web browser."""
        if not self.model.state.lenses:
            self.statusBar().showMessage("Add a lens before generating a report", 5000)
            return
        try:
            report_path = write_and_open_report(self.model)
        except Exception as error:  # show any problem instead of crashing the GUI
            QMessageBox.critical(self, "Report failed", f"Could not generate the report:\n{error}")
            return
        self.statusBar().showMessage(f"Report written to {report_path}", 8000)

    def on_lock_focus_toggled(self, locked: bool) -> None:
        """Lock focus on: the image plane follows best focus and cannot be dragged."""
        self.side_view.set_image_line_locked(locked)
        self.request_update()

    def refocus_if_focus_in_view(self) -> None:
        """With "Lock focus" on, move the image plane to best focus if that focus is visible.

        If the best focus falls beyond the right edge of the side view, or the search can
        only end against the back of the last lens (no real focus behind it, e.g. a negative
        lens), the image plane stays where it was rather than jumping somewhere unhelpful.
        """
        state = self.model.state
        if not state.lenses:
            return

        previous_image_z = state.image_z
        focus_z = self.model.autofocus()
        focus_is_in_view = state.last_back_z() + MIN_IMAGE_GAP_MM <= focus_z <= IMAGE_MAX_Z
        if not focus_is_in_view:
            state.image_z = previous_image_z

    # ------------------------------------------------------------------ redraw

    def recompute_and_redraw(self) -> None:
        """Re-trace the system; refresh the side view, spot, through-focus and readouts."""
        state = self.model.state
        start = time.perf_counter()
        if self.lock_focus_checkbox.isChecked():
            self.refocus_if_focus_in_view()
        fast_trace = self.model.trace_fast()
        through_focus = self.model.compute_through_focus(fast_trace)
        trace_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        self.side_view.update_view(fast_trace.paths)
        self.spot_panel.show_result(fast_trace, state.fields)
        self.through_focus_panel.show_result(through_focus)
        draw_ms = (time.perf_counter() - start) * 1000

        self.update_readouts(fast_trace)
        self.side_view.set_timings(trace_ms, draw_ms)

        if self.is_dragging():
            # The slower plots wait for the mouse to be released; faded, they read as stale.
            self.slow_timer.stop()
            self.set_slow_plots_faded(True)
        else:
            self.slow_timer.start(SLOW_PLOT_DELAY_MS)

    def update_readouts(self, fast_trace: FastTrace) -> None:
        """Status bar: image plane, RMS spot radius, first-order properties and Airy radius."""
        state = self.model.state
        readouts = [
            f"image z={state.image_z:.1f} mm",
            f"RMS={rms_radius_text(fast_trace.spot)}",
        ]

        summary = fast_trace.summary
        if summary is not None:
            readouts += [
                f"EFL={summary.effective_focal_length:.1f} mm",
                f"BFL={summary.back_focal_length:.1f} mm",
            ]
        # The f-number needs an extra trace: during a drag the last value stays on show.
        if not self.is_dragging():
            f_number = self.model.compute_f_number()
            self.f_number_readout = None if f_number is None else f_number_text(f_number)
        if self.f_number_readout is not None:
            readouts.append(self.f_number_readout)
        airy = airy_radius(fast_trace.spot)
        if airy is not None:
            readouts.append(f"Airy radius={airy * 1000:.1f} um")
        self.status_label.setText("   ".join(readouts))

    def is_dragging(self) -> bool:
        """True while the mouse holds a lens or the image plane."""
        return self.side_view.is_dragging()

    def set_slow_plots_faded(self, faded: bool) -> None:
        """Fade the MTF, ray fan and field curvature (out of date), or show them normally."""
        for fade in self.slow_plot_fades:
            fade.setEnabled(faded)

    def update_slow_plots(self) -> None:
        """Recompute the slower plots (run once changes have settled) and un-fade them."""
        if self.is_dragging():
            return  # they are recomputed when the drag ends
        self.update_mtf_tab()
        self.field_curvature_panel.show_result(
            self.model.compute_field_curvature(), self.model.state.selected_field.angle_deg
        )
        self.set_slow_plots_faded(False)

    def update_mtf_tab(self) -> None:
        """Compute whichever of the MTF and ray fan is showing; the other waits until selected."""
        field_angle = self.model.state.selected_field.angle_deg
        if self.mtf_tabs.currentWidget() is self.mtf_panel:
            self.mtf_panel.show_result(self.model.compute_mtf(), field_angle)
        else:
            self.ray_fan_panel.show_result(self.model.compute_ray_fan(), field_angle)
