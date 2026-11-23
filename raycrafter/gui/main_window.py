"""The RayCrafter main window: side view on top, controls and analysis plots below.

- Lenses are chosen in the drop-down and added with "Add lens"; each goes to the right of
  the existing lenses.
- Hover a lens and it turns a darker shade; that is the lens the actions apply to. Drag a
  lens to slide it along the axis; drag it past a neighbour's centre to swap them. Drag the
  green line to move the image plane (not while "Lock focus" is on).
- Right-click a lens for Flip (F), Delete (Del or Backspace) and Make / Split Plossl.

Updates come in two speeds. The side view, spot, through-focus plot and status bar are
recomputed on every change (coalesced, so a burst of drag events costs one recompute). The
MTF, ray fan and field curvature are slower, so they are recomputed only once changes have
stopped for SLOW_PLOT_DELAY_MS, and only for the tab that is showing.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pyqtgraph as pg
from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListView,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from raycrafter import palette
from raycrafter.core.analysis import FastTrace, airy_radius
from raycrafter.core.model import OpticalModel
from raycrafter.gui.analysis_panels import (
    FieldCurvaturePanel,
    MtfPanel,
    RayFanPanel,
    SpotPanel,
    ThroughFocusPanel,
    make_tabs,
)
from raycrafter.gui.side_view import IMAGE_MAX_Z, MIN_IMAGE_GAP_MM, SideView
from raycrafter.gui.style import style_sheet
from raycrafter.report import write_and_open_report

SLOW_PLOT_DELAY_MS = 120  # MTF, ray fan and field curvature wait this long after the last change
VERGENCE_LIMIT_DIOPTRES = 1.0  # the vergence slider spans +/- this (as for an f = 1000 mm lens)
DEFAULT_BEAM_DIAMETER_MM = 10  # what "Reset beam" restores
BEAM_DIAMETER_RANGE_MM = (1, 40)
BEAM_ANGLE_LIMIT_TENTHS = 100  # the angle slider works in tenths of a degree: +/-10 deg
CONTROL_PANEL_WIDTH_PX = 300
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
        self._build_widgets()
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
        self.populate_catalog_combo()
        self.side_view.rebuild_lens_graphics()
        self.recompute_and_redraw()

    # ------------------------------------------------------------------ setup

    def _build_widgets(self) -> None:
        """Create the side view, the analysis plots and the control panel, and lay them out."""
        self.side_view = SideView(self.model)
        self.spot_panel = SpotPanel()
        self.through_focus_panel = ThroughFocusPanel()
        self.mtf_panel = MtfPanel()
        self.ray_fan_panel = RayFanPanel()
        self.field_curvature_panel = FieldCurvaturePanel()

        # Three tab widgets of equal width. Field curvature gets a one-tab widget so that all
        # three plots line up.
        self.spot_tabs = make_tabs(
            [("Spot", self.spot_panel), ("Through-focus", self.through_focus_panel)]
        )
        self.mtf_tabs = make_tabs([("MTF", self.mtf_panel), ("Ray fan", self.ray_fan_panel)])
        self.field_tabs = make_tabs([("Field curvature", self.field_curvature_panel)])

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self._build_control_panel())
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
        """The framed panel of buttons, sliders and the lens drop-down."""
        state = self.model.state
        self.catalog_combo = QComboBox()
        # A plain QListView popup: the default one is drawn by the native macOS style, which
        # ignores the style sheet's hover and selection colours (white on white).
        self.catalog_combo.setView(QListView())
        self.add_button = QPushButton("Add lens")
        self.autofocus_button = QPushButton("Autofocus")
        self.reset_beam_button = QPushButton("Reset beam")
        self.save_zmx_button = QPushButton("Save ZMX")
        self.report_button = QPushButton("Generate report")
        self.lock_focus_checkbox = QCheckBox("Lock focus")  # off by default
        self.lock_focus_checkbox.setToolTip(
            "Keep the image plane at best focus as lenses or the beam change, "
            "whenever that focus is in view"
        )

        self.diameter_slider, self.diameter_label = self._make_slider(
            *BEAM_DIAMETER_RANGE_MM, value=int(state.beam_diameter)
        )
        self.angle_slider, self.angle_label = self._make_slider(
            -BEAM_ANGLE_LIMIT_TENTHS, BEAM_ANGLE_LIMIT_TENTHS, value=int(state.field_angle * 10)
        )
        vergence_limit_hundredths = round(VERGENCE_LIMIT_DIOPTRES * 100)
        self.vergence_slider, self.vergence_label = self._make_slider(
            -vergence_limit_hundredths,
            vergence_limit_hundredths,
            value=round(state.beam_vergence * 100),
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

        controls = QGridLayout()
        controls.setSpacing(8)
        controls.addWidget(self.catalog_combo, 0, 0, 1, 2)
        controls.addWidget(self.add_button, 1, 0)
        controls.addWidget(self.autofocus_button, 1, 1)
        controls.addWidget(self.diameter_label, 2, 0, 1, 2)
        controls.addWidget(self.diameter_slider, 3, 0, 1, 2)
        controls.addWidget(self.angle_label, 4, 0, 1, 2)
        controls.addWidget(self.angle_slider, 5, 0, 1, 2)
        controls.addWidget(self.vergence_label, 6, 0, 1, 2)
        controls.addWidget(self.vergence_slider, 7, 0, 1, 2)
        controls.addWidget(self.reset_beam_button, 8, 0)
        controls.addWidget(self.save_zmx_button, 8, 1)
        controls.addWidget(self.report_button, 9, 0, 1, 2)
        controls.addWidget(self.lock_focus_checkbox, 10, 0, 1, 2)
        controls.setRowStretch(11, 1)  # spare space goes above the hint, which sits at the foot
        controls.addWidget(hint_label, 12, 0, 1, 2, Qt.AlignmentFlag.AlignBottom)
        controls.setContentsMargins(14, 14, 14, 14)

        # A framed card (QFrame#controls in the style sheet), so the controls read as one panel.
        control_panel = QFrame()
        control_panel.setObjectName("controls")
        control_panel.setLayout(controls)
        control_panel.setFixedWidth(CONTROL_PANEL_WIDTH_PX)
        return control_panel

    @staticmethod
    def _make_slider(minimum: int, maximum: int, value: int) -> tuple[QSlider, QLabel]:
        """A horizontal slider and the label shown above it."""
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        label = QLabel()
        label.setObjectName("sliderLabel")
        return slider, label

    def _connect_signals(self) -> None:
        """Wire the buttons, sliders, side view and keyboard shortcuts to their handlers."""
        self.add_button.clicked.connect(self.add_selected_catalog_lens)
        self.autofocus_button.clicked.connect(self.run_autofocus)
        self.diameter_slider.valueChanged.connect(self.set_beam_diameter)
        self.angle_slider.valueChanged.connect(self.set_field_angle)
        self.vergence_slider.valueChanged.connect(self.set_beam_vergence)
        self.reset_beam_button.clicked.connect(self.reset_beam)
        self.save_zmx_button.clicked.connect(self.save_zmx)
        self.report_button.clicked.connect(self.generate_report)
        self.lock_focus_checkbox.toggled.connect(self.on_lock_focus_toggled)

        self.side_view.changed.connect(self.request_update)
        self.side_view.lens_menu_requested.connect(self.show_lens_menu)

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

        # Only the visible tab of each pair is computed, so switching tab must trigger a redraw.
        self.mtf_tabs.currentChanged.connect(lambda _index: self.update_mtf_tab())

    # --------------------------------------------------------------- controls

    def request_update(self) -> None:
        """Schedule one recompute; repeated requests before it runs collapse into one."""
        if not self.update_timer.isActive():
            self.update_timer.start(0)

    def populate_catalog_combo(self) -> None:
        """Fill the drop-down from the model's catalogue (the part number is the item data)."""
        self.catalog_combo.clear()
        for entry in self.model.catalog.values():
            self.catalog_combo.addItem(entry.label, entry.part_number)

    def add_selected_catalog_lens(self) -> None:
        """Place the lens chosen in the drop-down to the right of the existing lenses."""
        part_number = self.catalog_combo.currentData()
        if part_number is None:
            return
        self.model.add_lens(part_number)
        self.side_view.rebuild_lens_graphics()
        self.request_update()

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
        in_pair = self.model.state.lenses[lens_index].group_id is not None
        menu = QMenu(self)

        flip_action = menu.addAction("Flip\tF")
        flip_action.setEnabled(not in_pair)
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

    def set_beam_diameter(self, value_mm: int) -> None:
        """Slider callback: beam diameter in mm."""
        self.model.state.beam_diameter = float(value_mm)
        self.request_update()

    def set_field_angle(self, value_tenths: int) -> None:
        """Slider callback: beam angle in tenths of a degree."""
        self.model.state.field_angle = value_tenths / 10.0
        self.request_update()

    def set_beam_vergence(self, value_hundredths: int) -> None:
        """Slider callback: beam vergence in hundredths of a dioptre (positive = converging)."""
        self.model.state.beam_vergence = value_hundredths / 100.0
        self.request_update()

    def reset_beam(self) -> None:
        """Put the beam back to collimated, on axis, at the default diameter."""
        # Moving a slider fires its callback, which updates the model and schedules the redraw.
        self.vergence_slider.setValue(0)
        self.angle_slider.setValue(0)
        self.diameter_slider.setValue(DEFAULT_BEAM_DIAMETER_MM)

    def save_zmx(self) -> None:
        """Ask the user where to save, then export the current system as a Zemax .zmx file."""
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self, "Save Zemax file", "lens_system.zmx", "Zemax files (*.zmx)"
        )
        if not file_path:
            return  # dialog cancelled
        if not file_path.lower().endswith(".zmx"):
            file_path += ".zmx"
        try:
            self.model.save_zmx(file_path)
        except Exception as error:  # show any export problem instead of crashing the GUI
            QMessageBox.critical(self, "Save failed", f"Could not save {file_path}:\n{error}")
            return
        self.statusBar().showMessage(f"Saved {file_path}", 5000)

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
        self.diameter_label.setText(f"Beam diameter: {state.beam_diameter:.0f} mm")
        self.angle_label.setText(f"Beam angle: {state.field_angle:+.1f} deg")
        self.vergence_label.setText(
            f"Beam vergence: {state.beam_vergence:+.2f} D  ({self.vergence_description()})"
        )

        start = time.perf_counter()
        if self.lock_focus_checkbox.isChecked():
            self.refocus_if_focus_in_view()
        fast_trace = self.model.trace_fast()
        through_focus = self.model.compute_through_focus(fast_trace)
        trace_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        self.side_view.update_view(fast_trace.paths)
        self.spot_panel.show_result(fast_trace, state.field_angle)
        self.through_focus_panel.show_result(through_focus)
        draw_ms = (time.perf_counter() - start) * 1000

        self.update_readouts(fast_trace)
        self.side_view.set_timings(trace_ms, draw_ms)
        self.slow_timer.start(SLOW_PLOT_DELAY_MS)

    def update_readouts(self, fast_trace: FastTrace) -> None:
        """Status bar: image plane, RMS spot radius, first-order properties and Airy radius."""
        state = self.model.state
        readouts = [
            f"image z={state.image_z:.1f} mm",
            f"RMS={fast_trace.spot.rms_radius * 1000:.2f} um",
        ]

        summary = fast_trace.summary
        if summary is not None:
            readouts += [
                f"EFL={summary.effective_focal_length:.1f} mm",
                f"BFL={summary.back_focal_length:.1f} mm",
                f"f/{summary.f_number:.1f}",
            ]
        airy = airy_radius(fast_trace.spot.numerical_aperture)
        if airy is not None:
            readouts.append(f"Airy radius={airy * 1000:.1f} um")
        self.status_label.setText("   ".join(readouts))

    def vergence_description(self) -> str:
        """Plain-words description of the beam vergence: collimated, converging or diverging."""
        vergence = self.model.state.beam_vergence
        if vergence == 0.0:
            return "collimated"
        return "converging" if vergence > 0 else "diverging"

    def update_slow_plots(self) -> None:
        """Recompute the slower plots (run once changes have settled)."""
        self.update_mtf_tab()
        self.field_curvature_panel.show_result(
            self.model.compute_field_curvature(), self.model.state.field_angle
        )

    def update_mtf_tab(self) -> None:
        """Compute whichever of the MTF and ray fan is showing; the other waits until selected."""
        field_angle = self.model.state.field_angle
        if self.mtf_tabs.currentWidget() is self.mtf_panel:
            self.mtf_panel.show_result(self.model.compute_mtf(), field_angle)
        else:
            self.ray_fan_panel.show_result(self.model.compute_ray_fan(), field_angle)
