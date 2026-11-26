"""The analysis plots below the side view: spot, through-focus, MTF, ray fan, field curvature
and distortion.

Each panel has a ``show_result`` method that draws one result object from
``raycrafter.core.analysis`` (or ``raycrafter.core.distortion``). Most panels are a single
pyqtgraph plot widget. The panels do no optics themselves.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QRadioButton,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from raycrafter import palette
from raycrafter.core.analysis import (
    FIELD_CURVATURE_MAX_ANGLE,
    FastTrace,
    FieldCurvatureResult,
    MtfResult,
    RayFanResult,
    ThroughFocusResult,
    airy_radius,
)
from raycrafter.core.distortion import DistortionResult
from raycrafter.core.plot_geometry import (
    distortion_limit_percent,
    distortion_title,
    field_name,
    grid_polylines,
    largest_magnitude,
    ray_fan_limit_um,
    rms_radius_text,
    spot_view_half_width_um,
)
from raycrafter.core.state import AnyField
from raycrafter.gui.style import PLOT_MIN_SIZE_PX, style_plot

TITLE_SIZE = "11pt"


def make_tabs(tabs: list[tuple[str, QWidget]]) -> QTabWidget:
    """A tab widget from (label, plot widget) pairs, with the minimum plot size set."""
    tab_widget = QTabWidget()
    tab_bar = tab_widget.tabBar()
    tab_bar.setExpanding(False)
    tab_bar.setDrawBase(False)

    # Tab labels must never be shortened with "...": switch eliding off and keep scroll arrows
    # away. (Their font size is set on QTabBar itself in the style sheet, not on QTabBar::tab,
    # so the width Qt reserves for a tab is computed from the same font that is painted.)
    tab_bar.setElideMode(Qt.TextElideMode.ElideNone)
    tab_bar.setUsesScrollButtons(False)
    for label, plot_widget in tabs:
        plot_widget.setMinimumSize(*PLOT_MIN_SIZE_PX)
        tab_widget.addTab(plot_widget, label)
    return tab_widget


class SpotPanel(pg.PlotWidget):
    """Spot diagram of every field, each centred on its own centroid and drawn in the field's
    colour, with the selected field's Airy disk dashed.

    The selected field's spot is drawn on top, so it is never hidden by the others.
    """

    def __init__(self) -> None:
        super().__init__()
        style_plot(self, "x (um)", "y (um)")
        plot = self.getPlotItem()
        plot.setAspectLocked(True)
        plot.addLine(x=0, pen=pg.mkPen("#dee2e6"))
        plot.addLine(y=0, pen=pg.mkPen("#dee2e6"))
        self.airy_curve = pg.PlotCurveItem(
            pen=pg.mkPen(palette.AIRY, width=1.2, style=Qt.PenStyle.DashLine)
        )
        plot.addItem(self.airy_curve)
        self.field_scatters: list[pg.ScatterPlotItem] = []  # one per field

    def show_result(self, fast_trace: FastTrace, fields: list[AnyField]) -> None:
        """Draw every field's spot (micrometres about its centroid), the Airy disk and RMS.

        Parameters
        ----------
        fast_trace : FastTrace
            The spots, one per field.
        fields : list of BeamField or ObjectPoint
            The fields, for their colours.
        """
        self._match_scatter_count(len(fast_trace.spots))
        selected_index = fast_trace.selected_field_index
        for index, (spot, beam, scatter) in enumerate(
            zip(fast_trace.spots, fields, self.field_scatters)
        ):
            scatter.setData(
                spot.x * 1000.0, spot.y * 1000.0, brush=pg.mkBrush(beam.display_colour())
            )
            scatter.setZValue(10 if index == selected_index else 5)

        spot = fast_trace.spot
        airy = airy_radius(spot)
        if airy is None:
            self.airy_curve.setData([], [])
        else:
            airy_um = airy * 1000.0
            angles = np.linspace(0.0, 2.0 * np.pi, 120)
            self.airy_curve.setData(airy_um * np.cos(angles), airy_um * np.sin(angles))

        half_width = spot_view_half_width_um(fast_trace.spots, spot)
        plot = self.getPlotItem()
        plot.setRange(xRange=(-half_width, half_width), yRange=(-half_width, half_width), padding=0)

        # With several fields, the title names the selected one, in its colour.
        if not spot.has_light:
            title = "No light reaches the image plane"
        else:
            title = f"RMS radius {rms_radius_text(spot)}"
        if len(fields) > 1:
            colour = fields[selected_index].display_colour()
            title = f'<span style="color:{colour}">{field_name(selected_index)}: {title}</span>'
        plot.setTitle(title, size="10pt")

    def _match_scatter_count(self, field_count: int) -> None:
        """Add or remove scatter items so there is one per field."""
        plot = self.getPlotItem()
        while len(self.field_scatters) < field_count:
            scatter = pg.ScatterPlotItem(size=5, pen=None)
            plot.addItem(scatter)
            self.field_scatters.append(scatter)
        while len(self.field_scatters) > field_count:
            plot.removeItem(self.field_scatters.pop())


class ThroughFocusPanel(pg.PlotWidget):
    """RMS spot radius of the selected field against image plane position.

    Green is the current image plane, the dotted blue line the best focus and the dashed
    grey line the Airy radius.
    """

    def __init__(self) -> None:
        super().__init__()
        style_plot(self, "image plane offset (mm)", "RMS radius (um)")
        plot = self.getPlotItem()
        plot.setTitle("Through-focus RMS radius", size=TITLE_SIZE)
        plot.addLine(x=0, pen=pg.mkPen(palette.IMAGE, width=2))
        self.best_focus_line = plot.addLine(
            x=0, pen=pg.mkPen(palette.TANGENTIAL, width=1, style=Qt.PenStyle.DotLine)
        )
        self.airy_line = plot.addLine(
            y=0,
            pen=pg.mkPen(palette.AIRY, width=1.2, style=Qt.PenStyle.DashLine),
            label="Airy radius",
            labelOpts={"position": 0.12, "color": palette.AIRY},
        )
        self.current_curve = plot.plot(pen=pg.mkPen(palette.TANGENTIAL, width=2))

    def show_result(self, result: ThroughFocusResult) -> None:
        """Draw RMS radius against offset, with the best focus and the Airy radius marked."""
        rms_current_um = result.rms_current * 1000.0
        self.current_curve.setData(result.offsets, rms_current_um, connect="finite")
        curve_maximum_um = largest_magnitude(rms_current_um) or 1.0  # no light: any range

        self.best_focus_line.setValue(result.best_offset)

        # Show the Airy radius line unless it is far above the curve.
        y_maximum_um = curve_maximum_um
        self.airy_line.setVisible(result.airy_radius is not None)
        if result.airy_radius is not None:
            airy_um = result.airy_radius * 1000.0
            self.airy_line.setValue(airy_um)
            y_maximum_um = max(curve_maximum_um, min(airy_um, 3.0 * curve_maximum_um))

        plot = self.getPlotItem()
        plot.setXRange(result.offsets[0], result.offsets[-1], padding=0)
        plot.setYRange(0, y_maximum_um * 1.08, padding=0)


class MtfPanel(pg.PlotWidget):
    """Tangential and sagittal MTF, with the diffraction limit."""

    def __init__(self) -> None:
        super().__init__()
        style_plot(self, "spatial frequency (cycles/mm)", "modulation")
        plot = self.getPlotItem()
        plot.setTitle("MTF", size=TITLE_SIZE)
        plot.setYRange(0, 1.02, padding=0)
        plot.addLegend(offset=(-5, 5))
        self.limit_curve = plot.plot(
            pen=pg.mkPen(palette.AIRY, width=1.2, style=Qt.PenStyle.DashLine),
            name="diffraction limit",
        )
        self.tangential_curve = plot.plot(
            pen=pg.mkPen(palette.TANGENTIAL, width=2), name="tangential"
        )
        self.sagittal_curve = plot.plot(pen=pg.mkPen(palette.SAGITTAL, width=2), name="sagittal")

    def show_result(self, mtf: MtfResult, field_position: str) -> None:
        """Draw the MTF curves (or say why there are none).

        ``field_position`` says where the field is, e.g. "+2.5 deg" or "y = +0.50 mm" (see
        ``plot_geometry.field_position_text``).
        """
        self.limit_curve.setData(mtf.frequency, mtf.diffraction_limit)
        self.tangential_curve.setData(mtf.frequency, mtf.tangential)
        self.sagittal_curve.setData(mtf.frequency, mtf.sagittal)
        plot = self.getPlotItem()
        if mtf.frequency.size == 0:
            plot.setTitle(f"No MTF: light blocked at {field_position}", size=TITLE_SIZE)
            return
        plot.setTitle(f"MTF at {field_position}", size=TITLE_SIZE)
        plot.setXRange(0, mtf.frequency[-1], padding=0)


class RayFanPanel(pg.PlotWidget):
    """Image-plane error against pupil coordinate for the tangential and sagittal fans."""

    def __init__(self) -> None:
        super().__init__()
        style_plot(self, "pupil coordinate", "image plane error (um)")
        plot = self.getPlotItem()
        plot.setTitle("Ray fan", size=TITLE_SIZE)
        plot.setXRange(-1, 1, padding=0.02)
        plot.addLegend(offset=(-5, 5))
        plot.addLine(y=0, pen=pg.mkPen(palette.FAINT))
        self.tangential_curve = plot.plot(
            pen=pg.mkPen(palette.TANGENTIAL, width=2), name="tangential"
        )
        self.sagittal_curve = plot.plot(
            pen=pg.mkPen(palette.SAGITTAL, width=2, style=Qt.PenStyle.DashLine), name="sagittal"
        )

    def show_result(self, fan: RayFanResult, field_position: str) -> None:
        """Draw the ray fan (micrometres) of the field at ``field_position`` (as for the MTF)."""
        # Blocked rays are NaN; connect="finite" leaves gaps there instead of failing.
        self.tangential_curve.setData(fan.pupil, fan.tangential_error * 1000.0, connect="finite")
        self.sagittal_curve.setData(fan.pupil, fan.sagittal_error * 1000.0, connect="finite")
        plot = self.getPlotItem()
        plot.setTitle(f"Ray fan at {field_position}", size=TITLE_SIZE)

        largest_error_mm = largest_magnitude(fan.tangential_error, fan.sagittal_error)
        limit_um = ray_fan_limit_um(largest_error_mm)
        plot.setYRange(-limit_um, limit_um, padding=0)


class FieldCurvaturePanel(pg.PlotWidget):
    """Best-focus shift against field angle, relative to the image plane."""

    def __init__(self) -> None:
        super().__init__()
        style_plot(self, "focus shift from image plane (mm)", "field angle (deg)")
        plot = self.getPlotItem()
        plot.setTitle("Best focus against field angle", size=TITLE_SIZE)
        plot.setYRange(0, FIELD_CURVATURE_MAX_ANGLE, padding=0.02)
        plot.addLegend(offset=(-5, 5))
        plot.addLine(x=0, pen=pg.mkPen(palette.IMAGE, width=2))
        self.current_angle_line = plot.addLine(
            y=0, pen=pg.mkPen(palette.FAINT, style=Qt.PenStyle.DashLine)
        )
        self.tangential_curve = plot.plot(
            pen=pg.mkPen(palette.TANGENTIAL, width=2),
            symbol="o",
            symbolSize=5,
            symbolBrush=palette.TANGENTIAL,
            symbolPen=None,
            name="tangential",
        )
        self.sagittal_curve = plot.plot(
            pen=pg.mkPen(palette.SAGITTAL, width=2, style=Qt.PenStyle.DashLine),
            symbol="s",
            symbolSize=5,
            symbolBrush=palette.SAGITTAL,
            symbolPen=None,
            name="sagittal",
        )

    def show_result(self, curvature: FieldCurvatureResult, field_angle: float) -> None:
        """Draw the curves; a dashed line marks the selected field's angle.

        ``field_angle`` is the angle of the field's chief ray in degrees (for an object point,
        see ``fields.chief_ray_angle_deg``).
        """
        # Angles with too little light are NaN; connect="finite" leaves gaps there.
        self.tangential_curve.setData(
            curvature.tangential_shift, curvature.angles, connect="finite"
        )
        self.sagittal_curve.setData(curvature.sagittal_shift, curvature.angles, connect="finite")
        self.current_angle_line.setValue(abs(field_angle))

        largest_shift = largest_magnitude(
            curvature.tangential_shift, curvature.sagittal_shift, floor=0.5
        )
        self.getPlotItem().setXRange(-largest_shift * 1.1, largest_shift * 1.1, padding=0)


def _show_plain_numbers(plot_widget: pg.PlotWidget) -> None:
    """Stop pyqtgraph rescaling an axis with small values and adding "(x0.001)" to its label."""
    for axis_name in ("bottom", "left"):
        plot_widget.getPlotItem().getAxis(axis_name).enableAutoSIPrefix(False)


class DistortionPanel(QWidget):
    """Grid distortion of a finite object, or the distortion against object height.

    The grid shows the undistorted image of a square grid (grey, dashed) and where its points
    really land (blue). Two buttons above the plot choose which of the two plots shows, and a
    checkbox locates the image points by spot centroid rather than chief ray.
    ``reference_changed`` is emitted when that box is clicked: the result must then be
    computed again.
    """

    reference_changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.grid_plot = self._build_grid_plot()
        self.height_plot = self._build_height_plot()
        self.plots = (self.grid_plot, self.height_plot)
        self.stack = QStackedWidget()
        for plot_widget in self.plots:
            self.stack.addWidget(plot_widget)

        self.grid_button = QRadioButton("Grid")
        self.height_button = QRadioButton("Against height")
        self.grid_button.setChecked(True)
        view_buttons = QButtonGroup(self)
        view_buttons.addButton(self.grid_button)
        view_buttons.addButton(self.height_button)
        self.grid_button.toggled.connect(
            lambda grid_chosen: self.stack.setCurrentIndex(0 if grid_chosen else 1)
        )
        self.centroid_checkbox = QCheckBox("Spot centroid")
        self.centroid_checkbox.setToolTip(
            "Locate each image point by the centroid of its spot instead of its chief ray"
        )
        self.centroid_checkbox.toggled.connect(lambda _checked: self.reference_changed.emit())

        controls = QHBoxLayout()
        controls.addWidget(self.grid_button)
        controls.addWidget(self.height_button)
        controls.addStretch()
        controls.addWidget(self.centroid_checkbox)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(controls)
        layout.addWidget(self.stack, stretch=1)

    @property
    def reference(self) -> str:
        """How the image points are to be located: ``"chief_ray"`` or ``"centroid"``."""
        return "centroid" if self.centroid_checkbox.isChecked() else "chief_ray"

    def _build_grid_plot(self) -> pg.PlotWidget:
        """The plot of the ideal and real grids, with its two curves."""
        plot_widget = pg.PlotWidget()
        style_plot(plot_widget, "image x (mm)", "image y (mm)")
        _show_plain_numbers(plot_widget)
        plot = plot_widget.getPlotItem()
        plot.setAspectLocked(True)
        self.ideal_curve = plot.plot(
            pen=pg.mkPen(palette.AIRY, width=1, style=Qt.PenStyle.DashLine)
        )
        self.image_curve = plot.plot(pen=pg.mkPen(palette.TANGENTIAL, width=1.5))
        self.image_points = pg.ScatterPlotItem(size=4, pen=None, brush=palette.TANGENTIAL)
        plot.addItem(self.image_points)
        return plot_widget

    def _build_height_plot(self) -> pg.PlotWidget:
        """The plot of distortion (x) against object height (y)."""
        plot_widget = pg.PlotWidget()
        style_plot(plot_widget, "distortion (%)", "object height (mm)")
        _show_plain_numbers(plot_widget)
        plot = plot_widget.getPlotItem()
        plot.setTitle("Distortion against object height", size=TITLE_SIZE)
        plot.addLine(x=0, pen=pg.mkPen(palette.FAINT))
        self.selected_height_line = plot.addLine(
            y=0, pen=pg.mkPen(palette.FAINT, style=Qt.PenStyle.DashLine)
        )
        self.height_curve = plot.plot(
            pen=pg.mkPen(palette.TANGENTIAL, width=2),
            symbol="o",
            symbolSize=4,
            symbolBrush=palette.TANGENTIAL,
            symbolPen=None,
        )
        return plot_widget

    def show_result(self, result: DistortionResult, selected_height_mm: float) -> None:
        """Draw both plots; a dashed line marks the selected field's height (mm).

        With nothing to show, both are empty and the grid plot's title says why.
        """
        ideal_x, ideal_y = grid_polylines(result.ideal_x, result.ideal_y)
        self.ideal_curve.setData(ideal_x, ideal_y, connect="finite")
        # Blocked points are NaN; connect="finite" leaves gaps there.
        image_x, image_y = grid_polylines(result.image_x, result.image_y)
        self.image_curve.setData(image_x, image_y, connect="finite")
        reached = np.isfinite(result.image_x) & np.isfinite(result.image_y)
        self.image_points.setData(result.image_x[reached], result.image_y[reached])

        grid_plot = self.grid_plot.getPlotItem()
        grid_plot.setTitle(distortion_title(result), size=TITLE_SIZE)
        half_width = largest_magnitude(ideal_x, ideal_y, image_x, image_y) or 1.0
        half_width *= 1.08
        grid_plot.setRange(
            xRange=(-half_width, half_width), yRange=(-half_width, half_width), padding=0
        )

        self.height_curve.setData(
            result.height_distortion_percent, result.heights, connect="finite"
        )
        self.selected_height_line.setVisible(result.has_grid)
        self.selected_height_line.setValue(selected_height_mm)
        height_plot = self.height_plot.getPlotItem()
        limit = distortion_limit_percent(result)
        height_plot.setXRange(-limit, limit, padding=0)
        top_mm = result.heights[-1] if result.has_grid else 1.0
        height_plot.setYRange(0, top_mm, padding=0.02)
