"""The analysis plots below the side view: spot, through-focus, MTF, ray fan, field curvature.

Each panel is a pyqtgraph plot widget with a ``show_result`` method that draws one result
object from ``raycrafter.core.analysis``. The panels do no optics themselves.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTabWidget, QWidget

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
from raycrafter.core.plot_geometry import (
    largest_magnitude,
    ray_fan_limit_um,
    rms_radius_text,
    spot_view_half_width_um,
)
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
    """Spot diagram, each field centred on its own centroid, with the Airy disk dashed.

    The current-angle spot is blue. When the beam is tilted the 0 degree spot is drawn too
    (orange, underneath) so the off-axis degradation can be compared at a glance.
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
        self.on_axis_scatter = pg.ScatterPlotItem(
            size=5, pen=None, brush=pg.mkBrush(palette.ON_AXIS)
        )
        self.current_scatter = pg.ScatterPlotItem(
            size=5, pen=None, brush=pg.mkBrush(palette.TANGENTIAL)
        )
        for item in (self.airy_curve, self.on_axis_scatter, self.current_scatter):
            plot.addItem(item)

    def show_result(self, fast_trace: FastTrace, field_angle: float) -> None:
        """Draw the spots (micrometres about each centroid), the Airy disk and the RMS radii."""
        spot, on_axis_spot = fast_trace.spot, fast_trace.on_axis_spot
        self.current_scatter.setData(spot.x * 1000.0, spot.y * 1000.0)
        if on_axis_spot is not None:
            self.on_axis_scatter.setData(on_axis_spot.x * 1000.0, on_axis_spot.y * 1000.0)
        else:
            self.on_axis_scatter.setData([], [])

        airy = airy_radius(spot.numerical_aperture)
        if airy is None:
            self.airy_curve.setData([], [])
        else:
            airy_um = airy * 1000.0
            angles = np.linspace(0.0, 2.0 * np.pi, 120)
            self.airy_curve.setData(airy_um * np.cos(angles), airy_um * np.sin(angles))

        half_width = spot_view_half_width_um(spot, on_axis_spot)
        plot = self.getPlotItem()
        plot.setRange(xRange=(-half_width, half_width), yRange=(-half_width, half_width), padding=0)

        # The title doubles as the legend: each RMS radius is written in its spot colour.
        if on_axis_spot is None:
            title = f"RMS radius {rms_radius_text(spot)}"
            if not spot.has_light:
                title = "No light reaches the image plane"
        else:
            title = (
                f'<span style="color:{palette.TANGENTIAL}">{field_angle:+.1f} deg: '
                f"{rms_radius_text(spot)}</span>&nbsp;&nbsp;"
                f'<span style="color:{palette.ON_AXIS}">0 deg: '
                f"{rms_radius_text(on_axis_spot)}</span>"
            )
        plot.setTitle(title, size="10pt")


class ThroughFocusPanel(pg.PlotWidget):
    """RMS spot radius against image plane position.

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
        self.on_axis_curve = plot.plot(pen=pg.mkPen(palette.ON_AXIS, width=2))
        self.current_curve = plot.plot(pen=pg.mkPen(palette.TANGENTIAL, width=2))

    def show_result(self, result: ThroughFocusResult) -> None:
        """Draw RMS radius against offset, with the best focus and the Airy radius marked."""
        rms_current_um = result.rms_current * 1000.0
        self.current_curve.setData(result.offsets, rms_current_um, connect="finite")
        curve_maximum_um = largest_magnitude(rms_current_um)

        if result.rms_on_axis is not None:
            rms_on_axis_um = result.rms_on_axis * 1000.0
            self.on_axis_curve.setData(result.offsets, rms_on_axis_um, connect="finite")
            curve_maximum_um = largest_magnitude(rms_current_um, rms_on_axis_um)
        else:
            self.on_axis_curve.setData([], [])
        curve_maximum_um = curve_maximum_um or 1.0  # no light: any range will do

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

    def show_result(self, mtf: MtfResult, field_angle: float) -> None:
        """Draw the MTF curves computed at ``field_angle`` degrees (or say why there are none)."""
        self.limit_curve.setData(mtf.frequency, mtf.diffraction_limit)
        self.tangential_curve.setData(mtf.frequency, mtf.tangential)
        self.sagittal_curve.setData(mtf.frequency, mtf.sagittal)
        plot = self.getPlotItem()
        if mtf.frequency.size == 0:
            plot.setTitle(f"No MTF: beam blocked at {field_angle:+.1f} deg", size=TITLE_SIZE)
            return
        plot.setTitle(f"MTF at {field_angle:+.1f} deg", size=TITLE_SIZE)
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

    def show_result(self, fan: RayFanResult, field_angle: float) -> None:
        """Draw the ray fan (micrometres) computed at ``field_angle`` degrees."""
        # Blocked rays are NaN; connect="finite" leaves gaps there instead of failing.
        self.tangential_curve.setData(fan.pupil, fan.tangential_error * 1000.0, connect="finite")
        self.sagittal_curve.setData(fan.pupil, fan.sagittal_error * 1000.0, connect="finite")
        plot = self.getPlotItem()
        plot.setTitle(f"Ray fan at {field_angle:+.1f} deg", size=TITLE_SIZE)

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
        """Draw the curves; a dashed line marks the current beam angle."""
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
