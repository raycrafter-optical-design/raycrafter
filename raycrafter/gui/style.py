"""The look of the GUI: the Qt style sheet, the app icon and small plotting helpers.

Colours come from ``raycrafter.palette``, shared with the HTML report.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import QApplication

from raycrafter import palette

PLOT_BOTTOM_GAP_PX = 5  # space between each plot's bottom axis label and its border
PLOT_MIN_SIZE_PX = (280, 260)  # smallest size of each analysis plot

# COMBO_ARROW_PATH is replaced at start-up by the path of a generated image; see
# write_combo_arrow_image.
STYLE_SHEET = """
QWidget { font-size: 13px; }
QMainWindow { background: #f1f3f5; }
QFrame#controls { background: #f8f9fa; border: 1px solid #ced4da; border-radius: 8px; }
QFrame#controls QLabel { border: none; background: transparent; }
QPushButton { background: white; border: 1px solid #ced4da; border-radius: 4px; padding: 6px 10px; }
QPushButton:hover { background: #e7f0ff; }
QPushButton:disabled { color: #adb5bd; background: #f1f3f5; }
QComboBox { background: white; border: 1px solid #ced4da; border-radius: 4px; padding: 4px 8px; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 24px;
                       border: none; border-left: 1px solid #dee2e6; }
QComboBox::down-arrow { image: url(COMBO_ARROW_PATH); width: 10px; height: 10px; }
QComboBox QAbstractItemView {
    background: white; color: #343a40; border: 1px solid #ced4da; outline: 0;
    selection-background-color: #3b7dd8; selection-color: white;
}
QComboBox QAbstractItemView::item { padding: 4px 8px; }
QComboBox QAbstractItemView::item:hover, QComboBox QAbstractItemView::item:selected {
    background: #3b7dd8; color: white;
}
QLabel#sliderLabel { color: #495057; }
QLabel#hintLabel { color: #868e96; font-size: 11px; }
QTabWidget::pane { border: none; }
QTabWidget::tab-bar { alignment: left; }
QTabBar { font-size: 11px; }
QTabBar::tab { background: transparent; color: #868e96; padding: 1px 10px 2px 10px; min-width: 90px;
               margin-right: 2px; border-bottom: 2px solid transparent; }
QTabBar::tab:hover { color: #343a40; }
QTabBar::tab:selected { color: #343a40; border-bottom: 2px solid #3b7dd8; }
"""


def colour_with_alpha(colour: str, alpha: int) -> QColor:
    """A QColor from a hex string with the given alpha (0-255)."""
    qt_colour = pg.mkColor(colour)
    qt_colour.setAlpha(alpha)
    return qt_colour


def make_app_icon() -> QIcon:
    """A biconvex lens with three converging rays on a rounded tile, drawn in code.

    Set on the QApplication, this replaces the generic Python rocket in the macOS Dock and
    the window icon elsewhere, without shipping an image file.
    """
    size = 512
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Rounded tile, with the usual margin macOS icons leave round the edge.
    margin = size * 0.06
    tile = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
    painter.setPen(QPen(QColor("#ced4da"), 4))
    painter.setBrush(QBrush(QColor("#f8f9fa")))
    painter.drawRoundedRect(tile, size * 0.2, size * 0.2)

    centre_x, centre_y = size * 0.40, size * 0.5
    half_height, half_width = size * 0.30, size * 0.12

    # Converging rays: parallel on the left, meeting at a point to the right of the lens.
    focus_x = size * 0.84
    painter.setPen(QPen(QColor(palette.RAY), 11, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    for ray_offset in (-0.19, 0.0, 0.19):
        ray_y = centre_y + size * ray_offset
        painter.drawLine(QPointF(size * 0.14, ray_y), QPointF(centre_x, ray_y))
        painter.drawLine(QPointF(centre_x, ray_y), QPointF(focus_x, centre_y))

    # The lens: two arcs bulging outwards, filled translucent blue with a solid outline.
    lens_path = QPainterPath()
    lens_path.moveTo(centre_x, centre_y - half_height)
    lens_path.quadTo(centre_x + 2 * half_width, centre_y, centre_x, centre_y + half_height)
    lens_path.quadTo(centre_x - 2 * half_width, centre_y, centre_x, centre_y - half_height)
    lens_pen = QPen(
        QColor(palette.LENS),
        16,
        Qt.PenStyle.SolidLine,
        Qt.PenCapStyle.RoundCap,
        Qt.PenJoinStyle.RoundJoin,
    )
    painter.setPen(lens_pen)
    painter.setBrush(QBrush(colour_with_alpha(palette.LENS, 90)))
    painter.drawPath(lens_path)

    # Focus point.
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(palette.IMAGE)))
    painter.drawEllipse(QPointF(focus_x, centre_y), 15, 15)
    painter.end()
    return QIcon(pixmap)


def write_combo_arrow_image() -> str:
    """Draw a small flat down-chevron to a PNG in the temp folder and return its path.

    The native macOS combo-box arrow is drawn with shading (dark right and bottom edges). A
    style sheet can only replace it with an image file, so one is generated here.
    """
    size = 20
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    chevron_pen = QPen(
        QColor("#495057"),
        2.4,
        Qt.PenStyle.SolidLine,
        Qt.PenCapStyle.RoundCap,
        Qt.PenJoinStyle.RoundJoin,
    )
    painter.setPen(chevron_pen)
    painter.drawPolyline(QPolygonF([QPointF(4, 7), QPointF(10, 13), QPointF(16, 7)]))
    painter.end()

    arrow_path = Path(tempfile.gettempdir()) / "raycrafter_combo_arrow.png"
    pixmap.save(str(arrow_path))
    return arrow_path.as_posix()


def force_light_colour_scheme(app: QApplication) -> None:
    """Make Qt use its light palette whatever the operating system's light/dark setting is.

    The style sheet paints light backgrounds but leaves most text colours to the palette. With
    Windows 11 in dark mode the palette gives white text, so buttons became white on white.
    The whole design is light, so we pin the scheme instead of following the system.
    """
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)


def style_sheet() -> str:
    """The application style sheet, with the generated combo-box arrow filled in."""
    return STYLE_SHEET.replace("COMBO_ARROW_PATH", write_combo_arrow_image())


def join_with_gaps(x_segments: list, y_segments: list) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate several polylines into one pair of arrays, separated by NaN.

    pyqtgraph draws a single curve much faster than many, and ``connect="finite"`` breaks
    the line at each NaN, so all rays (or outlines) can live in one curve item.
    """
    x_parts, y_parts = [], []
    for x_values, y_values in zip(x_segments, y_segments):
        x_parts += [np.asarray(x_values, dtype=float), [np.nan]]
        y_parts += [np.asarray(y_values, dtype=float), [np.nan]]
    return np.concatenate(x_parts), np.concatenate(y_parts)


def style_plot(widget: pg.PlotWidget, x_label: str = "", y_label: str = "") -> None:
    """Give a plot widget the shared look: no mouse interaction, light grid, labelled axes."""
    widget.setStyleSheet(
        f"QGraphicsView {{ border: 1px solid {palette.CANVAS_BORDER}; border-radius: 4px; }}"
    )
    plot = widget.getPlotItem()
    plot.setMouseEnabled(False, False)
    plot.hideButtons()
    plot.setMenuEnabled(False)
    plot.layout.setContentsMargins(0, 0, 0, PLOT_BOTTOM_GAP_PX)  # lifts the x label off the border
    plot.showGrid(x=True, y=True, alpha=0.15)
    plot.setLabel("bottom", x_label)
    plot.setLabel("left", y_label)


def scaled_font(font, scale: float):
    """A copy of a QFont, scaled in whichever unit (pixels or points) it is specified in."""
    scaled = type(font)(font)
    if scaled.pixelSize() > 0:
        scaled.setPixelSize(max(1, round(scaled.pixelSize() * scale)))
    else:
        scaled.setPointSizeF(scaled.pointSizeF() * scale)
    return scaled
