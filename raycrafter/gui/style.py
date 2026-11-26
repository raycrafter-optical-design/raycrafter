"""The look of the GUI: the Qt style sheet, the app icon and small plotting helpers.

Colours come from ``raycrafter.palette``, shared with the HTML report.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QObject, QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import QApplication

from raycrafter import palette

PLOT_BOTTOM_GAP_PX = 5  # space between each plot's bottom axis label and its border
PLOT_MIN_SIZE_PX = (280, 260)  # smallest size of each analysis plot
CONTEXT_BAR_BOX_HEIGHT_PX = 24  # drop-downs in the context bar match its spin boxes

# DOWN_ARROW_PATH and UP_ARROW_PATH are replaced at start-up by the paths of generated
# images; see write_arrow_image.
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
QComboBox::down-arrow { image: url(DOWN_ARROW_PATH); width: 10px; height: 10px; }
QComboBox QAbstractItemView {
    background: white; color: #343a40; border: 1px solid #ced4da; outline: 0;
    selection-background-color: #3b7dd8; selection-color: white;
}
QComboBox QAbstractItemView::item { padding: 4px 8px; }
QComboBox QAbstractItemView::item:hover, QComboBox QAbstractItemView::item:selected {
    background: #3b7dd8; color: white;
}
QLineEdit { background: white; border: 1px solid #ced4da; border-radius: 4px; padding: 4px 6px; }
QLineEdit:focus { border: 1px solid #3b7dd8; }
QTreeView { background: white; border: 1px solid #ced4da; border-radius: 4px; outline: 0; }
QTreeView::item { padding: 2px 0px; }
QTreeView#lensTree { font-size: 12px; }  /* the lens browser: 90% of the usual size */
QTreeView::item:selected { background: #3b7dd8; color: white; }
QHeaderView::section { background: #f8f9fa; color: #495057; border: none;
                       border-bottom: 1px solid #dee2e6; padding: 4px 6px; }
QLabel#sliderLabel { color: #495057; }
QLabel#hintLabel { color: #868e96; font-size: 11px; }
QTabWidget::pane { border: none; }
QTabWidget::tab-bar { alignment: left; }
QTabBar { font-size: 11px; }
QTabBar::tab { background: transparent; color: #868e96; padding: 1px 10px 2px 10px; min-width: 90px;
               margin-right: 2px; border-bottom: 2px solid transparent; }
QTabBar::tab:hover { color: #343a40; }
QTabBar::tab:selected { color: #343a40; border-bottom: 2px solid #3b7dd8; }

/* The tool palette and the context bar: compact and flat, as in a graphics editor. */
QToolBar#contextBar { background: #e9ecef; border: none; border-bottom: 1px solid #ced4da;
                      spacing: 6px; }
QToolBar#toolPalette { background: #e9ecef; border: none; border-right: 1px solid #ced4da;
                       spacing: 2px; padding: 4px 7px; }
QToolBar QLabel { font-size: 11px; color: #495057; }
QToolBar QLabel#italicF { font-family: 'Times New Roman'; font-style: italic; font-size: 18px;
                          color: #343a40; }
QToolBar QLabel#documentTitle { font-size: 12px; font-weight: 600; color: #868e96;
                                padding: 0px 8px; }
QToolBar QLabel#diameterSign { font-size: 15px; color: #343a40; }
QToolBar QToolButton { border: none; border-radius: 5px; padding: 3px; color: #343a40; }
QToolBar QToolButton:hover { background: #dee2e6; }
QToolBar QToolButton:checked { background: #cfd4da; }
QToolBar#contextBar QToolButton { border: 1px solid #ced4da; background: #f8f9fa; padding: 0px; }
QToolBar#contextBar QToolButton#signButton { font-size: 18px; min-width: 22px; max-width: 22px;
                                             min-height: 22px; max-height: 22px; }
/* Show or hide positive and negative lenses: pressed (hidden) is drawn faded. */
QToolBar#contextBar QToolButton#lensSignButton { min-width: 22px; max-width: 22px;
    min-height: 22px; max-height: 22px; }
QToolBar#contextBar QToolButton#lensSignButton:checked { background: #f1f3f5;
    border-color: #dee2e6; }
QToolBar#contextBar QToolButton#lockFocusButton { min-width: 40px; max-width: 40px;
                                                  min-height: 28px; max-height: 28px; }
QToolBar#contextBar QToolButton#lockFocusButton:checked { background: #fff5f5;
                                                          border-color: #e03131; }
QToolBar QDoubleSpinBox, QToolBar QComboBox { font-size: 11px; padding: 2px 6px;
                                              min-height: 18px; }
/* combobox-popup: 0 drops an ordinary list below the box, tall enough for every item: the
   macOS style otherwise lays the popup over the box and showed only one row. */
QToolBar QComboBox { combobox-popup: 0; }
QToolBar QComboBox::drop-down { width: 16px; border: none; }
QToolBar QComboBox::down-arrow { width: 9px; height: 9px; }
QToolBar QComboBox QAbstractItemView { font-size: 11px; }
QToolBar QComboBox QAbstractItemView::item { padding: 0px 8px; height: 24px; }
/* Spin box arrows drawn on the border, so the text field keeps its width: Qt's macOS style
   otherwise collapses a styled spin box's text field to 1 px. */
QToolBar QDoubleSpinBox { background: white; border: 1px solid #ced4da; border-radius: 4px;
                          padding-right: 16px; }
QToolBar QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right;
    width: 14px; border: none; border-left: 1px solid #dee2e6; border-top-right-radius: 4px; }
QToolBar QDoubleSpinBox::down-button { subcontrol-origin: border;
    subcontrol-position: bottom right; width: 14px; border: none;
    border-left: 1px solid #dee2e6; border-bottom-right-radius: 4px; }
QToolBar QDoubleSpinBox::up-button:hover, QToolBar QDoubleSpinBox::down-button:hover {
    background: #e7f0ff; }
QToolBar QDoubleSpinBox::up-arrow { image: url(UP_ARROW_PATH); width: 7px; height: 7px; }
QToolBar QDoubleSpinBox::down-arrow { image: url(DOWN_ARROW_PATH); width: 7px; height: 7px; }
QSlider::groove:horizontal { height: 3px; background: #ced4da; border-radius: 1px; }
QSlider::sub-page:horizontal { background: #3b7dd8; border-radius: 1px; }
QSlider::handle:horizontal { background: white; border: 1px solid #868e96; width: 10px;
                             height: 10px; margin: -4px 0px; border-radius: 5px; }
QSlider::handle:horizontal:hover { border-color: #3b7dd8; }
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


def write_arrow_image(pointing_up: bool = False) -> str:
    """Draw a small flat chevron to a PNG in the temp folder and return its path.

    The native macOS combo-box arrow is drawn with shading (dark right and bottom edges), and
    a styled spin box loses its arrows. A style sheet can only replace them with image files,
    so these are generated here.
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
    if pointing_up:
        painter.drawPolyline(QPolygonF([QPointF(4, 13), QPointF(10, 7), QPointF(16, 13)]))
    else:
        painter.drawPolyline(QPolygonF([QPointF(4, 7), QPointF(10, 13), QPointF(16, 7)]))
    painter.end()

    file_name = "raycrafter_up_arrow.png" if pointing_up else "raycrafter_combo_arrow.png"
    arrow_path = Path(tempfile.gettempdir()) / file_name
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
    """The application style sheet, with the generated arrow images filled in."""
    return STYLE_SHEET.replace("DOWN_ARROW_PATH", write_arrow_image()).replace(
        "UP_ARROW_PATH", write_arrow_image(pointing_up=True)
    )


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


class WheelIgnorer(QObject):
    """Event filter that swallows mouse-wheel and trackpad scroll events.

    A plot is a QGraphicsView, which scrolls its contents on a wheel event even though its
    scrollbars are hidden. The scroll range is a few pixels, so a two-finger swipe over a plot
    visibly shifts the whole picture. Plots do not zoom or pan, so the wheel does nothing here.
    """

    def eventFilter(self, watched_object: QObject, event: QEvent) -> bool:
        """Return True for wheel events, so Qt never scrolls the plot; pass other events on."""
        return event.type() == QEvent.Type.Wheel


def style_plot(widget: pg.PlotWidget, x_label: str = "", y_label: str = "") -> None:
    """Give a plot widget the shared look: no mouse interaction, light grid, labelled axes.

    The wheel is also switched off, so a trackpad swipe cannot move the plot.
    """
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
    # Wheel events are sent to the viewport, so the filter goes there. It is parented to the
    # viewport so that it lives as long as the plot does.
    widget.viewport().installEventFilter(WheelIgnorer(widget.viewport()))


def scaled_font(font, scale: float):
    """A copy of a QFont, scaled in whichever unit (pixels or points) it is specified in."""
    scaled = type(font)(font)
    if scaled.pixelSize() > 0:
        scaled.setPixelSize(max(1, round(scaled.pixelSize() * scale)))
    else:
        scaled.setPointSizeF(scaled.pointSizeF() * scale)
    return scaled
