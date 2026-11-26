"""The gap dimensions on the side view, and the box in which a new gap is typed.

Each air gap is drawn along the bottom of the view as a line with arrowheads between the two
vertices it runs between, labelled with its length. Labels that would overlap are raised to
another row. A label is coloured by whether the gap can be typed; a locked gap's label is
also boxed. ``GapDimensions.update`` runs on every change, including each step of a drag, so
all the lines are one plot item and a label's text is set only when it changes.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QFocusEvent, QKeyEvent
from PyQt6.QtWidgets import QAbstractSpinBox, QDoubleSpinBox, QWidget

from raycrafter import palette
from raycrafter.core.gaps import AirGap
from raycrafter.gui.style import join_with_gaps, scaled_font

BOTTOM_MARGIN_PX = 4  # from the bottom of the plot to the lowest dimension line
ROW_HEIGHT_PX = 16  # between one row of dimensions and the next
LABEL_MARGIN_PX = 1  # around the text; kept small so the lowest row stays clear of the lenses
ROW_COUNT = 3  # labels that would overlap are spread over at most this many rows
LABEL_SPACE_PX = 4  # smallest space between two labels in a row
ARROW_LENGTH_PX = 6
ARROW_HALF_HEIGHT_PX = 3
LABEL_FONT_SCALE = 0.85
LABEL_Z_VALUE = 70  # above the lenses, their drag regions and the image line
MAX_TYPED_GAP_MM = 100000.0
EDITOR_WIDTH_PX = 100

# How a gap's label is drawn.
EDITABLE = "editable"  # can be typed
FIXED = "fixed"  # cannot be typed (e.g. "Lock focus" sets it)
LOCKED = "locked"  # locked by the user, or inside a Plossl pair
LABEL_COLOURS = {
    EDITABLE: palette.GAP_EDITABLE,
    FIXED: palette.GAP_FIXED,
    LOCKED: palette.GAP_LOCKED,
}


def gap_text(width_mm: float) -> str:
    """The label of a gap, e.g. ``"47.30 mm"``."""
    return f"{width_mm + 0.0:.2f} mm".replace("-0.00 ", "0.00 ")


class GapDimensions:
    """The plot items that dimension every air gap.

    Parameters
    ----------
    plot_item : pyqtgraph.PlotItem
        The side view's plot.
    """

    def __init__(self, plot_item: pg.PlotItem) -> None:
        self.plot_item = plot_item
        self.gaps: list[AirGap] = []  # those drawn, in the order of the labels
        self.labels: list[pg.TextItem] = []  # a pool: only the first len(gaps) are shown

        # Faint lines dropping from each vertex on the axis to its dimension line.
        self.extension_curve = pg.PlotCurveItem(
            pen=pg.mkPen(palette.FAINT, width=1, style=Qt.PenStyle.DotLine), connect="finite"
        )
        self.line_curve = pg.PlotCurveItem(
            pen=pg.mkPen(palette.GAP_LINE, width=1), connect="finite"
        )
        for curve in (self.extension_curve, self.line_curve):
            curve.setZValue(-5)  # above the grid, behind the lenses and rays
            plot_item.addItem(curve, ignoreBounds=True)

    def update(self, gaps: list[AirGap], label_style: Callable[[AirGap], str]) -> None:
        """Draw these gaps at the current axis ranges.

        ``label_style`` gives each gap's look: EDITABLE, FIXED or LOCKED.
        """
        self.gaps = gaps
        while len(self.labels) < len(gaps):
            self.labels.append(self._new_label())
        for label in self.labels[len(gaps) :]:
            label.hide()

        view_box = self.plot_item.vb
        mm_per_px_z, mm_per_px_y = view_box.viewPixelSize()
        bottom_y = view_box.viewRange()[1][0]
        row_right_edges_px = [-np.inf] * ROW_COUNT

        lines_z, lines_y, extensions_z, extensions_y = [], [], [], []
        for gap, label in zip(gaps, self.labels):
            self._set_label_text(label, gap_text(gap.width_mm), label_style(gap))

            # The lowest row in which this label clears the one before it.
            middle_z = 0.5 * (gap.start_z + gap.end_z)
            half_width_px = 0.5 * label.boundingRect().width()
            left_px = middle_z / mm_per_px_z - half_width_px
            row = next(
                (
                    row
                    for row, right_edge in enumerate(row_right_edges_px)
                    if left_px >= right_edge + LABEL_SPACE_PX
                ),
                int(np.argmin(row_right_edges_px)),  # all full: the one that overlaps least
            )
            row_right_edges_px[row] = left_px + 2.0 * half_width_px

            line_y = bottom_y + (BOTTOM_MARGIN_PX + row * ROW_HEIGHT_PX) * mm_per_px_y
            label.setPos(middle_z, line_y)
            label.show()

            for end_z in (gap.start_z, gap.end_z):
                extensions_z.append([end_z, end_z])
                extensions_y.append([0.0, line_y])
            lines_z.append([gap.start_z, gap.end_z])
            lines_y.append([line_y, line_y])

            # Arrowheads pointing outwards at the two ends, if the gap has room for both.
            arrow_z = ARROW_LENGTH_PX * mm_per_px_z
            arrow_y = ARROW_HALF_HEIGHT_PX * mm_per_px_y
            if gap.width_mm >= 2.0 * arrow_z:
                for tip_z, inwards in ((gap.start_z, 1.0), (gap.end_z, -1.0)):
                    base_z = tip_z + inwards * arrow_z
                    lines_z.append([base_z, tip_z, base_z])
                    lines_y.append([line_y + arrow_y, line_y, line_y - arrow_y])

        self._set_curve(self.line_curve, lines_z, lines_y)
        self._set_curve(self.extension_curve, extensions_z, extensions_y)

    def gap_at(self, scene_position: QPointF) -> AirGap | None:
        """The gap whose label is under the pointer, if any."""
        for gap, label in zip(self.gaps, self.labels):
            if label.isVisible() and label.sceneBoundingRect().contains(scene_position):
                return gap
        return None

    def label_scene_centre(self, lens_index: int) -> QPointF:
        """Where the label of the gap behind this lens is, in scene coordinates.

        ``lens_index`` is SOURCE_GAP_INDEX for the gap from the source plane.
        """
        position = [gap.lens_index for gap in self.gaps].index(lens_index)
        return self.labels[position].sceneBoundingRect().center()

    def _new_label(self) -> pg.TextItem:
        label = pg.TextItem(anchor=(0.5, 1), fill=pg.mkBrush(255, 255, 255, 220))
        label.setFont(scaled_font(label.textItem.font(), LABEL_FONT_SCALE))
        label.textItem.document().setDocumentMargin(LABEL_MARGIN_PX)
        label.setZValue(LABEL_Z_VALUE)
        label.shown_as = None  # (text, style) last set, to avoid laying text out again
        self.plot_item.addItem(label, ignoreBounds=True)
        return label

    @staticmethod
    def _set_label_text(label: pg.TextItem, text: str, style: str) -> None:
        if label.shown_as == (text, style):
            return
        colour = LABEL_COLOURS[style]
        label.setText(text, color=colour)
        # pyqtgraph keeps the border pen in ``label.border`` and draws it in ``paint``.
        label.border = pg.mkPen(colour if style == LOCKED else None)
        label.update()
        label.shown_as = (text, style)

    @staticmethod
    def _set_curve(curve: pg.PlotCurveItem, segments_z: list, segments_y: list) -> None:
        if segments_z:
            curve.setData(*join_with_gaps(segments_z, segments_y))
        else:
            curve.setData([], [])


class GapEditor(QDoubleSpinBox):
    """A number box laid over a gap's label, for typing the new length.

    Enter, or clicking elsewhere, accepts the value; Escape abandons it. The box removes
    itself either way.

    Signals
    -------
    accepted(int, float)
        A different value was entered for the gap behind this lens: (lens index, mm).
    closed
        The box has gone, whether or not a value was accepted.
    """

    accepted = pyqtSignal(int, float)
    closed = pyqtSignal()

    def __init__(self, gap: AirGap, centre: QPointF, parent: QWidget) -> None:
        super().__init__(parent)
        self.lens_index = gap.lens_index
        self.is_open = True
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setDecimals(3)
        self.setRange(0.0, MAX_TYPED_GAP_MM)
        self.setSuffix(" mm")
        self.setValue(gap.width_mm)
        self.initial_value = self.value()  # the gap as shown, rounded to the box's decimals

        height = self.sizeHint().height()
        left = min(max(centre.x() - EDITOR_WIDTH_PX / 2, 0), parent.width() - EDITOR_WIDTH_PX)
        top = min(max(centre.y() - height / 2, 0), parent.height() - height)
        self.setGeometry(int(left), int(top), EDITOR_WIDTH_PX, height)
        self.show()
        self.setFocus()
        self.selectAll()

    def accept_value(self) -> None:
        """Close the box and report its value, unless it is the one it started with."""
        if not self.is_open:
            return
        self.interpretText()
        value = self.value()
        self.close_editor()
        if value != self.initial_value:  # re-entering the rounded value must not move lenses
            self.accepted.emit(self.lens_index, value)

    def close_editor(self) -> None:
        """Remove the box without applying anything."""
        if not self.is_open:
            return
        self.is_open = False  # first: hiding a focused box sends it a focus-out
        self.hide()
        self.deleteLater()
        self.closed.emit()

    def event(self, event: QEvent) -> bool:
        """Keep Escape for this box; the window also uses it as a shortcut."""
        if event.type() == QEvent.Type.ShortcutOverride and event.key() == Qt.Key.Key_Escape:
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Enter accepts and Escape abandons."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.accept_value()
        elif event.key() == Qt.Key.Key_Escape:
            self.close_editor()
        else:
            super().keyPressEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        """Clicking elsewhere accepts; switching application or opening a menu does not."""
        super().focusOutEvent(event)
        temporary = (Qt.FocusReason.ActiveWindowFocusReason, Qt.FocusReason.PopupFocusReason)
        if event.reason() not in temporary:
            self.accept_value()
