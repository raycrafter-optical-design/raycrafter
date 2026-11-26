"""A small pair of axes in a corner of the side view, showing that y is up and z is right.

It replaces the side view's axis titles ("z (mm)", "y (mm)"), which took a lot of room to
say little: every length on the bench is in mm, so the unit is left out.
"""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsItem

from raycrafter import palette

GLYPH_SIZE_PX = 25  # length of each arm
CORNER_GAP_PX = 6  # distance from the bottom-left corner of the plot area
ARROWHEAD_PX = 4
LETTER_SIZE_PX = 10
GLYPH_OPACITY = 0.55  # faint, so it never competes with the rays


class AxisGlyph(QGraphicsItem):
    """Two arrows from a common origin: z to the right, y upwards, each with its letter.

    It is drawn in screen pixels, so it keeps its size whatever the axis ranges are. It is a
    child of the plot item (not of the view box, which would scale it with the data); call
    ``place`` whenever the view box is resized.

    Parameters
    ----------
    plot_item : pg.PlotItem
        The side view's plot; the glyph sits in the bottom-left corner of its data area.
    """

    def __init__(self, plot_item: pg.PlotItem) -> None:
        super().__init__(plot_item)
        self.plot_item = plot_item
        self.setOpacity(GLYPH_OPACITY)
        self.setZValue(30)  # above the grid and rays, below the captions
        # The glyph is decoration: clicks and hovers pass through to the bench beneath it.
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setAcceptHoverEvents(False)
        self.letter_font = QFont()
        self.letter_font.setPixelSize(LETTER_SIZE_PX)
        self.letter_font.setItalic(True)

    def boundingRect(self) -> QRectF:
        """The arrows run from (0, 0) to the right and upwards (negative y on screen)."""
        margin = LETTER_SIZE_PX
        return QRectF(
            -2, -GLYPH_SIZE_PX - margin, GLYPH_SIZE_PX + margin + 2, GLYPH_SIZE_PX + margin + 2
        )

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        """Draw the two arrows and their letters."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        colour = QColor(palette.TEXT)
        painter.setPen(QPen(colour, 1.2))
        painter.drawLine(QPointF(0, 0), QPointF(GLYPH_SIZE_PX, 0))
        painter.drawLine(QPointF(0, 0), QPointF(0, -GLYPH_SIZE_PX))

        heads = QPainterPath()
        heads.moveTo(GLYPH_SIZE_PX, 0)
        heads.lineTo(GLYPH_SIZE_PX - ARROWHEAD_PX, -ARROWHEAD_PX / 2)
        heads.lineTo(GLYPH_SIZE_PX - ARROWHEAD_PX, ARROWHEAD_PX / 2)
        heads.closeSubpath()
        heads.moveTo(0, -GLYPH_SIZE_PX)
        heads.lineTo(-ARROWHEAD_PX / 2, -GLYPH_SIZE_PX + ARROWHEAD_PX)
        heads.lineTo(ARROWHEAD_PX / 2, -GLYPH_SIZE_PX + ARROWHEAD_PX)
        heads.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colour)
        painter.drawPath(heads)

        # Each letter sits just past the tip of its arrow, centred on it.
        painter.setPen(colour)
        painter.setFont(self.letter_font)
        letter_box = QRectF(0, 0, LETTER_SIZE_PX, LETTER_SIZE_PX)
        painter.drawText(
            letter_box.translated(GLYPH_SIZE_PX + 1, -LETTER_SIZE_PX / 2 - 1),
            Qt.AlignmentFlag.AlignCenter,
            "z",
        )
        painter.drawText(
            letter_box.translated(-LETTER_SIZE_PX / 2, -GLYPH_SIZE_PX - LETTER_SIZE_PX - 1),
            Qt.AlignmentFlag.AlignCenter,
            "y",
        )

    def place(self, *_args) -> None:
        """Move the glyph's origin to just inside the bottom-left corner of the data area."""
        data_area = self.plot_item.vb.geometry()
        self.setPos(
            data_area.left() + CORNER_GAP_PX + ARROWHEAD_PX,
            data_area.bottom() - CORNER_GAP_PX - ARROWHEAD_PX,
        )
