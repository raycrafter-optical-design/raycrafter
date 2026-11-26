"""Icons for the tool palette and the context bar, drawn in code so no image files are shipped.

Each tool icon is one symbol in one colour, so the tools are easy to tell apart at a glance:
an arrow (Move), a paraxial lens, a biconvex lens, a user part, parallel rays (fields) and a
focus target (Autofocus). The focus-lock icon is an italic f beside a padlock.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap, QPolygonF

from raycrafter import palette

ICON_CANVAS_PX = 48  # icons are drawn on this square canvas; Qt scales them down
ITALIC_F = "\U0001d453"  # MATHEMATICAL ITALIC SMALL F: the cursive f of optics formulas
ITALIC_F_FONT_FAMILY = "Times New Roman"  # a serif italic gives the f its flowing tail

# One colour per tool, so the palette can be read by colour as well as shape.
MOVE_COLOUR = palette.TEXT
PARAXIAL_COLOUR = palette.ASSEMBLY  # purple
LENS_COLOUR = palette.LENS  # blue
USER_PART_COLOUR = palette.SAGITTAL  # orange
FIELD_COLOUR = "#d6336c"  # pink
AUTOFOCUS_COLOUR = palette.IMAGE  # green, the colour of the image plane
FOCUS_LOCKED_COLOUR = palette.GHOST_BLOCKED  # red
FOCUS_UNLOCKED_COLOUR = palette.AIRY  # grey


def italic_f_font(size_px: int) -> QFont:
    """A serif italic font of the given pixel size, for the italic f."""
    font = QFont(ITALIC_F_FONT_FAMILY)
    font.setItalic(True)
    font.setPixelSize(size_px)
    return font


def _draw_icon(
    colour: str, draw: Callable[[QPainter], None], width_px: int = ICON_CANVAS_PX
) -> QIcon:
    """An icon made by ``draw(painter)``, with the pen and brush already set to ``colour``."""
    pixmap = QPixmap(width_px, ICON_CANVAS_PX)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(
        QPen(
            QColor(colour),
            3.5,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
    )
    draw(painter)
    painter.end()
    return QIcon(pixmap)


def _points(*coordinates: tuple[float, float]) -> QPolygonF:
    return QPolygonF([QPointF(x, y) for x, y in coordinates])


def move_icon() -> QIcon:
    """A filled pointer arrow."""

    def draw(painter: QPainter) -> None:
        painter.setBrush(QColor(MOVE_COLOUR))
        painter.drawPolygon(
            _points((14, 8), (14, 38), (22, 30), (28, 42), (33, 40), (27, 28), (37, 28))
        )

    return _draw_icon(MOVE_COLOUR, draw)


def paraxial_lens_icon() -> QIcon:
    """The side view's symbol for a paraxial lens: a line with an arrowhead at each end."""

    def draw(painter: QPainter) -> None:
        painter.drawLine(QPointF(24, 8), QPointF(24, 40))
        painter.drawPolyline(_points((17, 14), (24, 7), (31, 14)))
        painter.drawPolyline(_points((17, 34), (24, 41), (31, 34)))

    return _draw_icon(PARAXIAL_COLOUR, draw)


def catalogue_lens_icon() -> QIcon:
    """A biconvex lens, filled translucent like the lenses in the side view."""

    def draw(painter: QPainter) -> None:
        fill = QColor(LENS_COLOUR)
        fill.setAlpha(90)
        painter.setBrush(fill)
        painter.drawEllipse(QRectF(16, 6, 16, 36))

    return _draw_icon(LENS_COLOUR, draw)


def user_part_icon() -> QIcon:
    """A box divided into three: several lenses held together as one part."""

    def draw(painter: QPainter) -> None:
        painter.drawRoundedRect(QRectF(9, 14, 30, 20), 3, 3)
        painter.drawLine(QPointF(19, 14), QPointF(19, 34))
        painter.drawLine(QPointF(29, 14), QPointF(29, 34))

    return _draw_icon(USER_PART_COLOUR, draw)


def field_icon() -> QIcon:
    """Three parallel rays: a beam."""

    def draw(painter: QPainter) -> None:
        for y in (14, 24, 34):
            painter.drawLine(QPointF(8, y), QPointF(40, y))

    return _draw_icon(FIELD_COLOUR, draw)


def autofocus_icon() -> QIcon:
    """A focus target: a circle with a dot and four ticks."""

    def draw(painter: QPainter) -> None:
        painter.drawEllipse(QRectF(12, 12, 24, 24))
        painter.drawPoint(QPointF(24, 24))
        for start, end in (
            ((24, 4), (24, 12)),
            ((24, 36), (24, 44)),
            ((4, 24), (12, 24)),
            ((36, 24), (44, 24)),
        ):
            painter.drawLine(QPointF(*start), QPointF(*end))

    return _draw_icon(AUTOFOCUS_COLOUR, draw)


def focus_lock_icon(locked: bool) -> QIcon:
    """An italic f beside a padlock: red and closed when locked, grey and open when not."""
    colour = QColor(FOCUS_LOCKED_COLOUR if locked else FOCUS_UNLOCKED_COLOUR)

    def draw(painter: QPainter) -> None:
        painter.setPen(QColor(palette.TEXT))
        painter.setFont(italic_f_font(40))
        painter.drawText(QRectF(0, 0, 30, ICON_CANVAS_PX), Qt.AlignmentFlag.AlignCenter, ITALIC_F)

        # The open shackle is lifted and shifted right, free on one side.
        body = QRectF(34, 22, 30, 22)
        shackle = QRectF(40, 6, 18, 26) if locked else QRectF(48, 0, 18, 26)
        painter.setPen(QPen(colour, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(shackle, 0, 180 * 16)  # Qt angles are in 1/16 of a degree
        middle_y = shackle.center().y()
        painter.drawLine(QPointF(shackle.left(), middle_y), QPointF(shackle.left(), body.top()))
        if locked:
            painter.drawLine(
                QPointF(shackle.right(), middle_y), QPointF(shackle.right(), body.top())
            )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colour)
        painter.drawRoundedRect(body, 4, 4)

    return _draw_icon(palette.TEXT, draw, width_px=72)
