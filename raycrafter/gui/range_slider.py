"""A horizontal slider with two handles, for choosing a range (Qt has none built in)."""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from raycrafter import palette

HANDLE_RADIUS_PX = 7
GROOVE_HEIGHT_PX = 4


class RangeSlider(QWidget):
    """Two handles on one groove: the low and high ends of a range of whole numbers.

    The handles cannot cross. The part of the groove between them is drawn in the accent
    colour. Pressing anywhere on the groove grabs the nearer handle.

    Parameters
    ----------
    minimum, maximum : int
        The slider's range; both handles start at the ends of it.
    handle_radius_px : int
        Size of the handles; the context bar uses small ones.

    Signals
    -------
    values_changed(int, int)
        The user moved a handle; gives the new low and high values.
    """

    values_changed = pyqtSignal(int, int)

    def __init__(
        self, minimum: int = 0, maximum: int = 1000, handle_radius_px: int = HANDLE_RADIUS_PX
    ) -> None:
        super().__init__()
        self.minimum, self.maximum = minimum, maximum
        self.handle_radius_px = handle_radius_px
        self.low_value, self.high_value = minimum, maximum
        self.dragged_handle: str | None = None  # "low" or "high" while a handle is held
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def sizeHint(self) -> QSize:
        """Wide enough to be usable; tall enough for the handles."""
        return QSize(160, 2 * self.handle_radius_px + 6)

    def values(self) -> tuple[int, int]:
        """The low and high values."""
        return self.low_value, self.high_value

    def set_values(self, low: int, high: int) -> None:
        """Move both handles (clipped to the range) without emitting ``values_changed``."""
        low = min(max(low, self.minimum), self.maximum)
        high = min(max(high, low), self.maximum)
        self.low_value, self.high_value = low, high
        self.update()

    # ------------------------------------------------------------- geometry

    def _groove_span_px(self) -> tuple[float, float]:
        """x of the groove's left and right ends; the handle centres travel between them."""
        return self.handle_radius_px + 1.0, self.width() - self.handle_radius_px - 1.0

    def _x_of_value(self, value: int) -> float:
        left, right = self._groove_span_px()
        fraction = (value - self.minimum) / max(self.maximum - self.minimum, 1)
        return left + fraction * (right - left)

    def _value_at_x(self, x: float) -> int:
        left, right = self._groove_span_px()
        fraction = min(max((x - left) / max(right - left, 1.0), 0.0), 1.0)
        return round(self.minimum + fraction * (self.maximum - self.minimum))

    # ------------------------------------------------------------- drawing

    def paintEvent(self, _event) -> None:
        """Groove, highlighted span between the handles, and the two round handles."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        centre_y = self.height() / 2.0
        left, right = self._groove_span_px()
        low_x, high_x = self._x_of_value(self.low_value), self._x_of_value(self.high_value)

        painter.setPen(Qt.PenStyle.NoPen)
        groove_top = centre_y - GROOVE_HEIGHT_PX / 2.0
        painter.setBrush(QBrush(QColor(palette.FAINT)))
        painter.drawRoundedRect(QRectF(left, groove_top, right - left, GROOVE_HEIGHT_PX), 2, 2)
        painter.setBrush(QBrush(QColor(palette.LENS)))
        painter.drawRect(QRectF(low_x, groove_top, high_x - low_x, GROOVE_HEIGHT_PX))

        painter.setPen(QPen(QColor(palette.LENS), 1.5))
        painter.setBrush(QBrush(QColor("white")))
        for handle_x in (low_x, high_x):
            radius = self.handle_radius_px
            painter.drawEllipse(QPointF(handle_x, centre_y), radius, radius)
        painter.end()

    # ------------------------------------------------------------- mouse

    def mousePressEvent(self, event) -> None:
        """Grab the handle nearer the pointer and move it there."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pointer_x = event.position().x()
        low_distance = abs(pointer_x - self._x_of_value(self.low_value))
        high_distance = abs(pointer_x - self._x_of_value(self.high_value))
        if low_distance == high_distance:
            # The handles sit on top of each other: pick by which side the pointer is on.
            self.dragged_handle = "low" if pointer_x < self._x_of_value(self.low_value) else "high"
        else:
            self.dragged_handle = "low" if low_distance < high_distance else "high"
        self._move_dragged_handle(pointer_x)

    def mouseMoveEvent(self, event) -> None:
        """Drag the grabbed handle; it stops at the other one."""
        if self.dragged_handle is not None:
            self._move_dragged_handle(event.position().x())

    def mouseReleaseEvent(self, _event) -> None:
        """Let go of the handle."""
        self.dragged_handle = None

    def _move_dragged_handle(self, pointer_x: float) -> None:
        value = self._value_at_x(pointer_x)
        if self.dragged_handle == "low":
            new_values = (min(value, self.high_value), self.high_value)
        else:
            new_values = (self.low_value, max(value, self.low_value))
        if new_values != self.values():
            self.low_value, self.high_value = new_values
            self.update()
            self.values_changed.emit(*new_values)
