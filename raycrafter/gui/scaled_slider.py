"""A slider with a number box beside it, whose scale gives more room to small values.

The context bar uses it for the focal length and the diameter of a paraxial lens. Most
lenses are short and narrow, so the slider's scale is a power law: with an exponent of 1.5,
0 to 250 mm takes 40% of a 1000 mm slider, rather than 25%. The box beside it takes any value
in its range, beyond the slider's end too; the handle then waits at the end.

A signed slider (the focal length) moves the size of the value and keeps its sign; the
sign is changed in the box or with the context bar's sign button.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QSizePolicy, QSlider, QWidget

from raycrafter.gui.undo_keys import leave_undo_keys_to_window

SLIDER_STEPS = 1000  # resolution of the slider
SLIDER_WIDTH_PX = 140
BOX_WIDTH_PX = 96  # room for "-9999.9 mm" and the arrows


def value_at_position(position: float, slider_maximum: float, exponent: float) -> float:
    """The value at a slider position from 0 (left end) to 1 (right end)."""
    return slider_maximum * position**exponent


def position_of_value(value: float, slider_maximum: float, exponent: float) -> float:
    """The slider position (0 to 1) of a value; values beyond the end sit at the end."""
    fraction = min(max(abs(value) / slider_maximum, 0.0), 1.0)
    return fraction ** (1.0 / exponent)


class ScaledSlider(QWidget):
    """A power-law slider and a number box that show one value.

    Parameters
    ----------
    slider_maximum : float
        The value at the slider's right end.
    exponent : float
        Of the power law; above 1 gives small values more room.
    box_range : (float, float)
        What the box accepts.
    suffix : str
        Shown after the number in the box, e.g. " mm".
    signed : bool
        If True, the slider sets the size of the value and keeps its sign.
    minimum_size : float
        The smallest size the slider sets (at its left end the value would be 0).

    Signals
    -------
    value_changed(float)
        The user moved the slider or changed the box. Not emitted by ``set_value``.
    """

    value_changed = pyqtSignal(float)

    def __init__(
        self,
        slider_maximum: float,
        exponent: float,
        box_range: tuple[float, float],
        suffix: str = " mm",
        signed: bool = False,
        minimum_size: float = 0.1,
    ) -> None:
        super().__init__()
        self.slider_maximum = slider_maximum
        self.exponent = exponent
        self.signed = signed
        self.minimum_size = minimum_size

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.setFixedWidth(SLIDER_WIDTH_PX)
        self.box = QDoubleSpinBox()
        self.box.setRange(*box_range)
        self.box.setDecimals(1)
        self.box.setSuffix(suffix)
        self.box.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.box.setFixedWidth(BOX_WIDTH_PX)
        leave_undo_keys_to_window(self.box)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.slider)
        layout.addWidget(self.box)
        # Only as wide as the slider and box: spare width would open a gap between them.
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.slider.valueChanged.connect(self._on_slider_moved)
        self.box.valueChanged.connect(self._on_box_changed)

    def value(self) -> float:
        """The value shown in the box."""
        return self.box.value()

    def set_value(self, value: float) -> None:
        """Show a value (e.g. the selected lens's) without emitting ``value_changed``."""
        self.box.blockSignals(True)
        self.box.setValue(value)
        self.box.blockSignals(False)
        self._place_handle(value)

    def setToolTip(self, text: str) -> None:  # noqa: N802 (Qt's name)
        """The same tooltip on the slider and the box."""
        self.slider.setToolTip(text)
        self.box.setToolTip(text)

    def _place_handle(self, value: float) -> None:
        position = position_of_value(value, self.slider_maximum, self.exponent)
        self.slider.blockSignals(True)
        self.slider.setValue(round(position * SLIDER_STEPS))
        self.slider.blockSignals(False)

    def _on_slider_moved(self, step: int) -> None:
        """Turn the slider position into a value (with the box's sign), show it and report it."""
        size = value_at_position(step / SLIDER_STEPS, self.slider_maximum, self.exponent)
        size = max(size, self.minimum_size)
        value = -size if (self.signed and self.box.value() < 0) else size
        self.box.blockSignals(True)  # the box's own signal would move the handle back
        self.box.setValue(value)
        self.box.blockSignals(False)
        self.value_changed.emit(self.box.value())

    def _on_box_changed(self, value: float) -> None:
        self._place_handle(value)
        self.value_changed.emit(value)
