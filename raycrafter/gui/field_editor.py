"""The field window: a small, non-modal window for adding, removing and editing fields.

A field is a beam of light entering the bench (``raycrafter.core.state.BeamField``). The
list on the left shows every field with a swatch of its colour; "+" adds a field (a copy of
the selected one) and "-" removes the selected fields (one always remains). The form on the
right edits the field that is current in the list, which is also the field the single-field
analyses use. Every change is applied to the model at once, so the user sees its effect
while the window is open.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from raycrafter.core.model import OpticalModel
from raycrafter.core.state import BeamField

SWATCH_SIZE_PX = 12  # the colour square beside each field in the list
LIST_WIDTH_PX = 110

# The editable properties: (BeamField attribute, label, minimum, maximum, step, decimals).
FIELD_PROPERTIES = [
    ("diameter_mm", "Width (mm)", 0.1, 60.0, 1.0, 1),
    ("angle_deg", "Angle (deg)", -20.0, 20.0, 0.1, 1),
    ("vergence_dioptres", "Vergence (D)", -10.0, 10.0, 0.05, 2),
    ("offset_mm", "Offset (mm)", -30.0, 30.0, 0.5, 1),
    ("wavelength_nm", "Wavelength (nm)", 300.0, 2000.0, 5.0, 0),
]


def colour_swatch(colour: str) -> QIcon:
    """A small filled square of the given colour, for the field list."""
    pixmap = QPixmap(SWATCH_SIZE_PX, SWATCH_SIZE_PX)
    pixmap.fill(QColor(colour))
    return QIcon(pixmap)


class FieldEditor(QDialog):
    """List of fields with "+" and "-", and a form for the current field.

    Parameters
    ----------
    model : OpticalModel
        The bench whose fields are edited.
    parent : QWidget, optional
        The main window; the field window stays on top of it.

    Signals
    -------
    fields_changed
        A field was added, removed, edited or selected; the model is already updated.
    """

    fields_changed = pyqtSignal()

    def __init__(self, model: OpticalModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self.setWindowTitle("Fields")
        # A tool window floats above the main window without taking over the keyboard, so
        # the bench stays usable while it is open.
        self.setWindowFlag(Qt.WindowType.Tool)
        self.setModal(False)

        self.field_list = QListWidget()
        self.field_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.field_list.setFixedWidth(LIST_WIDTH_PX)
        self.add_button = QPushButton("+")
        self.remove_button = QPushButton("−")  # a minus sign, the width of the "+"
        self.add_button.setToolTip("Add a field: a copy of the selected one")
        self.remove_button.setToolTip("Remove the selected fields")

        self.value_boxes: dict[str, QDoubleSpinBox] = {}
        form = QFormLayout()
        for attribute, label, minimum, maximum, step, decimals in FIELD_PROPERTIES:
            box = QDoubleSpinBox()
            box.setRange(minimum, maximum)
            box.setSingleStep(step)
            box.setDecimals(decimals)
            # Bind the attribute name now; the lambda would otherwise see the loop's last one.
            box.valueChanged.connect(
                lambda value, attribute=attribute: self.set_field_value(attribute, value)
            )
            self.value_boxes[attribute] = box
            form.addRow(label, box)

        self.colour_button = QPushButton("Choose...")
        self.colour_button.setToolTip("Draw this beam in a colour of your choice")
        self.auto_colour_button = QPushButton("From wavelength")
        self.auto_colour_button.setToolTip("Draw this beam in the colour of its wavelength")
        colour_row = QHBoxLayout()
        colour_row.addWidget(self.colour_button)
        colour_row.addWidget(self.auto_colour_button)
        form.addRow("Colour", colour_row)

        self._lay_out(form)
        self.add_button.clicked.connect(self.add_field)
        self.remove_button.clicked.connect(self.remove_selected_fields)
        self.field_list.currentRowChanged.connect(self.on_current_row_changed)
        self.field_list.itemSelectionChanged.connect(self.update_remove_button)
        self.colour_button.clicked.connect(self.choose_colour)
        self.auto_colour_button.clicked.connect(self.use_wavelength_colour)
        self.refresh()

    def _lay_out(self, form: QFormLayout) -> None:
        """The list with its +/- buttons on the left, the form on the right."""
        plus_minus_row = QHBoxLayout()
        plus_minus_row.addWidget(self.add_button)
        plus_minus_row.addWidget(self.remove_button)
        list_column = QVBoxLayout()
        list_column.addWidget(self.field_list)
        list_column.addLayout(plus_minus_row)

        form_column = QVBoxLayout()
        form_column.addLayout(form)
        form_column.addStretch(1)

        layout = QHBoxLayout(self)
        layout.addLayout(list_column)
        layout.addLayout(form_column)

    # ------------------------------------------------------------------ showing the fields

    def refresh(self) -> None:
        """Rebuild the list from the model, select the model's selected field, fill the form."""
        fields = self.model.state.fields
        self.field_list.blockSignals(True)
        self.field_list.clear()
        for index, beam in enumerate(fields):
            item = QListWidgetItem(f"Field {index + 1}")
            item.setIcon(colour_swatch(beam.display_colour()))
            self.field_list.addItem(item)
        self.field_list.setCurrentRow(self.model.state.selected_field_index)
        self.field_list.blockSignals(False)
        self.update_remove_button()
        self.fill_form(self.model.state.selected_field)

    def fill_form(self, beam: BeamField) -> None:
        """Show a field's values, without these updates counting as edits."""
        for attribute, box in self.value_boxes.items():
            box.blockSignals(True)
            box.setValue(getattr(beam, attribute))
            box.blockSignals(False)
        self.auto_colour_button.setEnabled(beam.colour is not None)

    def update_remove_button(self) -> None:
        """ "-" works only when something is selected and at least one field would remain."""
        selected_count = len(self.field_list.selectedIndexes())
        field_count = len(self.model.state.fields)
        self.remove_button.setEnabled(0 < selected_count < field_count)

    def _update_current_swatch(self) -> None:
        """Recolour the current field's swatch (after a colour or wavelength change)."""
        item = self.field_list.item(self.model.state.selected_field_index)
        if item is not None:
            item.setIcon(colour_swatch(self.model.state.selected_field.display_colour()))

    # ------------------------------------------------------------------ user actions

    def on_current_row_changed(self, row: int) -> None:
        """Another field was picked in the list: it becomes the selected field."""
        if row < 0:
            return
        self.model.select_field(row)
        self.fill_form(self.model.state.selected_field)
        self.fields_changed.emit()

    def add_field(self) -> None:
        """Add a copy of the selected field and select it."""
        self.model.add_field()
        self.refresh()
        self.fields_changed.emit()

    def remove_selected_fields(self) -> None:
        """Remove every field selected in the list (one always remains)."""
        rows = [index.row() for index in self.field_list.selectedIndexes()]
        self.model.remove_fields(rows)
        self.refresh()
        self.fields_changed.emit()

    def set_field_value(self, attribute: str, value: float) -> None:
        """A value box changed: apply it to the selected field straight away."""
        setattr(self.model.state.selected_field, attribute, float(value))
        if attribute == "wavelength_nm":
            self._update_current_swatch()
        self.fields_changed.emit()

    def choose_colour(self) -> None:
        """Let the user pick the selected field's colour."""
        beam = self.model.state.selected_field
        colour = QColorDialog.getColor(QColor(beam.display_colour()), self, "Beam colour")
        if colour.isValid():
            self.set_colour(colour.name())

    def use_wavelength_colour(self) -> None:
        """Go back to drawing the selected field in the colour of its wavelength."""
        self.set_colour(None)

    def set_colour(self, colour: str | None) -> None:
        """Set the selected field's own colour (None for the colour of its wavelength)."""
        self.model.state.selected_field.colour = colour
        self.auto_colour_button.setEnabled(colour is not None)
        self._update_current_swatch()
        self.fields_changed.emit()
