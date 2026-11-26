"""The bench viewer: a list of everything on the bench, numbered as the Python API numbers it.

Scripts and AI agents refer to lenses and fields by index (``bench.lenses[0]``, ``lens_index=0``),
so the user needs a way to see which number is which. This small, non-modal window lists the
lenses (index, part number, where they sit on the axis, and details of the part such as its
vendor and surface numbers), the fields and the image plane, and keeps the positions up to date
while lenses are dragged. Clicking a row highlights that item in the side view. The selected
lens or fields can be deleted from here; nothing else is edited.

The window shows the model; it does not change it. Highlighting and deleting are requested
with signals, which the main window carries out.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from raycrafter.core.model import OpticalModel
from raycrafter.core.plot_geometry import field_name
from raycrafter.core.state import OpticalState
from raycrafter.gui.field_editor import colour_swatch

COLUMNS = ["Index", "Item", "z (mm)", "Details"]
LENS, FIELD, IMAGE = "lens", "field", "image"  # the kinds of row
ROW_KIND_ROLE = Qt.ItemDataRole.UserRole  # where each row keeps (kind, index)
WINDOW_SIZE_PX = (720, 360)


def part_details(state: OpticalState, lens_index: int) -> list[str]:
    """The lens as a part: vendor and mount (if known), clear aperture and surface numbers."""
    lens = state.lenses[lens_index]
    first, last = state.surface_range(lens_index)
    surfaces = f"surface {first}" if first == last else f"surfaces {first}–{last}"
    details = [lens.vendor, lens.mount_type, f"Ø{2.0 * lens.semi_aperture:g} mm", surfaces]
    return [detail for detail in details if detail is not None]


def lens_rows(state: OpticalState) -> list[tuple[str, int, list[str]]]:
    """One (kind, index, cell texts) per lens, for the lens section of the list."""
    rows = []
    for index, lens in enumerate(state.lenses):
        details = part_details(state, index)
        if lens.flipped:
            details.append("flipped")
        if lens.assembly_id is not None:
            details.append(f"in {state.assembly_names[lens.assembly_id]} ({lens.assembly_id})")
        if lens.is_paraxial:
            details.append("ideal thin lens")
        if lens.locked_to_next:
            details.append("gap behind locked")
        elif index == len(state.lenses) - 1 and state.image_gap_locked:
            details.append("gap to image locked")
        position = f"{lens.z:.1f}" if lens.is_paraxial else f"{lens.z:.1f} – {lens.back_z:.1f}"
        rows.append((LENS, index, [str(index), lens.part_number, position, ", ".join(details)]))
    return rows


def field_rows(state: OpticalState) -> list[tuple[str, int, list[str]]]:
    """One (kind, index, cell texts) per field, for the field section of the list."""
    rows = []
    for index, beam in enumerate(state.fields):
        details = (
            f"{beam.diameter_mm:g} mm wide, {beam.angle_deg:g}°, {beam.wavelength_nm:g} nm"
            + (", selected" if index == state.selected_field_index else "")
        )
        rows.append((FIELD, index, [str(index), field_name(index), "", details]))
    return rows


class BenchViewer(QDialog):
    """A read-only list of the lenses, fields and image plane, with their API indices.

    Parameters
    ----------
    model : OpticalModel
        The bench to list.
    parent : QWidget, optional
        The main window; the viewer stays on top of it.

    Signals
    -------
    highlight_requested(object, object)
        The user selected a row: the lens index and field index to highlight (either may be
        None). Both are None when nothing is selected.
    delete_lens_requested(int)
        Delete this lens.
    delete_fields_requested(list)
        Delete these fields.
    """

    highlight_requested = pyqtSignal(object, object)
    delete_lens_requested = pyqtSignal(int)
    delete_fields_requested = pyqtSignal(list)

    def __init__(self, model: OpticalModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self.setWindowTitle("Bench")
        # A tool window floats above the main window without taking over the keyboard, so
        # the bench stays usable while it is open (as for the field window).
        self.setWindowFlag(Qt.WindowType.Tool)
        self.setModal(False)
        self.resize(*WINDOW_SIZE_PX)

        explanation = QLabel(
            "Indices are those used by the Python API and AI agents, "
            "e.g. bench.lenses[0] or lens_index=0."
        )
        explanation.setWordWrap(True)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(COLUMNS)
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lens_section = QTreeWidgetItem(["", "Lenses"])
        self.field_section = QTreeWidgetItem(["", "Fields"])
        self.image_row = QTreeWidgetItem(["", "Image plane"])
        self.image_row.setData(0, ROW_KIND_ROLE, (IMAGE, 0))
        for section in (self.lens_section, self.field_section):
            section.setFlags(Qt.ItemFlag.ItemIsEnabled)  # a heading: not selectable
            font = section.font(1)
            font.setBold(True)
            section.setFont(1, font)
        self.tree.addTopLevelItems([self.lens_section, self.field_section, self.image_row])
        self.lens_section.setExpanded(True)
        self.field_section.setExpanded(True)

        self.delete_button = QPushButton("Delete")
        self.delete_button.setToolTip("Delete the selected lens or fields (Del)")
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.delete_button)

        layout = QVBoxLayout(self)
        layout.addWidget(explanation)
        layout.addWidget(self.tree)
        layout.addLayout(buttons)

        self.tree.itemSelectionChanged.connect(self.on_selection_changed)
        self.delete_button.clicked.connect(self.delete_selected)

    # ------------------------------------------------------------------ showing the bench

    def refresh(
        self, highlighted_lens_index: int | None = None, highlighted_field_index: int | None = None
    ) -> None:
        """Show the bench as it is now, with the highlighted lens or field selected.

        Called after every change while the window is open, including during a drag, so it
        only changes the text of existing rows unless lenses or fields were added or removed.
        """
        state = self.model.state
        self.tree.blockSignals(True)  # the selection is restored, not changed by the user
        lenses_changed = self._fill_section(self.lens_section, lens_rows(state))
        fields_changed = self._fill_section(self.field_section, field_rows(state))
        self.image_row.setText(2, f"{state.image_z:.1f}")
        if lenses_changed or fields_changed:
            for column in range(len(COLUMNS)):
                self.tree.resizeColumnToContents(column)

        for row in self._rows():
            kind, index = row.data(0, ROW_KIND_ROLE)
            should_be_selected = (kind == LENS and index == highlighted_lens_index) or (
                kind == FIELD and index == highlighted_field_index
            )
            if row.isSelected() != should_be_selected:
                row.setSelected(should_be_selected)
        self.tree.blockSignals(False)
        self.update_delete_button()

    def _fill_section(self, section: QTreeWidgetItem, rows: list) -> bool:
        """Make the section's rows match ``rows``, reusing the existing row items.

        Returns True if rows were added or removed.
        """
        rows_were_added_or_removed = section.childCount() != len(rows)
        while section.childCount() > len(rows):
            section.removeChild(section.child(section.childCount() - 1))
        while section.childCount() < len(rows):
            section.addChild(QTreeWidgetItem())
        for position, (kind, index, texts) in enumerate(rows):
            row = section.child(position)
            row.setData(0, ROW_KIND_ROLE, (kind, index))
            for column, text in enumerate(texts):
                if row.text(column) != text:  # most rows are unchanged on most updates, and
                    # setting a text, even to the same value, makes the list repaint
                    row.setText(column, text)
            if kind == FIELD:
                colour = self.model.state.fields[index].display_colour()
                if row.data(1, ROW_KIND_ROLE) != colour:  # remember it, to redraw only on change
                    row.setData(1, ROW_KIND_ROLE, colour)
                    row.setIcon(1, colour_swatch(colour))
        return rows_were_added_or_removed

    def _rows(self) -> list[QTreeWidgetItem]:
        """Every lens and field row (not the section headings), then the image plane."""
        rows = []
        for section in (self.lens_section, self.field_section):
            rows += [section.child(position) for position in range(section.childCount())]
        return rows + [self.image_row]

    def selected(self, kind: str) -> list[int]:
        """Indices of the selected rows of one kind (LENS or FIELD), in order."""
        return sorted(
            row.data(0, ROW_KIND_ROLE)[1]
            for row in self.tree.selectedItems()
            if row.data(0, ROW_KIND_ROLE) is not None and row.data(0, ROW_KIND_ROLE)[0] == kind
        )

    # ------------------------------------------------------------------ user actions

    def on_selection_changed(self) -> None:
        """Ask for the item the user just selected to be highlighted."""
        lens_indices, field_indices = self.selected(LENS), self.selected(FIELD)
        current = self.tree.currentItem()
        current_kind = None if current is None else current.data(0, ROW_KIND_ROLE)
        # With several rows selected, the one clicked last is highlighted.
        if current_kind is not None and current_kind[0] in (LENS, FIELD) and current.isSelected():
            kind, index = current_kind
            lens_index = index if kind == LENS else None
            field_index = index if kind == FIELD else None
        else:
            lens_index = lens_indices[0] if lens_indices else None
            field_index = field_indices[0] if field_indices else None
        self.highlight_requested.emit(lens_index, field_index)
        self.update_delete_button()

    def update_delete_button(self) -> None:
        """Delete is possible for one lens, or for fields as long as one field would remain."""
        lens_indices, field_indices = self.selected(LENS), self.selected(FIELD)
        can_delete_lens = len(lens_indices) == 1 and not field_indices
        can_delete_fields = (
            bool(field_indices)
            and not lens_indices
            and len(field_indices) < len(self.model.state.fields)
        )
        self.delete_button.setEnabled(can_delete_lens or can_delete_fields)

    def delete_selected(self) -> None:
        """Ask for the selected lens or fields to be deleted."""
        if not self.delete_button.isEnabled():
            return
        lens_indices, field_indices = self.selected(LENS), self.selected(FIELD)
        if lens_indices:
            self.delete_lens_requested.emit(lens_indices[0])
        else:
            self.delete_fields_requested.emit(field_indices)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Delete or Backspace (delete on Mac keyboards) deletes the selection."""
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(event)

    def hideEvent(self, event) -> None:
        """Closing the window removes the highlight from the side view."""
        super().hideEvent(event)
        self.highlight_requested.emit(None, None)
