"""The lens picker: browse and filter the catalogue, then pick one lens.

It is the "Add catalogue lens" dialog, and also the dialog that chooses a lens to replace
one on the bench (``prepare_for_replacement``). The lenses are shown as a tree: category ->
clear diameter -> lenses, positive lenses before negative ones and shortest focal length
first. The part-number search box above the tree and the filters below it (coating, focal
length range, hide positive / negative lenses) narrow the list. By default only lenses whose
coating suits the wavelengths of the beams are shown. The filtering itself is done by
``raycrafter.lenscatalog.selection``; this module only displays it.

The dialog is kept between uses, so the filters stay as the user left them.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListView,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from raycrafter.core.replacement import SAME_FOCAL_LENGTH_TOLERANCE
from raycrafter.gui.range_slider import RangeSlider
from raycrafter.lenscatalog.catalog import CatalogEntry
from raycrafter.lenscatalog.selection import (
    LensFilter,
    available_coatings,
    diameter_label,
    focal_length_span_mm,
    group_for_tree,
)

ALL_COATINGS = "All coatings"  # the coating drop-down's "no filter" option
SUITS_BEAMS = "suits beams"  # data of the drop-down's default: coatings that suit the beams
EXPAND_ALL_BELOW = 40  # with fewer lenses shown than this, every group is opened
SLIDER_STEPS = 1000  # resolution of the focal length slider
DIALOG_SIZE_PX = (680, 560)
COATING_COMBO_WIDTH_PX = 170
FILTER_COLUMN_GAP_PX = 18
PART_NUMBER_ROLE = Qt.ItemDataRole.UserRole  # where a lens row keeps its part number
GROUP_KEY_ROLE = Qt.ItemDataRole.UserRole + 1  # where a group row keeps its key (see below)


class LensPickerDialog(QDialog):
    """Choose one catalogue lens. ``exec()`` returns Accepted once Insert is pressed.

    Parameters
    ----------
    catalog : dict
        Part number -> CatalogEntry: the lenses to choose from.
    parent : QWidget, optional
        The window the dialog belongs to.
    action_text : str
        The text of the button that accepts the chosen lens, e.g. "Insert" or "Replace".
    """

    def __init__(
        self,
        catalog: dict[str, CatalogEntry],
        parent: QWidget | None = None,
        action_text: str = "Insert",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add catalogue lens")
        self.resize(*DIALOG_SIZE_PX)
        self.entries = list(catalog.values())
        self.action_text = action_text
        self.lens_filter = LensFilter()
        self.design_wavelengths_nm: tuple[float, ...] = ()  # the beams' (set_design_wavelengths)

        # The ends of the chosen focal length range, in mm. None means the handle is at that
        # end of the slider and follows the shortest (or longest) lens on the list, so that
        # changing another filter does not leave the range set narrower than the list.
        self.shortest_focal_length_mm: float | None = None
        self.longest_focal_length_mm: float | None = None
        self.slider_span_mm: tuple[float, float] | None = None  # |f| at the slider's two ends

        # Which groups are open, as "category" or "category|diameter" keys. Kept across
        # rebuilds of the tree, so filtering does not collapse what the user opened.
        self.expanded_group_keys: set[str] = set()
        # True while choosing a lens of the same focal length: the list is short, so every
        # group is opened and the user only has to scroll (see prepare_for_replacement).
        self.open_every_group = False

        self._build_widgets()
        self._connect_signals()
        self.refresh()

    # ------------------------------------------------------------------ setup

    def _build_widgets(self) -> None:
        """Search box, tree, a strip of filters, then the lens count and the buttons."""
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search part numbers, e.g. LA1 or AC254")
        self.search_box.setClearButtonEnabled(True)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Lens", "f (mm)", "Coating", "Type"])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setUniformRowHeights(True)  # much faster with many rows
        header = self.tree.header()
        for column in (0, 1, 2):  # part number, f and coating: as wide as their text
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)  # the product family takes the remaining width

        self.coating_combo = QComboBox()
        self.coating_combo.setView(QListView())  # see MainWindow: the native popup ignores CSS
        self.coating_combo.addItem("Suits the beams", SUITS_BEAMS)  # see set_design_wavelengths
        self.coating_combo.addItem(ALL_COATINGS, None)
        for coating in available_coatings(self.entries):
            self.coating_combo.addItem(coating, coating)

        self.coating_combo.setMinimumWidth(COATING_COMBO_WIDTH_PX)

        self.focal_length_label = QLabel()
        self.focal_length_label.setObjectName("sliderLabel")
        self.focal_length_slider = RangeSlider(0, SLIDER_STEPS)
        self.hide_positive_checkbox = QCheckBox("Hide positive lenses")
        self.hide_negative_checkbox = QCheckBox("Hide negative lenses")
        self.count_label = QLabel()
        self.count_label.setObjectName("hintLabel")

        # The filters sit in a strip under the tree: each filter is a column with its label
        # on the top row and its control below, so the strip reads as one tidy line. The
        # focal length slider takes the spare width.
        coating_label = QLabel("Coating")
        coating_label.setObjectName("sliderLabel")
        filters = QGridLayout()
        filters.setHorizontalSpacing(FILTER_COLUMN_GAP_PX)
        filters.setVerticalSpacing(4)
        filters.addWidget(coating_label, 0, 0)
        filters.addWidget(self.coating_combo, 1, 0)
        filters.addWidget(self.focal_length_label, 0, 1)
        filters.addWidget(self.focal_length_slider, 1, 1)
        filters.addWidget(self.hide_positive_checkbox, 0, 2)
        filters.addWidget(self.hide_negative_checkbox, 1, 2)
        filters.setColumnStretch(1, 1)

        self.buttons = QDialogButtonBox()
        self.insert_button = self.buttons.addButton(
            self.action_text, QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.buttons.addButton(QDialogButtonBox.StandardButton.Cancel)

        # The count of shown lenses shares the bottom row with the buttons.
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.count_label)
        bottom_row.addStretch(1)
        bottom_row.addWidget(self.buttons)

        layout = QVBoxLayout(self)
        layout.addWidget(self.search_box)
        layout.addWidget(self.tree, stretch=1)
        layout.addSpacing(6)
        layout.addLayout(filters)
        layout.addSpacing(6)
        layout.addLayout(bottom_row)

    def _connect_signals(self) -> None:
        """Every filter change rebuilds the tree; Insert and double-click accept the dialog."""
        self.search_box.textChanged.connect(self.refresh)
        self.coating_combo.currentIndexChanged.connect(self.refresh)
        self.hide_positive_checkbox.toggled.connect(self.refresh)
        self.hide_negative_checkbox.toggled.connect(self.refresh)
        self.focal_length_slider.values_changed.connect(self.on_focal_length_slider_moved)

        self.tree.itemSelectionChanged.connect(self.update_insert_button)
        self.tree.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.tree.itemExpanded.connect(lambda item: self.on_group_toggled(item, True))
        self.tree.itemCollapsed.connect(lambda item: self.on_group_toggled(item, False))

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    # ------------------------------------------------------------------ setting up a use

    def set_design_wavelengths(self, wavelengths_nm: list[float]) -> None:
        """Tell the dialog the wavelengths of the beams, for its "suits the beams" filter."""
        self.design_wavelengths_nm = tuple(sorted(set(wavelengths_nm)))
        listed = ", ".join(f"{wavelength:g}" for wavelength in self.design_wavelengths_nm)
        self.coating_combo.setItemText(0, f"Suits the beams ({listed} nm)")
        self.refresh()

    def prepare_for_replacement(self, part_number: str, focal_length_mm: float | None) -> None:
        """Set the dialog up to choose a lens to replace ``part_number`` with.

        With a focal length, only lenses of the same sign within
        ``replacement.SAME_FOCAL_LENGTH_TOLERANCE`` of it are shown ("Replace with same f");
        with None, the focal length and sign filters are cleared ("Swap lens"). The search
        text is cleared too, and the coating filter left as the user last set it.

        With a focal length, every group of the tree is opened, so the user only has to
        scroll to choose; with None (the whole catalogue) they start closed.
        """
        self.setWindowTitle(f"Replace {part_number}")
        self.expanded_group_keys.clear()
        self.search_box.blockSignals(True)  # one refresh at the end, not one per change
        self.hide_positive_checkbox.blockSignals(True)
        self.hide_negative_checkbox.blockSignals(True)
        self.search_box.clear()
        if focal_length_mm is None:
            self.shortest_focal_length_mm = self.longest_focal_length_mm = None
            self.hide_positive_checkbox.setChecked(False)
            self.hide_negative_checkbox.setChecked(False)
        else:
            magnitude_mm = abs(focal_length_mm)
            self.shortest_focal_length_mm = magnitude_mm * (1.0 - SAME_FOCAL_LENGTH_TOLERANCE)
            self.longest_focal_length_mm = magnitude_mm * (1.0 + SAME_FOCAL_LENGTH_TOLERANCE)
            self.hide_positive_checkbox.setChecked(focal_length_mm < 0)
            self.hide_negative_checkbox.setChecked(focal_length_mm > 0)
        self.search_box.blockSignals(False)
        self.hide_positive_checkbox.blockSignals(False)
        self.hide_negative_checkbox.blockSignals(False)
        self.open_every_group = focal_length_mm is not None
        self.refresh()

    # ------------------------------------------------------------------ results

    def selected_part_number(self) -> str | None:
        """The part number of the selected lens, or None if no lens row is selected."""
        selected = self.tree.selectedItems()
        if not selected:
            return None
        return selected[0].data(0, PART_NUMBER_ROLE)

    def select_part_number(self, part_number: str) -> bool:
        """Select (and scroll to) a lens row, opening its groups. False if it is not shown."""
        for item in self._lens_items():
            if item.data(0, PART_NUMBER_ROLE) == part_number:
                parent = item.parent()
                while parent is not None:
                    parent.setExpanded(True)
                    parent = parent.parent()
                self.tree.setCurrentItem(item)
                self.tree.scrollToItem(item)
                return True
        return False

    # ------------------------------------------------------------------ filtering

    def refresh(self) -> None:
        """Read the filter widgets, update the focal length slider and rebuild the tree."""
        coating_choice = self.coating_combo.currentData()
        suits_beams = coating_choice == SUITS_BEAMS
        self.lens_filter.coating = None if suits_beams else coating_choice
        self.lens_filter.wavelengths_nm = self.design_wavelengths_nm if suits_beams else None
        self.lens_filter.part_number_text = self.search_box.text()
        self.lens_filter.hide_positive = self.hide_positive_checkbox.isChecked()
        self.lens_filter.hide_negative = self.hide_negative_checkbox.isChecked()

        # The slider spans the focal lengths of the lenses that pass the other filters.
        eligible = [
            entry for entry in self.entries if self.lens_filter.matches_ignoring_focal_length(entry)
        ]
        self.slider_span_mm = focal_length_span_mm(eligible)
        self.lens_filter.focal_length_range_mm = self.chosen_focal_length_range_mm()
        self._place_slider_handles()
        self._update_focal_length_label()

        self.rebuild_tree(self.lens_filter.apply(eligible))

    def chosen_focal_length_range_mm(self) -> tuple[float, float] | None:
        """The |f| range to filter on: the user's choice, with free ends at the slider's ends."""
        if self.slider_span_mm is None:
            return None
        span_shortest, span_longest = self.slider_span_mm
        shortest = self.shortest_focal_length_mm
        longest = self.longest_focal_length_mm
        return (
            span_shortest if shortest is None else max(shortest, span_shortest),
            span_longest if longest is None else min(longest, span_longest),
        )

    def on_focal_length_slider_moved(self, low_step: int, high_step: int) -> None:
        """Store the new range in mm (an end at the slider's end follows the list), and refresh."""
        span_mm = self.slider_span_mm
        if span_mm is None:
            return  # the slider is disabled when no lens passes the other filters
        self.shortest_focal_length_mm = (
            None if low_step == 0 else focal_length_at_step(low_step, span_mm)
        )
        self.longest_focal_length_mm = (
            None if high_step == SLIDER_STEPS else focal_length_at_step(high_step, span_mm)
        )
        self.refresh()

    def _place_slider_handles(self) -> None:
        """Put the handles where the chosen range is on the current slider span."""
        span_mm = self.slider_span_mm
        self.focal_length_slider.setEnabled(span_mm is not None)
        if span_mm is None:
            return
        shortest, longest = self.shortest_focal_length_mm, self.longest_focal_length_mm
        low_step = 0 if shortest is None else step_of_focal_length(shortest, span_mm)
        high_step = SLIDER_STEPS if longest is None else step_of_focal_length(longest, span_mm)
        self.focal_length_slider.set_values(low_step, high_step)

    def _update_focal_length_label(self) -> None:
        """Show the chosen |f| range above the slider."""
        chosen_range = self.lens_filter.focal_length_range_mm
        if chosen_range is None:
            self.focal_length_label.setText("Focal length |f|: no lenses")
            return
        shortest, longest = chosen_range
        self.focal_length_label.setText(
            f"Focal length |f|: {format_focal_length(shortest)} to "
            f"{format_focal_length(longest)} mm"
        )

    # ------------------------------------------------------------------ the tree

    def rebuild_tree(self, shown_entries: list[CatalogEntry]) -> None:
        """Replace the tree's rows with the given lenses, keeping open groups and selection.

        While a search text is typed, while choosing a lens of the same focal length, or when
        only a few lenses are shown, every group is opened, so the matches are visible.
        """
        previously_selected = self.selected_part_number()
        expand_everything = (
            self.open_every_group
            or bool(self.lens_filter.part_number_text.strip())
            or len(shown_entries) < EXPAND_ALL_BELOW
        )

        self.tree.blockSignals(True)  # no expand/collapse or selection signals while rebuilding
        self.tree.setUpdatesEnabled(False)
        self.tree.clear()
        for category, by_diameter in group_for_tree(shown_entries).items():
            category_count = sum(len(lenses) for lenses in by_diameter.values())
            category_item = self._add_group_item(
                self.tree.invisibleRootItem(), f"{category}  ({category_count})", category
            )
            for diameter_mm, lenses in by_diameter.items():
                diameter_item = self._add_group_item(
                    category_item,
                    f"{diameter_label(diameter_mm)}  ({len(lenses)})",
                    f"{category}|{diameter_mm:g}",
                )
                for entry in lenses:
                    self._add_lens_item(diameter_item, entry)

        for item in self._group_items():
            key = item.data(0, GROUP_KEY_ROLE)
            item.setExpanded(expand_everything or key in self.expanded_group_keys)
        self.tree.setUpdatesEnabled(True)
        self.tree.blockSignals(False)

        if previously_selected is not None:
            self.select_part_number(previously_selected)
        self.count_label.setText(f"{len(shown_entries)} of {len(self.entries)} lenses shown")
        self.update_insert_button()

    @staticmethod
    def _add_group_item(parent: QTreeWidgetItem, text: str, key: str) -> QTreeWidgetItem:
        """A category or diameter row, which opens and closes but cannot be selected."""
        item = QTreeWidgetItem(parent, [text])
        item.setData(0, GROUP_KEY_ROLE, key)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)  # groups open and close but cannot be chosen
        item.setFirstColumnSpanned(True)
        return item

    @staticmethod
    def _add_lens_item(parent: QTreeWidgetItem, entry: CatalogEntry) -> None:
        """One lens row: part number, focal length, coating and product family."""
        item = QTreeWidgetItem(
            parent,
            [entry.part_number, f"{entry.focal_length:g}", entry.coating, entry.family],
        )
        item.setData(0, PART_NUMBER_ROLE, entry.part_number)
        item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def _all_items(self) -> list[QTreeWidgetItem]:
        """Every row of the tree, depth first."""
        items = []
        pending = [self.tree.topLevelItem(index) for index in range(self.tree.topLevelItemCount())]
        while pending:
            item = pending.pop(0)
            items.append(item)
            pending[0:0] = [item.child(index) for index in range(item.childCount())]
        return items

    def _group_items(self) -> list[QTreeWidgetItem]:
        return [item for item in self._all_items() if item.data(0, GROUP_KEY_ROLE) is not None]

    def _lens_items(self) -> list[QTreeWidgetItem]:
        return [item for item in self._all_items() if item.data(0, PART_NUMBER_ROLE) is not None]

    def on_group_toggled(self, item: QTreeWidgetItem, expanded: bool) -> None:
        """Remember which groups the user opened, so a rebuild can open them again."""
        key = item.data(0, GROUP_KEY_ROLE)
        if key is None:
            return
        if expanded:
            self.expanded_group_keys.add(key)
        else:
            self.expanded_group_keys.discard(key)

    def update_insert_button(self) -> None:
        """Insert is available only while a lens (not a group) is selected."""
        self.insert_button.setEnabled(self.selected_part_number() is not None)

    def on_item_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        """Double-clicking a lens is the same as selecting it and pressing Insert."""
        if item.data(0, PART_NUMBER_ROLE) is not None:
            self.accept()


def focal_length_at_step(step: int, span_mm: tuple[float, float]) -> float:
    """|f| in mm at a step of the focal length slider, whose ends are at ``span_mm``.

    The scale is logarithmic: 5 to 50 mm gets the same width as 50 to 500 mm, which suits
    catalogues running from a few mm to a metre.
    """
    log_shortest, log_longest = math.log10(span_mm[0]), math.log10(span_mm[1])
    return 10 ** (log_shortest + (log_longest - log_shortest) * step / SLIDER_STEPS)


def step_of_focal_length(focal_length_mm: float, span_mm: tuple[float, float]) -> int:
    """The slider step nearest a |f| in mm (the inverse of ``focal_length_at_step``)."""
    log_shortest, log_longest = math.log10(span_mm[0]), math.log10(span_mm[1])
    if log_longest == log_shortest:
        return 0
    fraction = (math.log10(focal_length_mm) - log_shortest) / (log_longest - log_shortest)
    return round(min(max(fraction, 0.0), 1.0) * SLIDER_STEPS)


def format_focal_length(focal_length_mm: float) -> str:
    """A focal length with sensible precision for the slider label: 4.0, 35, 1000."""
    return f"{focal_length_mm:.1f}" if focal_length_mm < 10 else f"{focal_length_mm:.0f}"
