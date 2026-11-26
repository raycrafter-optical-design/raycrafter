"""The lens browser: find a catalogue lens and pick it, inside the main window.

It has two parts. The browser itself (``LensBrowser``: a part-number search box and a tree
of lenses) takes the place of the bottom-right plots while the "Add catalogue lens" tool is
active. Its filters (``LensFilterControls``) are shown in the context bar: show or hide the
positive and negative lenses, the |f| and clear-diameter ranges, the coating, and the
catalogue. Both parts stay in the window, so the user can filter while watching the list.

The tree is category -> clear diameter -> lenses, positive lenses before negative ones and
shortest focal length first. By default only lenses whose coating suits the wavelengths of
the beams are shown. The filtering itself is done by ``raycrafter.lenscatalog.selection``;
this module only displays it.

Choosing a lens emits ``lens_selected``, and the tool makes its ghost follow the pointer.
The browser is also used to replace a lens on the bench (``start_replacement``): it then
shows Replace and Cancel buttons, and puts the user's filters back afterwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListView,
    QPushButton,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from raycrafter.core.replacement import SAME_FOCAL_LENGTH_TOLERANCE
from raycrafter.gui.range_slider import RangeSlider
from raycrafter.gui.style import CONTEXT_BAR_BOX_HEIGHT_PX
from raycrafter.gui.tool_icons import ITALIC_F, lens_sign_icon
from raycrafter.lenscatalog.catalog import CatalogEntry
from raycrafter.lenscatalog.selection import (
    LensFilter,
    available_coatings,
    diameter_label,
    diameter_span_mm,
    focal_length_span_mm,
    group_for_tree,
)

ALL_COATINGS = "All coatings"  # the coating drop-down's "no filter" option
SUITS_BEAMS = "suits beams"  # data of the drop-down's default: coatings that suit the beams
CATALOGUE_NAME = "Thorlabs"  # the only catalogue so far
EXPAND_ALL_BELOW = 40  # with fewer lenses shown than this, every group is opened
SLIDER_STEPS = 1000  # resolution of the range sliders
RANGE_SLIDER_WIDTH_PX = 110
RANGE_SLIDER_HANDLE_PX = 5  # small handles, like the context bar's other sliders
RANGE_TEXT_WIDTH_PX = 84  # room for "4.0-2500 mm"
COATING_COMBO_WIDTH_PX = 190  # room for "Suits the beams (555 nm)"
SIGN_ICON_SIZE = QSize(14, 14)  # the + and - in their 22 px buttons
SEARCH_TO_TREE_PX = 6
TREE_TO_COUNT_PX = 11
PART_NUMBER_ROLE = Qt.ItemDataRole.UserRole  # where a lens row keeps its part number
GROUP_KEY_ROLE = Qt.ItemDataRole.UserRole + 1  # where a group row keeps its key (see below)


# ---------------------------------------------------------------------- range sliders


def value_at_step(step: int, span: tuple[float, float]) -> float:
    """The value at a step of a range slider whose ends are at ``span``.

    The scale is logarithmic: 5 to 50 mm gets the same width as 50 to 500 mm, which suits
    focal lengths from a few mm to a metre and diameters from a few mm to a few inches.
    """
    log_low, log_high = math.log10(span[0]), math.log10(span[1])
    return 10 ** (log_low + (log_high - log_low) * step / SLIDER_STEPS)


def step_of_value(value: float, span: tuple[float, float]) -> int:
    """The slider step nearest a value (the inverse of ``value_at_step``)."""
    log_low, log_high = math.log10(span[0]), math.log10(span[1])
    if log_high == log_low:
        return 0
    fraction = (math.log10(value) - log_low) / (log_high - log_low)
    return round(min(max(fraction, 0.0), 1.0) * SLIDER_STEPS)


def format_mm(value_mm: float) -> str:
    """A length with sensible precision for a range label: 4.0, 35, 1000."""
    return f"{value_mm:.1f}" if value_mm < 10 else f"{value_mm:.0f}"


@dataclass
class ChosenRange:
    """The ends of a range the user chose on a slider, in mm.

    An end of None is at the end of the slider: it follows the shortest (or longest) lens
    on the list, so that changing another filter never leaves the range narrower than the
    list. ``span`` is the values at the slider's two ends, set from the lenses that pass the
    other filters (None when there are none).
    """

    low: float | None = None
    high: float | None = None
    span: tuple[float, float] | None = None

    def range_mm(self) -> tuple[float, float] | None:
        """The range to filter on: the user's choice, with free ends at the slider's ends."""
        if self.span is None:
            return None
        return (
            self.span[0] if self.low is None else max(self.low, self.span[0]),
            self.span[1] if self.high is None else min(self.high, self.span[1]),
        )

    def set_from_steps(self, low_step: int, high_step: int) -> None:
        """Take a slider position; an end at the slider's end becomes free."""
        if self.span is None:
            return
        self.low = None if low_step == 0 else value_at_step(low_step, self.span)
        self.high = None if high_step == SLIDER_STEPS else value_at_step(high_step, self.span)

    def steps(self) -> tuple[int, int]:
        """Where the slider's handles go for the chosen range."""
        if self.span is None:
            return 0, SLIDER_STEPS
        low_step = 0 if self.low is None else step_of_value(self.low, self.span)
        high_step = SLIDER_STEPS if self.high is None else step_of_value(self.high, self.span)
        return low_step, high_step

    def text(self) -> str:
        """The range for the label beside the slider, e.g. "4-1000 mm"."""
        chosen = self.range_mm()
        if chosen is None:
            return "none"
        return f"{format_mm(chosen[0])}–{format_mm(chosen[1])} mm"


# ---------------------------------------------------------------------- the filters


class LensFilterControls(QWidget):
    """The lens filters, laid out in a row for the context bar.

    Signals
    -------
    changed
        The user changed a filter.
    """

    changed = pyqtSignal()

    def __init__(self, coatings: list[str]) -> None:
        super().__init__()
        # Latched buttons: pressed (checked) means those lenses are hidden; they are then
        # drawn faded grey (see the style sheet).
        self.hide_positive_button = QToolButton()
        self.hide_positive_button.setIcon(lens_sign_icon(positive=True))
        self.hide_positive_button.setToolTip("Show or hide positive (converging) lenses")
        self.hide_negative_button = QToolButton()
        self.hide_negative_button.setIcon(lens_sign_icon(positive=False))
        self.hide_negative_button.setToolTip("Show or hide negative (diverging) lenses")
        for button in (self.hide_positive_button, self.hide_negative_button):
            button.setObjectName("lensSignButton")
            button.setCheckable(True)
            button.setIconSize(SIGN_ICON_SIZE)

        self.focal_length_slider = RangeSlider(0, SLIDER_STEPS, RANGE_SLIDER_HANDLE_PX)
        self.focal_length_text = QLabel()
        self.diameter_slider = RangeSlider(0, SLIDER_STEPS, RANGE_SLIDER_HANDLE_PX)
        self.diameter_text = QLabel()
        for slider in (self.focal_length_slider, self.diameter_slider):
            slider.setFixedWidth(RANGE_SLIDER_WIDTH_PX)
        for text in (self.focal_length_text, self.diameter_text):
            text.setFixedWidth(RANGE_TEXT_WIDTH_PX)
        self.focal_length_slider.setToolTip(
            "Range of focal length shown (its size: positive and negative lenses alike)"
        )
        self.diameter_slider.setToolTip("Range of clear diameter shown")

        self.coating_combo = QComboBox()
        # A plain list view: the native macOS popup ignores the style sheet (see FieldEditor).
        self.coating_combo.setView(QListView())
        self.coating_combo.addItem("Suits the beams", SUITS_BEAMS)  # see set_design_wavelengths
        self.coating_combo.addItem(ALL_COATINGS, None)
        for coating in coatings:
            self.coating_combo.addItem(coating, coating)
        self.coating_combo.setFixedWidth(COATING_COMBO_WIDTH_PX)
        self.coating_combo.setToolTip("Coating")

        self.catalogue_combo = QComboBox()
        self.catalogue_combo.addItem(CATALOGUE_NAME)
        self.catalogue_combo.setEnabled(False)  # one catalogue so far
        self.catalogue_combo.setToolTip("Catalogue (only Thorlabs so far)")

        focal_length_label = QLabel(ITALIC_F)
        focal_length_label.setObjectName("italicF")
        diameter_label_widget = QLabel("⌀")
        diameter_label_widget.setObjectName("diameterSign")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.hide_positive_button)
        layout.addWidget(self.hide_negative_button)
        layout.addSpacing(10)
        layout.addWidget(focal_length_label)
        layout.addWidget(self.focal_length_slider)
        layout.addWidget(self.focal_length_text)
        layout.addSpacing(4)
        layout.addWidget(diameter_label_widget)
        layout.addWidget(self.diameter_slider)
        layout.addWidget(self.diameter_text)
        layout.addSpacing(4)
        layout.addWidget(self.coating_combo)
        layout.addWidget(self.catalogue_combo)
        layout.addStretch(1)

        for combo in (self.coating_combo, self.catalogue_combo):
            combo.setFixedHeight(CONTEXT_BAR_BOX_HEIGHT_PX)  # as on the bar's other pages

        self.hide_positive_button.toggled.connect(self.changed.emit)
        self.hide_negative_button.toggled.connect(self.changed.emit)
        self.coating_combo.currentIndexChanged.connect(self.changed.emit)


# ---------------------------------------------------------------------- the browser


class LensBrowser(QWidget):
    """A search box over a tree of catalogue lenses, filtered by ``filter_controls``.

    Parameters
    ----------
    catalog : dict
        Part number -> CatalogEntry: the lenses to choose from.

    Attributes
    ----------
    filter_controls : LensFilterControls
        The filters, for the main window to put in the context bar.

    Signals
    -------
    lens_selected(object)
        Outside replacing: the user chose a lens (its part number), or None.
    replace_confirmed(str)
        Replacing: the user pressed Replace (or double-clicked a lens).
    replace_cancelled
        Replacing: the user pressed Cancel.
    """

    lens_selected = pyqtSignal(object)
    replace_confirmed = pyqtSignal(str)
    replace_cancelled = pyqtSignal()

    def __init__(self, catalog: dict[str, CatalogEntry]) -> None:
        super().__init__()
        self.entries = list(catalog.values())
        self.lens_filter = LensFilter()
        self.design_wavelengths_nm: tuple[float, ...] = ()  # the beams' (set_design_wavelengths)
        self.focal_length_range = ChosenRange()
        self.diameter_range = ChosenRange()
        # Which groups are open, as "category" or "category|diameter" keys. Kept across
        # rebuilds of the tree, so filtering does not collapse what the user opened.
        self.expanded_group_keys: set[str] = set()
        # True while choosing a lens of the same focal length: the list is short, so every
        # group is opened and the user only has to scroll (see start_replacement).
        self.open_every_group = False
        self.replacing = False
        self.filters_before_replacing: tuple | None = None  # put back by end_replacement

        self.filter_controls = LensFilterControls(available_coatings(self.entries))
        self._build_widgets()
        self._connect_signals()
        self.refresh()

    # ------------------------------------------------------------------ setup

    def _build_widgets(self) -> None:
        """Search box, tree, then the lens count with (while replacing) the buttons."""
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search part numbers, e.g. LA1 or AC254")
        self.search_box.setClearButtonEnabled(True)
        self.expand_button = QPushButton("Expand all")
        self.expand_button.setToolTip("Open or close every group of the list")
        search_row = QHBoxLayout()
        search_row.addWidget(self.search_box, stretch=1)
        search_row.addWidget(self.expand_button)

        self.tree = QTreeWidget()
        self.tree.setObjectName("lensTree")  # its text is a little smaller (see the style sheet)
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Lens", "f (mm)", "Type"])
        # The coating is not shown: it is in the part number's suffix (e.g. -A).
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setUniformRowHeights(True)  # much faster with many rows
        header = self.tree.header()
        for column in (0, 1):  # part number and f: as wide as their text
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)  # the product family takes the remaining width

        self.count_label = QLabel()
        self.count_label.setObjectName("hintLabel")
        self.replace_button = QPushButton("Replace")
        self.cancel_button = QPushButton("Cancel")
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.count_label)
        bottom_row.addStretch(1)
        bottom_row.addWidget(self.cancel_button)
        bottom_row.addWidget(self.replace_button)
        self._show_replace_buttons(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)  # the gaps are set one by one
        layout.addLayout(search_row)
        layout.addSpacing(SEARCH_TO_TREE_PX)
        layout.addWidget(self.tree, stretch=1)
        layout.addSpacing(TREE_TO_COUNT_PX)
        layout.addLayout(bottom_row)

    def _connect_signals(self) -> None:
        """Every filter change rebuilds the tree; choosing a lens is reported."""
        controls = self.filter_controls
        self.search_box.textChanged.connect(self.refresh)
        controls.changed.connect(self.refresh)
        controls.focal_length_slider.values_changed.connect(
            lambda low, high: self._on_range_moved(self.focal_length_range, low, high)
        )
        controls.diameter_slider.values_changed.connect(
            lambda low, high: self._on_range_moved(self.diameter_range, low, high)
        )

        self.tree.itemSelectionChanged.connect(self.on_selection_changed)
        self.tree.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.tree.itemExpanded.connect(lambda item: self.on_group_toggled(item, True))
        self.tree.itemCollapsed.connect(lambda item: self.on_group_toggled(item, False))
        self.expand_button.clicked.connect(self.toggle_expand_all)
        self.replace_button.clicked.connect(self.confirm_replacement)
        self.cancel_button.clicked.connect(self.replace_cancelled.emit)

    # ------------------------------------------------------------------ setting up a use

    def set_design_wavelengths(self, wavelengths_nm: list[float]) -> None:
        """Tell the browser the wavelengths of the beams, for its "suits the beams" filter."""
        self.design_wavelengths_nm = tuple(sorted(set(wavelengths_nm)))
        listed = ", ".join(f"{wavelength:g}" for wavelength in self.design_wavelengths_nm)
        self.filter_controls.coating_combo.setItemText(0, f"Suits the beams ({listed} nm)")
        self.refresh()

    def start_replacement(self, part_number: str, focal_length_mm: float | None) -> None:
        """Show the lenses that could replace ``part_number``, with Replace and Cancel.

        With a focal length, only lenses of the same sign within
        ``replacement.SAME_FOCAL_LENGTH_TOLERANCE`` of it are shown ("Replace with same f"),
        and every group of the tree is opened, so the user only has to scroll. With None
        ("Swap lens"), the focal length and sign filters are cleared. The search text is
        cleared too, and the coating filter left as the user set it. ``end_replacement``
        puts the filters back as they were.
        """
        controls = self.filter_controls
        self.filters_before_replacing = (
            self.search_box.text(),
            controls.hide_positive_button.isChecked(),
            controls.hide_negative_button.isChecked(),
            (self.focal_length_range.low, self.focal_length_range.high),
        )
        self.replacing = True
        self.expanded_group_keys.clear()
        if focal_length_mm is None:
            self._set_filters("", False, False, (None, None))
        else:
            magnitude_mm = abs(focal_length_mm)
            self._set_filters(
                "",
                focal_length_mm < 0,
                focal_length_mm > 0,
                (
                    magnitude_mm * (1.0 - SAME_FOCAL_LENGTH_TOLERANCE),
                    magnitude_mm * (1.0 + SAME_FOCAL_LENGTH_TOLERANCE),
                ),
            )
        self.open_every_group = focal_length_mm is not None
        self.replace_button.setText(f"Replace {part_number}")
        self._show_replace_buttons(True)
        self.tree.clearSelection()
        self.refresh()

    def end_replacement(self) -> None:
        """Leave replacing: hide the buttons and put the user's filters back."""
        if not self.replacing:
            return
        self.replacing = False
        self.open_every_group = False
        self._show_replace_buttons(False)
        if self.filters_before_replacing is not None:
            self._set_filters(*self.filters_before_replacing)
        self.filters_before_replacing = None
        self.tree.clearSelection()
        self.refresh()

    def _set_filters(
        self,
        search_text: str,
        hide_positive: bool,
        hide_negative: bool,
        focal_length_ends: tuple[float | None, float | None],
    ) -> None:
        """Set the search and sign filters and the |f| range, without a refresh for each."""
        controls = self.filter_controls
        for widget in (self.search_box, controls):
            widget.blockSignals(True)
        for button in (controls.hide_positive_button, controls.hide_negative_button):
            button.blockSignals(True)
        self.search_box.setText(search_text)
        controls.hide_positive_button.setChecked(hide_positive)
        controls.hide_negative_button.setChecked(hide_negative)
        self.focal_length_range.low, self.focal_length_range.high = focal_length_ends
        for widget in (self.search_box, controls):
            widget.blockSignals(False)
        for button in (controls.hide_positive_button, controls.hide_negative_button):
            button.blockSignals(False)

    def _show_replace_buttons(self, shown: bool) -> None:
        self.replace_button.setVisible(shown)
        self.cancel_button.setVisible(shown)
        self.update_replace_button()

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

    def on_selection_changed(self) -> None:
        """Report the chosen lens; while replacing, only enable Replace."""
        self.update_replace_button()
        if not self.replacing:
            self.lens_selected.emit(self.selected_part_number())

    def update_replace_button(self) -> None:
        """Replace is available only while a lens (not a group) is selected."""
        self.replace_button.setEnabled(self.selected_part_number() is not None)

    def confirm_replacement(self) -> None:
        """Replace (or a double-click while replacing): report the chosen lens."""
        part_number = self.selected_part_number()
        if self.replacing and part_number is not None:
            self.replace_confirmed.emit(part_number)

    def on_item_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        """While replacing, double-clicking a lens is the same as choosing it and Replace."""
        if item.data(0, PART_NUMBER_ROLE) is not None:
            self.confirm_replacement()

    # ------------------------------------------------------------------ filtering

    def refresh(self) -> None:
        """Read the filters, update the range sliders and rebuild the tree."""
        controls = self.filter_controls
        lens_filter = self.lens_filter
        coating_choice = controls.coating_combo.currentData()
        suits_beams = coating_choice == SUITS_BEAMS
        lens_filter.coating = None if suits_beams else coating_choice
        lens_filter.wavelengths_nm = self.design_wavelengths_nm if suits_beams else None
        lens_filter.part_number_text = self.search_box.text()
        lens_filter.hide_positive = controls.hide_positive_button.isChecked()
        lens_filter.hide_negative = controls.hide_negative_button.isChecked()

        # Each slider spans the lenses that pass every other filter. The |f| span is taken
        # with the diameter range as it was, and then the diameter span with the new |f|
        # range, so that each always covers the lenses the other leaves.
        self.focal_length_range.span = focal_length_span_mm(
            entry for entry in self.entries if lens_filter.matches(entry, check_focal_length=False)
        )
        lens_filter.focal_length_range_mm = self.focal_length_range.range_mm()
        self.diameter_range.span = diameter_span_mm(
            entry for entry in self.entries if lens_filter.matches(entry, check_diameter=False)
        )
        lens_filter.diameter_range_mm = self.diameter_range.range_mm()
        self._show_range(self.focal_length_range, controls.focal_length_slider)
        self._show_range(self.diameter_range, controls.diameter_slider)
        controls.focal_length_text.setText(self.focal_length_range.text())
        controls.diameter_text.setText(self.diameter_range.text())

        self.rebuild_tree(lens_filter.apply(self.entries))

    def _on_range_moved(self, chosen: ChosenRange, low_step: int, high_step: int) -> None:
        """A range slider moved: store the new range and refresh."""
        chosen.set_from_steps(low_step, high_step)
        self.refresh()

    @staticmethod
    def _show_range(chosen: ChosenRange, slider: RangeSlider) -> None:
        """Put a slider's handles where the chosen range is on its current span."""
        slider.setEnabled(chosen.span is not None)
        slider.set_values(*chosen.steps())

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

        if previously_selected is not None and not self.select_part_number(previously_selected):
            self.on_selection_changed()  # the chosen lens was filtered out
        self.count_label.setText(f"{len(shown_entries)} of {len(self.entries)} lenses shown")
        self.update_expand_button()
        self.update_replace_button()

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
        """One lens row: part number, focal length and product family."""
        item = QTreeWidgetItem(parent, [entry.part_number, f"{entry.focal_length:g}", entry.family])
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

    def toggle_expand_all(self) -> None:
        """Open every group, or close them all if every one is already open."""
        groups = self._group_items()
        expand = not all(item.isExpanded() for item in groups)
        self.tree.setUpdatesEnabled(False)
        for item in groups:
            item.setExpanded(expand)  # on_group_toggled remembers each one
        self.tree.setUpdatesEnabled(True)
        self.update_expand_button()

    def update_expand_button(self) -> None:
        """The button offers to close the groups once they are all open."""
        groups = self._group_items()
        all_open = bool(groups) and all(item.isExpanded() for item in groups)
        self.expand_button.setText("Collapse all" if all_open else "Expand all")

    def on_group_toggled(self, item: QTreeWidgetItem, expanded: bool) -> None:
        """Remember which groups the user opened, so a rebuild can open them again."""
        key = item.data(0, GROUP_KEY_ROLE)
        if key is None:
            return
        if expanded:
            self.expanded_group_keys.add(key)
        else:
            self.expanded_group_keys.discard(key)
        self.update_expand_button()
