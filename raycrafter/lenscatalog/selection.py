"""Filtering and grouping catalogue lenses for the lens picker.

This is the logic behind the GUI's "Add lens" dialog, kept free of Qt so it can be tested on
its own and reused by other front ends. ``LensFilter`` decides which lenses are shown;
``group_for_tree`` arranges them as category -> clear diameter -> lenses.

Example
-------
>>> from raycrafter.lenscatalog import thorlabs
>>> from raycrafter.lenscatalog.selection import LensFilter, group_for_tree
>>> lenses = thorlabs.load_lenses().values()
>>> shown = LensFilter(part_number_text="LA1").apply(lenses)
>>> tree = group_for_tree(shown)
>>> list(tree)
['Spherical lenses']
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from raycrafter.lenscatalog.catalog import CATEGORY_ORDER, UNCOATED, CatalogEntry

MM_PER_INCH = 25.4
# Diameters that are these fractions of an inch are labelled in inches too, e.g. 'Ø 25.4 mm (1")'.
INCH_DIAMETER_LABELS = {0.25: '1/4"', 0.5: '1/2"', 1.0: '1"', 1.5: '1.5"', 2.0: '2"', 3.0: '3"'}


@dataclass
class LensFilter:
    """Which lenses the picker shows.

    Attributes
    ----------
    coating : str or None
        Show only lenses with this coating name; None shows every coating.
    focal_length_range_mm : (float, float) or None
        Show only lenses whose |focal length| lies within this range (inclusive, mm).
        None means no limit. The magnitude is used so that one range serves both positive
        and negative lenses; the sign is chosen with ``hide_positive`` and ``hide_negative``.
    hide_positive, hide_negative : bool
        Leave out converging or diverging lenses.
    part_number_text : str
        Show only lenses whose part number contains this text (case does not matter).
    """

    coating: str | None = None
    focal_length_range_mm: tuple[float, float] | None = None
    hide_positive: bool = False
    hide_negative: bool = False
    part_number_text: str = ""

    def matches_ignoring_focal_length(self, entry: CatalogEntry) -> bool:
        """True if the lens passes every test except the focal length range.

        The picker sets its focal length slider's limits from the lenses that pass these
        tests, so this is kept separate from ``matches``.
        """
        if self.coating is not None and entry.coating != self.coating:
            return False
        if entry.is_negative and self.hide_negative:
            return False
        if not entry.is_negative and self.hide_positive:
            return False
        return self.part_number_text.strip().upper() in entry.part_number.upper()

    def matches(self, entry: CatalogEntry) -> bool:
        """True if the lens should be shown."""
        if not self.matches_ignoring_focal_length(entry):
            return False
        if self.focal_length_range_mm is None:
            return True
        shortest, longest = self.focal_length_range_mm
        return shortest <= abs(entry.focal_length) <= longest

    def apply(self, entries: Iterable[CatalogEntry]) -> list[CatalogEntry]:
        """The entries that should be shown, in their original order."""
        return [entry for entry in entries if self.matches(entry)]


def available_coatings(entries: Iterable[CatalogEntry]) -> list[str]:
    """The distinct coating names of the given lenses: UNCOATED first, then alphabetical."""
    coatings = {entry.coating for entry in entries}
    return sorted(coatings, key=lambda coating: (coating != UNCOATED, coating))


def focal_length_span_mm(entries: Iterable[CatalogEntry]) -> tuple[float, float] | None:
    """Smallest and largest |focal length| (mm) of the given lenses; None if there are none."""
    magnitudes = [abs(entry.focal_length) for entry in entries]
    if not magnitudes:
        return None
    return min(magnitudes), max(magnitudes)


def tree_order(entry: CatalogEntry) -> tuple:
    """Sort key for the picker: positive lenses before negative, then shortest |f| first.

    The part number breaks ties, so lenses that differ only in coating sit together.
    """
    return (entry.is_negative, abs(entry.focal_length), entry.part_number)


def group_for_tree(
    entries: Iterable[CatalogEntry],
) -> dict[str, dict[float, list[CatalogEntry]]]:
    """Arrange lenses as category -> clear diameter (mm) -> lenses, ready to show as a tree.

    Categories follow CATEGORY_ORDER (empty ones are left out), diameters increase, and the
    lenses of each diameter are ordered by ``tree_order``.
    """
    grouped: dict[str, dict[float, list[CatalogEntry]]] = {}
    for entry in entries:
        by_diameter = grouped.setdefault(entry.category, {})
        by_diameter.setdefault(entry.clear_diameter_mm, []).append(entry)

    ordered: dict[str, dict[float, list[CatalogEntry]]] = {}
    for category in sorted(grouped, key=CATEGORY_ORDER.index):
        ordered[category] = {
            diameter: sorted(grouped[category][diameter], key=tree_order)
            for diameter in sorted(grouped[category])
        }
    return ordered


def diameter_label(diameter_mm: float) -> str:
    """Tree label for a diameter group, e.g. 'Ø 25.4 mm (1")' or 'Ø 6 mm'."""
    label = f"Ø {diameter_mm:g} mm"
    inches = round(diameter_mm / MM_PER_INCH, 3)
    if inches in INCH_DIAMETER_LABELS:
        label += f" ({INCH_DIAMETER_LABELS[inches]})"
    return label
