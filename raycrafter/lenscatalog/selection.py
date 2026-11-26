"""Filtering and grouping catalogue lenses for the lens picker.

This is the logic behind the GUI's lens browser (the "Add catalogue lens" tool), kept free of Qt so it can be
tested on its own and reused by other front ends. ``LensFilter`` decides which lenses are shown;
``group_for_tree`` arranges them as category -> clear diameter -> lenses. ``search_catalog`` is
the simpler keyword-and-range search used by scripts and AI agents.

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
    """Which lenses the lens browser shows.

    Attributes
    ----------
    coating : str or None
        Show only lenses with this coating name; None shows every coating.
    focal_length_range_mm : (float, float) or None
        Show only lenses whose |focal length| lies within this range (inclusive, mm).
        None means no limit. The magnitude is used so that one range serves both positive
        and negative lenses; the sign is chosen with ``hide_positive`` and ``hide_negative``.
    diameter_range_mm : (float, float) or None
        Show only lenses whose clear diameter lies within this range (inclusive, mm). None
        means no limit.
    hide_positive, hide_negative : bool
        Leave out converging or diverging lenses.
    part_number_text : str
        Show only lenses whose part number contains this text (case does not matter).
    wavelengths_nm : tuple of float or None
        Show only lenses whose coating suits all these wavelengths (see
        ``CatalogEntry.suits_wavelengths``); None shows every coating.
    """

    coating: str | None = None
    focal_length_range_mm: tuple[float, float] | None = None
    diameter_range_mm: tuple[float, float] | None = None
    hide_positive: bool = False
    hide_negative: bool = False
    part_number_text: str = ""
    wavelengths_nm: tuple[float, ...] | None = None

    def matches(
        self, entry: CatalogEntry, check_focal_length: bool = True, check_diameter: bool = True
    ) -> bool:
        """True if the lens should be shown.

        The browser sets each range slider's limits from the lenses that pass every other
        test, so either range test can be left out with ``check_focal_length=False`` or
        ``check_diameter=False``.
        """
        if self.coating is not None and entry.coating != self.coating:
            return False
        if self.wavelengths_nm is not None and not entry.suits_wavelengths(self.wavelengths_nm):
            return False
        if entry.is_negative and self.hide_negative:
            return False
        if not entry.is_negative and self.hide_positive:
            return False
        if self.part_number_text.strip().upper() not in entry.part_number.upper():
            return False
        if check_focal_length and not _within(abs(entry.focal_length), self.focal_length_range_mm):
            return False
        return not check_diameter or _within(entry.clear_diameter_mm, self.diameter_range_mm)

    def apply(self, entries: Iterable[CatalogEntry]) -> list[CatalogEntry]:
        """The entries that should be shown, in their original order."""
        return [entry for entry in entries if self.matches(entry)]


def search_catalog(
    entries: Iterable[CatalogEntry],
    query: str = "",
    min_focal_length_mm: float | None = None,
    max_focal_length_mm: float | None = None,
    min_diameter_mm: float | None = None,
    category: str | None = None,
) -> list[CatalogEntry]:
    """The lenses that pass every given filter, shortest |focal length| first.

    This is the search offered to scripts and AI agents (the picker uses ``LensFilter``).

    Parameters
    ----------
    entries : iterable of CatalogEntry
        The lenses to search, e.g. ``thorlabs.load_lenses().values()``.
    query : str
        Text that must appear in the part number, family or coating (case does not matter).
    min_focal_length_mm, max_focal_length_mm : float, optional
        Limits on the *signed* focal length: a diverging lens has a negative one.
    min_diameter_mm : float, optional
        Smallest acceptable clear diameter.
    category : str, optional
        One of ``catalog.CATEGORY_ORDER``, e.g. ``"Achromats"`` (case does not matter).

    Returns
    -------
    list of CatalogEntry
        Sorted by |focal length|, then part number.

    Example
    -------
    >>> from raycrafter.lenscatalog import thorlabs
    >>> found = search_catalog(
    ...     thorlabs.load_lenses().values(),
    ...     "AC254-100",
    ...     min_focal_length_mm=99,
    ...     max_focal_length_mm=101,
    ... )
    >>> sorted(entry.part_number for entry in found)
    ['AC254-100-A', 'AC254-100-AB', 'AC254-100-B', 'AC254-100-C']
    """
    needle = query.lower()
    matches = []
    for entry in entries:
        text = f"{entry.part_number} {entry.family} {entry.coating}".lower()
        if needle not in text:
            continue
        if min_focal_length_mm is not None and entry.focal_length < min_focal_length_mm:
            continue
        if max_focal_length_mm is not None and entry.focal_length > max_focal_length_mm:
            continue
        if min_diameter_mm is not None and entry.clear_diameter_mm < min_diameter_mm:
            continue
        if category is not None and entry.category.lower() != category.lower():
            continue
        matches.append(entry)

    matches.sort(key=lambda entry: (abs(entry.focal_length), entry.part_number))
    return matches


def _within(value: float, value_range: tuple[float, float] | None) -> bool:
    """True if ``value`` lies in the inclusive range; every value lies in a range of None."""
    return value_range is None or value_range[0] <= value <= value_range[1]


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


def diameter_span_mm(entries: Iterable[CatalogEntry]) -> tuple[float, float] | None:
    """Smallest and largest clear diameter (mm) of the given lenses; None if there are none."""
    diameters = [entry.clear_diameter_mm for entry in entries]
    if not diameters:
        return None
    return min(diameters), max(diameters)


def tree_order(entry: CatalogEntry) -> tuple:
    """Sort key for the lens browser: positive lenses before negative, then shortest |f| first.

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
