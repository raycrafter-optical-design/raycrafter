"""Tests of filtering and grouping lenses for the lens picker."""

import pytest

from raycrafter.lenscatalog import thorlabs
from raycrafter.lenscatalog.catalog import ACHROMATS, SPHERICAL_LENSES, UNCOATED
from raycrafter.lenscatalog.selection import (
    LensFilter,
    available_coatings,
    diameter_label,
    diameter_span_mm,
    focal_length_span_mm,
    group_for_tree,
)


@pytest.fixture(scope="module")
def entries():
    """All offered Thorlabs lenses, as a list."""
    return list(thorlabs.load_lenses().values())


def test_default_filter_shows_every_lens(entries):
    assert len(LensFilter().apply(entries)) == len(entries)


def test_coating_filter_shows_only_that_coating(entries):
    shown = LensFilter(coating=UNCOATED).apply(entries)
    assert shown
    assert all(entry.coating == UNCOATED for entry in shown)


def test_uncoated_comes_first_in_the_coating_list(entries):
    coatings = available_coatings(entries)
    assert coatings[0] == UNCOATED
    assert "A: 350-700 nm" in coatings
    assert coatings[1:] == sorted(coatings[1:])


def test_part_number_search_ignores_case(entries):
    shown = LensFilter(coating=None, part_number_text="la1050").apply(entries)
    assert {entry.part_number for entry in shown} >= {"LA1050", "LA1050-A", "LA1050-B"}
    assert all("LA1050" in entry.part_number for entry in shown)


def test_hide_positive_and_negative(entries):
    only_negative = LensFilter(coating=None, hide_positive=True).apply(entries)
    only_positive = LensFilter(coating=None, hide_negative=True).apply(entries)
    assert only_negative and all(entry.focal_length < 0 for entry in only_negative)
    assert only_positive and all(entry.focal_length > 0 for entry in only_positive)
    assert len(only_negative) + len(only_positive) == len(entries)


def test_focal_length_range_uses_the_magnitude(entries):
    lens_filter = LensFilter(coating=None, focal_length_range_mm=(40.0, 60.0))
    shown = lens_filter.apply(entries)
    assert any(entry.focal_length < 0 for entry in shown)  # e.g. f = -50 mm lenses
    assert all(40.0 <= abs(entry.focal_length) <= 60.0 for entry in shown)


def test_focal_length_span(entries):
    shortest, longest = focal_length_span_mm(entries)
    assert 0 < shortest < 10 < 1000 <= longest
    assert focal_length_span_mm([]) is None


def test_diameter_range_and_span(entries):
    """#70: the aperture filter of the lens browser."""
    narrowest, widest = diameter_span_mm(entries)
    assert 0 < narrowest < 12.7 < 50.8 <= widest
    assert diameter_span_mm([]) is None
    lens_filter = LensFilter(coating=None, diameter_range_mm=(20.0, 30.0))
    shown = lens_filter.apply(entries)
    assert shown and all(20.0 <= entry.clear_diameter_mm <= 30.0 for entry in shown)
    # Each range can be left out, so a slider's limits can come from the other filters.
    outside = [entry for entry in entries if not 20.0 <= entry.clear_diameter_mm <= 30.0]
    assert lens_filter.matches(outside[0], check_diameter=False)


def test_tree_is_ordered_by_category_diameter_sign_and_focal_length(entries):
    tree = group_for_tree(LensFilter(coating=None).apply(entries))
    assert list(tree)[:2] == [SPHERICAL_LENSES, ACHROMATS]
    for by_diameter in tree.values():
        assert list(by_diameter) == sorted(by_diameter)
        for lenses in by_diameter.values():
            order = [(entry.focal_length < 0, abs(entry.focal_length)) for entry in lenses]
            assert order == sorted(order)


def test_diameter_labels_show_inches_for_inch_sizes():
    assert diameter_label(25.4) == 'Ø 25.4 mm (1")'
    assert diameter_label(12.7) == 'Ø 12.7 mm (1/2")'
    assert diameter_label(6.0) == "Ø 6 mm"
