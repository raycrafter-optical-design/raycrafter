"""Thorlabs lenses.

For now RayCrafter offers a short list of Thorlabs achromats: the A-coated (visible) 2"
positive achromats, plus the A-coated negative achromats, which Thorlabs makes only in 1/2"
and 1" sizes. Near-duplicate focal lengths are dropped. The packaged file
``data/thorlabs_achromats.json`` holds just these lenses. It is cut from the full decoded
Thorlabs catalogue by ``scripts/make_thorlabs_subset.py``, which calls ``select_offered``
below, so the selection rule lives in one place.

Example
-------
>>> from raycrafter.lenscatalog import thorlabs
>>> achromats = thorlabs.load_achromats()
>>> achromats["ACT508-200-A"].focal_length
200.0
"""

from __future__ import annotations

import re

from raycrafter.lenscatalog.catalog import (
    CatalogEntry,
    drop_near_duplicates,
    entry_from_lens_record,
    read_packaged_catalog,
)

CATALOG_FILE_NAME = "thorlabs_achromats.json"

# Positive 2" achromats have 508 in the part number (AC508-100-A, ACT508-200-A, ...); the
# negative ones are ACN127-... and ACN254-.... The "-A" suffix is the visible coating.
OFFERED_PART_PATTERN = re.compile(r"^((AC|ACT)508|ACN(127|254))-.*-A$")
EXCLUDED_TAGS = {"cylindrical", "pair"}  # not single rotationally symmetric lenses


def is_offered_type(part_number: str, lens_record: dict) -> bool:
    """True if a lens is of a type RayCrafter offers (before near-duplicates are dropped)."""
    return (
        lens_record["category"] == "Achromats"
        and lens_record["coating"] == "A"
        and OFFERED_PART_PATTERN.match(part_number) is not None
        and not EXCLUDED_TAGS & set(lens_record["tags"])
    )


def select_offered(lens_records: dict) -> dict[str, CatalogEntry]:
    """Pick the offered lenses from catalogue records, in the order the GUI lists them.

    Positive lenses come first, then negative ones; each group is ordered by increasing
    |focal length|. Near-duplicate focal lengths within one lens size are removed so the
    list stays short.

    Parameters
    ----------
    lens_records : dict
        Part number -> lens record, as in the ``lenses`` block of a catalogue JSON file.

    Returns
    -------
    dict
        Part number -> CatalogEntry.
    """
    entries = [
        entry_from_lens_record(part_number, lens_record)
        for part_number, lens_record in lens_records.items()
        if is_offered_type(part_number, lens_record)
    ]

    def list_order(entry: CatalogEntry) -> tuple:
        """Sort key: positive lenses before negative ones, then by |focal length|."""
        return (entry.focal_length < 0, abs(entry.focal_length))

    entries.sort(key=list_order)
    sizes = sorted({entry.semi_aperture for entry in entries})
    kept = []
    for size in sizes:
        kept += drop_near_duplicates([entry for entry in entries if entry.semi_aperture == size])
    kept.sort(key=list_order)
    return {entry.part_number: entry for entry in kept}


def load_achromats() -> dict[str, CatalogEntry]:
    """Load the packaged Thorlabs achromats (see ``select_offered`` for the order)."""
    return select_offered(read_packaged_catalog(CATALOG_FILE_NAME)["lenses"])
