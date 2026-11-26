"""Thorlabs lenses.

The packaged file ``data/thorlabs.json.gz`` holds every lens in Thorlabs' Zemax catalogue
except the cylindrical ones. It is made from the full decoded catalogue by
``scripts/make_thorlabs_catalog.py``. This module turns its records into ``CatalogEntry``
objects: it maps Thorlabs product families onto RayCrafter's categories and coating codes
onto readable names.

Example
-------
>>> from raycrafter.lenscatalog import thorlabs
>>> lenses = thorlabs.load_lenses()
>>> lenses["ACT508-200-A"].focal_length
200.0
"""

from __future__ import annotations

import re

from raycrafter.lenscatalog.catalog import (
    ACHROMATIC_TRIPLETS,
    ACHROMATS,
    AIR_SPACED_DOUBLETS,
    ASPHERES,
    MATCHED_ACHROMATIC_PAIRS,
    SPHERICAL_LENSES,
    UNCOATED,
    CatalogEntry,
    read_packaged_catalog,
)

CATALOG_FILE_NAME = "thorlabs.json.gz"
VENDOR_NAME = "Thorlabs"

# Thorlabs family -> RayCrafter category. Families not listed take the category named by
# their Thorlabs category in VENDOR_CATEGORY_TO_CATEGORY.
FAMILY_TO_CATEGORY = {
    "Air-spaced doublets (high power)": AIR_SPACED_DOUBLETS,
    "Hastings achromatic triplets": ACHROMATIC_TRIPLETS,
    "Steinheil achromatic triplets": ACHROMATIC_TRIPLETS,
    "Matched achromatic pairs": MATCHED_ACHROMATIC_PAIRS,
}
VENDOR_CATEGORY_TO_CATEGORY = {
    "Spherical singlets": SPHERICAL_LENSES,
    "Achromats": ACHROMATS,
    "Aspheres": ASPHERES,
}

# Thorlabs coating suffixes. The four broadband anti-reflection coatings get their design
# wavelength range; the rarer codes are shown as they are.
COATING_NAMES = {
    "": UNCOATED,
    "A": "A: 350-700 nm",
    "AB": "AB: 400-1100 nm",
    "B": "B: 650-1050 nm",
    "C": "C: 1050-1700 nm",
}

# The design wavelength range (nm) of each broadband coating, used to offer only lenses
# whose coating suits the beams. The rarer codes (E, E1, D and the laser-line coatings such
# as 532 or YAG) have no range here, so they are only listed under "All coatings".
COATING_RANGES_NM = {
    "A": (350.0, 700.0),
    "AB": (400.0, 1100.0),
    "B": (650.0, 1050.0),
    "C": (1050.0, 1700.0),
}

# ---------------------------------------------------------------------------------------
# Aspheres are not offered yet: LensSurface stores only a radius, so they would be traced
# as spheres. Remove this exclusion once aspheres are supported; see
# https://github.com/raycrafter-optical-design/raycrafter/issues/2
UNSUPPORTED_CATEGORIES = {ASPHERES}
# ---------------------------------------------------------------------------------------


def coating_name(coating_code: str) -> str:
    """Readable name of a Thorlabs coating code, e.g. "B" -> "B: 650-1050 nm"."""
    return COATING_NAMES.get(coating_code, f"{coating_code} coating")


def category_of(lens_record: dict) -> str:
    """The RayCrafter category of one Thorlabs lens record."""
    family_category = FAMILY_TO_CATEGORY.get(lens_record["family"])
    if family_category is not None:
        return family_category
    return VENDOR_CATEGORY_TO_CATEGORY[lens_record["category"]]


def mount_type_of(lens_record: dict) -> str | None:
    """The mount type, "Mounted" or "Unmounted", if the record's description gives it.

    The catalogue has no field for the mounting and only a few descriptions mention it. The
    other lenses give None (unknown) rather than being assumed to be unmounted.
    """
    match = re.search(r"\b(un)?mounted\b", lens_record["description"], re.IGNORECASE)
    return match.group().capitalize() if match else None


def entry_from_lens_record(part_number: str, lens_record: dict) -> CatalogEntry:
    """Make a CatalogEntry from one lens record of the Thorlabs catalogue."""
    return CatalogEntry(
        part_number=part_number,
        zmx_text=lens_record["zmx"],
        semi_aperture=lens_record["max_clear_diameter_mm"] / 2.0,
        focal_length=lens_record["efl_mm"],
        category=category_of(lens_record),
        family=lens_record["family"],
        coating=coating_name(lens_record["coating"]),
        vendor=VENDOR_NAME,
        mount_type=mount_type_of(lens_record),
        coating_range_nm=COATING_RANGES_NM.get(lens_record["coating"]),
    )


def entries_from_records(lens_records: dict) -> dict[str, CatalogEntry]:
    """Turn catalogue records into entries, leaving out the categories not yet supported.

    Parameters
    ----------
    lens_records : dict
        Part number -> lens record, as in the ``lenses`` block of the catalogue file.

    Returns
    -------
    dict
        Part number -> CatalogEntry, in the catalogue's order.
    """
    entries = (
        entry_from_lens_record(part_number, lens_record)
        for part_number, lens_record in lens_records.items()
    )
    return {
        entry.part_number: entry
        for entry in entries
        if entry.category not in UNSUPPORTED_CATEGORIES
    }


def load_lenses() -> dict[str, CatalogEntry]:
    """Load the packaged Thorlabs lenses (all those RayCrafter can model)."""
    return entries_from_records(read_packaged_catalog(CATALOG_FILE_NAME)["lenses"])
