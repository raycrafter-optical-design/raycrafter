"""Reading lens catalogues shipped with RayCrafter.

A catalogue is a JSON file in ``raycrafter/lenscatalog/data``. It has a ``source`` block
describing where it came from, a ``tree`` of category -> family -> part numbers, and a
``lenses`` dictionary keyed by part number. Each lens entry includes its focal length
(``efl_mm``), clear diameter, coating, tags and the full Zemax prescription as text
(``zmx``). Vendor-specific selection rules live in their own modules, e.g.
``raycrafter.lenscatalog.thorlabs``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

NEAR_DUPLICATE_TOLERANCE = 0.005  # focal lengths closer than 0.5% count as the same


@dataclass
class CatalogEntry:
    """One selectable lens, with the data needed to place it on the bench.

    Attributes
    ----------
    part_number : str
        The vendor's part number.
    label : str
        Text shown in the GUI's drop-down list.
    zmx_text : str
        The lens prescription as the text of a Zemax .zmx file.
    semi_aperture : float
        Half the clear diameter, in mm.
    focal_length : float
        Effective focal length in mm; negative for a diverging lens.
    """

    part_number: str
    label: str
    zmx_text: str
    semi_aperture: float
    focal_length: float


def read_packaged_catalog(file_name: str) -> dict:
    """Load one of the catalogue JSON files shipped inside the package.

    Parameters
    ----------
    file_name : str
        Name of a file in ``raycrafter/lenscatalog/data``, e.g. ``"thorlabs_achromats.json"``.

    Returns
    -------
    dict
        The parsed JSON, with ``source``, ``tree`` and ``lenses`` keys.
    """
    catalog_file = resources.files("raycrafter.lenscatalog").joinpath("data", file_name)
    return json.loads(catalog_file.read_text(encoding="utf-8"))


def entry_from_lens_record(part_number: str, lens_record: dict) -> CatalogEntry:
    """Make a CatalogEntry from one lens record of a catalogue JSON file."""
    focal_length = lens_record["efl_mm"]
    return CatalogEntry(
        part_number=part_number,
        label=f"{part_number}   f = {focal_length:g} mm",
        zmx_text=lens_record["zmx"],
        semi_aperture=lens_record["max_clear_diameter_mm"] / 2.0,
        focal_length=focal_length,
    )


def drop_near_duplicates(entries: list[CatalogEntry]) -> list[CatalogEntry]:
    """Of lenses whose focal lengths nearly coincide (e.g. 74.9 and 75.0), keep the rounder one.

    Parameters
    ----------
    entries : list of CatalogEntry
        Lenses of one size, sorted by focal length. (Lenses of different diameters are never
        duplicates of each other, so the caller groups them by size first.)

    Returns
    -------
    list of CatalogEntry
        The input without near-duplicates. "Rounder" means closer to a whole number of mm;
        ties keep the earlier lens.
    """

    def distance_from_whole_mm(entry: CatalogEntry) -> float:
        return abs(entry.focal_length - round(entry.focal_length))

    kept: list[CatalogEntry] = []
    for entry in entries:
        if kept:
            previous = kept[-1]
            difference = abs(entry.focal_length - previous.focal_length)
            if difference <= NEAR_DUPLICATE_TOLERANCE * abs(entry.focal_length):
                if distance_from_whole_mm(entry) < distance_from_whole_mm(previous):
                    kept[-1] = entry
                continue
        kept.append(entry)
    return kept
