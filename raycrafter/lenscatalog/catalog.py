"""Reading lens catalogues shipped with RayCrafter.

A catalogue is a gzipped JSON file in ``raycrafter/lenscatalog/data``. It has a ``source``
block describing where it came from, a ``tree`` of vendor category -> family -> part
numbers, and a ``lenses`` dictionary keyed by part number. Each lens entry includes its
focal length (``efl_mm``), clear diameter, coating, tags and the full Zemax prescription as
text (``zmx``). Vendor-specific rules (which lenses to offer, what the coating codes mean,
which RayCrafter category each vendor family belongs to) live in their own modules, e.g.
``raycrafter.lenscatalog.thorlabs``.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from importlib import resources

# The categories lenses are grouped into in the lens picker, in the order it lists them.
# Every vendor module maps its own product families onto these names.
SPHERICAL_LENSES = "Spherical lenses"
ACHROMATS = "Achromats"
AIR_SPACED_DOUBLETS = "Air-spaced doublets"
ACHROMATIC_TRIPLETS = "Achromatic triplets"
MATCHED_ACHROMATIC_PAIRS = "Matched achromatic pairs"
ASPHERES = "Aspheres"
CATEGORY_ORDER = [
    SPHERICAL_LENSES,
    ACHROMATS,
    AIR_SPACED_DOUBLETS,
    ACHROMATIC_TRIPLETS,
    MATCHED_ACHROMATIC_PAIRS,
    ASPHERES,
]

UNCOATED = "Uncoated"  # the coating name of a lens with no anti-reflection coating


@dataclass
class CatalogEntry:
    """One selectable lens, with the data needed to list it and to place it on the bench.

    Attributes
    ----------
    part_number : str
        The vendor's part number.
    zmx_text : str
        The lens prescription as the text of a Zemax .zmx file.
    semi_aperture : float
        Half the clear diameter, in mm.
    focal_length : float
        Effective focal length in mm; negative for a diverging lens.
    category : str
        One of CATEGORY_ORDER.
    family : str
        The vendor's product family, e.g. "Plano-convex lenses".
    coating : str
        Human-readable coating name, e.g. "A: 350-700 nm", or UNCOATED.
    vendor : str
        Who makes the lens, e.g. "Thorlabs".
    mount_type : str or None
        How the lens is mounted, e.g. "Unmounted"; None if the catalogue does not say.
    """

    part_number: str
    zmx_text: str
    semi_aperture: float
    focal_length: float
    category: str
    family: str
    coating: str
    vendor: str
    mount_type: str | None = None

    @property
    def clear_diameter_mm(self) -> float:
        """The clear diameter, in mm."""
        return 2.0 * self.semi_aperture

    @property
    def is_negative(self) -> bool:
        """True for a diverging lens."""
        return self.focal_length < 0


def read_packaged_catalog(file_name: str) -> dict:
    """Load one of the gzipped catalogue JSON files shipped inside the package.

    Parameters
    ----------
    file_name : str
        Name of a file in ``raycrafter/lenscatalog/data``, e.g. ``"thorlabs.json.gz"``.

    Returns
    -------
    dict
        The parsed JSON, with ``source``, ``tree`` and ``lenses`` keys.
    """
    catalog_file = resources.files("raycrafter.lenscatalog").joinpath("data", file_name)
    return json.loads(gzip.decompress(catalog_file.read_bytes()).decode("utf-8"))
