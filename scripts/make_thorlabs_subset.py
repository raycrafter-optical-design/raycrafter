"""Cut the packaged Thorlabs catalogue out of the full decoded Thorlabs catalogue.

The full catalogue (about 2100 lenses, 4 MB) is made by the ZMF decoder from Thorlabs'
Zemax catalogue file. RayCrafter ships only the lenses its GUI offers, chosen by
``raycrafter.lenscatalog.thorlabs.select_offered``. Run this again whenever the full
catalogue or the selection rule changes:

    uv run python scripts/make_thorlabs_subset.py path/to/thorlabs_zmx.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from raycrafter.lenscatalog.thorlabs import CATALOG_FILE_NAME, select_offered

OUTPUT_PATH = Path(__file__).parents[1] / "raycrafter" / "lenscatalog" / "data" / CATALOG_FILE_NAME


def make_subset(full_catalog: dict) -> dict:
    """Return a catalogue with the same structure as ``full_catalog`` but only offered lenses."""
    all_lenses = full_catalog["lenses"]
    lenses = {part_number: all_lenses[part_number] for part_number in select_offered(all_lenses)}

    # Keep the category -> family -> parts tree, pruned to the lenses that remain.
    tree: dict = {}
    for category, families in full_catalog["tree"].items():
        for family, part_numbers in families.items():
            kept_parts = [part for part in part_numbers if part in lenses]
            if kept_parts:
                tree.setdefault(category, {})[family] = kept_parts

    source = dict(full_catalog["source"])
    source["n_lenses"] = len(lenses)
    source["subset"] = (
        "Only the lenses RayCrafter offers (raycrafter.lenscatalog.thorlabs."
        "select_offered); made by scripts/make_thorlabs_subset.py."
    )
    return {"source": source, "tree": tree, "lenses": lenses}


def main() -> None:
    """Read the full catalogue named on the command line and write the packaged subset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("full_catalog", type=Path, help="the full decoded thorlabs_zmx.json")
    arguments = parser.parse_args()

    full_catalog = json.loads(arguments.full_catalog.read_text(encoding="utf-8"))
    subset = make_subset(full_catalog)
    OUTPUT_PATH.write_text(
        json.dumps(subset, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {subset['source']['n_lenses']} lenses to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
