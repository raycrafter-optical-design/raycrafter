"""Make the packaged Thorlabs catalogue from the full decoded Thorlabs catalogue.

The full catalogue (about 2100 lenses, 4 MB) is made by the ZMF decoder from Thorlabs'
Zemax catalogue file. RayCrafter ships every lens in it except the cylindrical ones, which
the bench cannot model, as gzipped JSON. Coating variants of a lens share a prescription, so
the file compresses from about 3 MB to about 110 KB. Run this again whenever the full
catalogue changes:

    uv run python scripts/make_thorlabs_catalog.py path/to/thorlabs_zmx.json
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from raycrafter.lenscatalog.thorlabs import CATALOG_FILE_NAME

OUTPUT_PATH = Path(__file__).parents[1] / "raycrafter" / "lenscatalog" / "data" / CATALOG_FILE_NAME


def is_cylindrical(lens_record: dict) -> bool:
    """True for lenses that are not rotationally symmetric (they cannot go on the bench)."""
    return lens_record["category"] == "Cylindrical lenses" or "cylindrical" in lens_record["tags"]


def make_packaged_catalog(full_catalog: dict) -> dict:
    """Return ``full_catalog`` without its cylindrical lenses, in the same structure."""
    all_lenses = full_catalog["lenses"]
    lenses = {
        part_number: lens_record
        for part_number, lens_record in all_lenses.items()
        if not is_cylindrical(lens_record)
    }

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
        "All lenses except the cylindrical ones; made by scripts/make_thorlabs_catalog.py."
    )
    return {"source": source, "tree": tree, "lenses": lenses}


def main() -> None:
    """Read the full catalogue named on the command line and write the packaged one."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("full_catalog", type=Path, help="the full decoded thorlabs_zmx.json")
    arguments = parser.parse_args()

    full_catalog = json.loads(arguments.full_catalog.read_text(encoding="utf-8"))
    packaged = make_packaged_catalog(full_catalog)
    # mtime=0 keeps the gzip header free of a time stamp, so re-running the script on the
    # same input gives an identical file (and no spurious git change).
    compressed = gzip.compress(json.dumps(packaged, ensure_ascii=False).encode("utf-8"), mtime=0)
    OUTPUT_PATH.write_bytes(compressed)
    print(f"Wrote {packaged['source']['n_lenses']} lenses to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
