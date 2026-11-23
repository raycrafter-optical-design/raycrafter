"""Tests of the packaged lens catalogue.

The catalogue is read with importlib.resources, so under tox (which installs the built
package, not the source folder) these tests also check that the JSON is in the wheel.
"""

import subprocess
import sys

from raycrafter.lenscatalog import thorlabs
from raycrafter.lenscatalog.catalog import NEAR_DUPLICATE_TOLERANCE, read_packaged_catalog


def test_packaged_catalogue_holds_only_the_offered_lenses():
    catalog = read_packaged_catalog(thorlabs.CATALOG_FILE_NAME)
    offered = thorlabs.load_achromats()
    assert set(catalog["lenses"]) == set(offered)
    assert len(offered) == 20


def test_positive_lenses_are_listed_before_negative_ones():
    focal_lengths = [entry.focal_length for entry in thorlabs.load_achromats().values()]
    signs = [length < 0 for length in focal_lengths]
    assert signs == sorted(signs)


def test_no_near_duplicate_focal_lengths_within_a_size():
    entries = list(thorlabs.load_achromats().values())
    for index, entry in enumerate(entries):
        for other in entries[index + 1 :]:
            if other.semi_aperture == entry.semi_aperture:
                difference = abs(other.focal_length - entry.focal_length)
                assert difference > NEAR_DUPLICATE_TOLERANCE * abs(entry.focal_length)


def test_lens_catalogue_and_core_do_not_import_qt():
    # Run in a fresh interpreter, since this test process may already have imported Qt.
    code = (
        "import sys, raycrafter.lenscatalog.thorlabs, raycrafter.core, raycrafter.report;"
        "sys.exit(any(name.startswith('PyQt6') for name in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
