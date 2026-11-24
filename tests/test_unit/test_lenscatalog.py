"""Tests of the packaged lens catalogue.

The catalogue is read with importlib.resources, so under tox (which installs the built
package, not the source folder) these tests also check that the data file is in the wheel.
"""

import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import pytest
from optiland.fileio import load_zemax_file

from raycrafter.core import OpticalModel, optic_builder
from raycrafter.lenscatalog import thorlabs
from raycrafter.lenscatalog.catalog import ASPHERES, CATEGORY_ORDER, UNCOATED, read_packaged_catalog


@pytest.fixture(scope="module")
def lenses():
    """The packaged Thorlabs lenses, loaded once for this module."""
    return thorlabs.load_lenses()


def test_catalogue_holds_all_rotationally_symmetric_thorlabs_lenses():
    records = read_packaged_catalog(thorlabs.CATALOG_FILE_NAME)["lenses"]
    assert len(records) == 1564
    assert not any("cylindrical" in record["tags"] for record in records.values())


def test_aspheres_are_not_offered_until_supported(lenses):
    # See https://github.com/raycrafter-optical-design/raycrafter/issues/2
    assert len(lenses) == 1409
    assert not any(entry.category == ASPHERES for entry in lenses.values())


def test_every_lens_has_a_known_category_and_a_coating_name(lenses):
    assert {entry.category for entry in lenses.values()} <= set(CATEGORY_ORDER)
    assert lenses["LA1050"].coating == UNCOATED
    assert lenses["LA1050-A"].coating == "A: 350-700 nm"
    assert lenses["AC254-100-C"].coating == "C: 1050-1700 nm"
    assert lenses["ACA254-030-1064"].coating == "1064 coating"


def one_lens_per_family(lenses) -> list[str]:
    """The first part number of each product family: a small sample covering every kind."""
    first_of_family = {}
    for part_number, entry in lenses.items():
        first_of_family.setdefault(entry.family, part_number)
    return sorted(first_of_family.values())


def test_bench_rebuilds_each_kind_of_lens_like_optiland_imports_it(lenses, monkeypatch):
    """Our prescription reading and system building must give the same EFL as Optiland's own
    ZMX import, at the lens's design wavelength. (At our 555 nm, IR-designed lenses differ
    from their catalogue focal length by a few percent because of dispersion.)
    """
    for part_number in one_lens_per_family(lenses):
        with tempfile.TemporaryDirectory() as folder, warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Optiland warns about unused catalogue details
            zmx_path = Path(folder) / "lens.zmx"
            zmx_path.write_text(lenses[part_number].zmx_text, encoding="utf-8")
            imported = load_zemax_file(str(zmx_path))
        design_wavelength_um = imported.wavelengths.primary_wavelength.value
        monkeypatch.setattr(optic_builder, "WAVELENGTH_UM", design_wavelength_um)

        model = OpticalModel(catalog=lenses, add_default_lens=False)
        model.add_lens(part_number)
        rebuilt_efl = float(optic_builder.build_optic(model.state).paraxial.f2())

        assert rebuilt_efl == pytest.approx(float(imported.paraxial.f2()), rel=1e-4), part_number
        assert rebuilt_efl == pytest.approx(lenses[part_number].focal_length, rel=5e-3), part_number


def test_lens_catalogue_and_core_do_not_import_qt():
    # Run in a fresh interpreter, since this test process may already have imported Qt.
    code = (
        "import sys, raycrafter.lenscatalog.thorlabs, raycrafter.lenscatalog.selection,"
        " raycrafter.core, raycrafter.report;"
        "sys.exit(any(name.startswith('PyQt6') for name in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
