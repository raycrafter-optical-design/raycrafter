"""Tests of loading Zemax files back into a bench (``raycrafter.core.zmx_import``)."""

import warnings

import numpy as np
import pytest
from optiland.fileio import save_zemax_file
from optiland.optic import Optic

from raycrafter.core import OpticalModel
from raycrafter.core.zmx_import import ZmxImportError, load_zmx


@pytest.fixture
def mixed_bench() -> OpticalModel:
    """A catalogue doublet, a flipped one, a paraxial lens and a diverging beam."""
    model = OpticalModel()
    model.add_lens("AC254-050-A")
    model.flip_lens(1)
    model.insert_lens(model.make_paraxial_lens(-80.0, z=300.0))
    model.state.image_z = 400.0
    model.state.selected_field.vergence_dioptres = -2.0
    return model


def test_a_saved_design_loads_back_the_same(mixed_bench, tmp_path):
    zmx_path = str(tmp_path / "design.zmx")
    mixed_bench.save_zmx(zmx_path)
    original = mixed_bench.state

    loaded = load_zmx(zmx_path, mixed_bench.catalog)
    lenses = loaded.state.lenses
    assert [lens.part_number for lens in lenses] == [lens.part_number for lens in original.lenses]
    assert [lens.flipped for lens in lenses] == [False, True, False]
    assert [lens.z for lens in lenses] == pytest.approx([lens.z for lens in original.lenses])
    assert lenses[2].paraxial_focal_length_mm == -80.0
    assert loaded.state.image_z == pytest.approx(400.0)
    assert loaded.state.selected_field.vergence_dioptres == pytest.approx(-2.0)
    assert loaded.notes == []

    # Same optics, so the same focal length.
    efl_before = mixed_bench.trace_fast().summary.effective_focal_length
    mixed_bench.state = loaded.state
    assert mixed_bench.trace_fast().summary.effective_focal_length == pytest.approx(efl_before)


def test_a_vendor_zmx_file_is_recognised_as_the_catalogue_lens(tmp_path):
    model = OpticalModel()
    zmx_path = tmp_path / "LC1054.zmx"
    zmx_path.write_text(model.catalog["LC1054"].zmx_text, encoding="utf-8")
    lens = load_zmx(str(zmx_path), model.catalog).state.lenses[0]
    assert lens.part_number == "LC1054" and lens.is_negative


def test_a_lens_not_in_the_catalogue_keeps_its_prescription(mixed_bench, tmp_path):
    zmx_path = tmp_path / "modified.zmx"
    mixed_bench.save_zmx(str(zmx_path))
    lens = mixed_bench.state.lenses[0]
    # Change the first curvature slightly, so no catalogue lens matches.
    text = zmx_path.read_text()
    old_curvature = f"CURV {1.0 / lens.surfaces[0].radius:.16E}"
    assert old_curvature in text
    zmx_path.write_text(text.replace(old_curvature, "CURV 1.0E-02", 1))

    loaded = load_zmx(str(zmx_path), mixed_bench.catalog)
    imported = loaded.state.lenses[0]
    assert imported.part_number == "Imported lens 1"
    assert imported.surfaces[0].radius == pytest.approx(100.0)
    assert imported.semi_aperture == pytest.approx(lens.semi_aperture)  # from the file's CLAP
    assert not imported.is_negative
    assert any("not in the catalogue" in note for note in loaded.notes)


def test_vergence_counts_the_air_before_the_first_lens(tmp_path):
    # Object 100 mm in front of a stop surface, then 50 mm of air to a paraxial lens: the
    # beam at the lens comes from a point 150 mm away.
    optic = Optic()
    optic.surfaces.add(index=0, radius=np.inf, thickness=100.0)
    optic.surfaces.add(index=1, radius=np.inf, thickness=50.0, is_stop=True)
    optic.surfaces.add(index=2, surface_type="paraxial", f=50.0, thickness=100.0)
    optic.surfaces.add(index=3)
    optic.set_aperture("EPD", 5.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(0.55, is_primary=True)
    zmx_path = str(tmp_path / "leading_air.zmx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save_zemax_file(optic, zmx_path)

    loaded = load_zmx(zmx_path, OpticalModel().catalog)
    assert loaded.state.selected_field.vergence_dioptres == pytest.approx(-1000.0 / 150.0)


def test_a_conic_surface_is_refused(mixed_bench, tmp_path):
    zmx_path = tmp_path / "conic.zmx"
    mixed_bench.save_zmx(str(zmx_path))
    text = zmx_path.read_text().replace("SURF 1\n", "SURF 1\n  CONI -1.0\n", 1)
    zmx_path.write_text(text)
    with pytest.raises(ZmxImportError, match="conic"):
        load_zmx(str(zmx_path), mixed_bench.catalog)


def test_new_design_empties_the_bench(mixed_bench):
    mixed_bench.new_design()
    assert mixed_bench.state.lenses == [] and len(mixed_bench.state.fields) == 1
