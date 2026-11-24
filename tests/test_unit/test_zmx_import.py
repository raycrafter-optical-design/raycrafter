"""Tests of loading Zemax files back into a bench (``raycrafter.core.zmx_import``)."""

import warnings

import numpy as np
import pytest
from optiland.fileio import load_zemax_file, save_zemax_file
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
    # The 5 mm entrance pupil is at the stop; the beam has widened by 150/100 at the lens.
    assert loaded.state.selected_field.diameter_mm == pytest.approx(7.5)


# A small Zemax file written by hand, in the style of a laser scanning system: the stop (a
# scan mirror) is in air 40 mm in front of a biconvex N-BK7 singlet, then the image.
# SURFACES_AFTER_LENS is replaced by the surfaces each test wants between lens and image.
SCANNER_ZMX = """VERS 260127 957 1055371 01055371
MODE SEQ
UNIT MM X W X CM MR CPMM LU
ENPD 4
GCAT SCHOTT
FTYP 0 0 3 1 0 0 0
XFLN 0 0 0
YFLN 0 4 8
WAVM 1 0.55 1
PWAV 1
SURF 0
  TYPE STANDARD
  CURV 0.0
  DISZ INFINITY
SURF 1
  COMM scan mirror
  STOP
  TYPE STANDARD
  CURV 0.0
  DISZ 40
SURF 2
  TYPE STANDARD
  CURV 0.02
  DISZ 5
  GLAS N-BK7 0 0 1.5 40 0 0 0 0 0 0
  FLAP 0 12.7 0
SURF 3
  TYPE STANDARD
  CURV -0.02
  DISZ 30
  FLAP 0 12.7 0
SURFACES_AFTER_LENS
SURF IMAGE_NUMBER
  TYPE STANDARD
  CURV 0.0
  DISZ 0
"""


def write_scanner_zmx(tmp_path, surfaces_after_lens: list[str] = ()) -> str:
    """Write SCANNER_ZMX with the given surface blocks (text after "SURF n") after the lens."""
    blocks = [
        f"SURF {number}\n{block.strip(chr(10))}"
        for number, block in enumerate(surfaces_after_lens, start=4)
    ]
    text = SCANNER_ZMX.replace("SURFACES_AFTER_LENS", "\n".join(blocks))
    text = text.replace("IMAGE_NUMBER", str(4 + len(blocks))).replace("\n\n", "\n")
    zmx_path = tmp_path / "scanner.zmx"
    zmx_path.write_text(text, encoding="utf-8")
    return str(zmx_path)


def test_fields_are_aimed_through_a_stop_in_front_of_the_first_lens(tmp_path):
    zmx_path = write_scanner_zmx(tmp_path)
    loaded = load_zmx(zmx_path, OpticalModel().catalog)
    fields = loaded.state.fields
    assert [field.angle_deg for field in fields] == [0.0, 4.0, 8.0]
    expected_offsets = [40.0 * np.tan(np.radians(field.angle_deg)) for field in fields]
    assert [field.offset_mm for field in fields] == pytest.approx(expected_offsets)
    assert len({field.colour for field in fields}) == 3  # each angle its own colour

    # Each beam lands where Optiland, tracing the file itself, puts it, with the same spot.
    model = OpticalModel()
    model.state = loaded.state
    spots = model.trace_fast().spots
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optic = load_zemax_file(zmx_path)
    for field, spot in zip(fields, spots):
        relative_field = field.angle_deg / 8.0
        optic.trace_generic(0.0, relative_field, 0.0, 0.0, 0.55)
        chief_ray_y = float(np.asarray(optic.surfaces.y)[-1][0])
        assert spot.position_y[0] == pytest.approx(chief_ray_y, abs=1e-9)  # ray 0: the centre

        optic.trace(0.0, relative_field, 0.55, 6, "hexapolar")
        x, y = np.asarray(optic.surfaces.x)[-1], np.asarray(optic.surfaces.y)[-1]
        rms_radius = np.sqrt(np.mean((x - x.mean()) ** 2 + (y - y.mean()) ** 2))
        assert spot.rms_radius == pytest.approx(rms_radius, rel=1e-6)


def test_apertures_in_the_file_are_read_reported_or_respected(tmp_path):
    zmx_path = write_scanner_zmx(
        tmp_path,
        [
            # A baffle in air that blocks the 8 degree beam, which reaches 8 mm off axis here.
            "  COMM baffle\n  TYPE STANDARD\n  CURV 0.0\n  DISZ 10\n  CLAP 0 5 0",
            # A glass cube with a 12 x 16 mm rectangular aperture (which Optiland ignores).
            "  TYPE STANDARD\n  CURV 0.0\n  DISZ 10\n  GLAS N-BK7 0 0 1.5 40 0 0 0 0 0 0\n"
            "  SQAP 6 8 0",
            "  TYPE STANDARD\n  CURV 0.0\n  DISZ 10\n  SQAP 6 8 0",
            # A paraxial lens drawn 2 mm across, with no aperture: Zemax blocks nothing there.
            '  TYPE PARAXIAL\n  CURV 0.0\n  PARM 1 50\n  PARM 2 1\n  DISZ 50\n  DIAM 1 1 0 0 1 ""',
        ],
    )
    loaded = load_zmx(zmx_path, OpticalModel().catalog)
    _singlet, cube, paraxial = loaded.state.lenses
    notes = "\n".join(loaded.notes)

    assert cube.semi_aperture == pytest.approx(6.0) and not cube.is_negative
    assert "12 x 16 mm rectangular aperture" in notes
    assert "no clear aperture" not in notes

    # Kept at its drawn size, which blocks rays Zemax would let through; the user is told.
    assert paraxial.semi_aperture == pytest.approx(1.0)
    assert "Paraxial f=+50 mm has no aperture in the file" in notes
    assert "RayCrafter blocks rays outside the size the file draws it (2 mm across)" in notes

    assert 'Surface 4 ("baffle") has a 10 mm round aperture in air' in notes
    assert "it blocks some of the file's rays" in notes


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
