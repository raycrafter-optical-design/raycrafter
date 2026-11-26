"""Tests of the RayCrafter project file (``raycrafter.core.project_file``)."""

import copy
import json
from dataclasses import replace

import pytest

from raycrafter.core import OpticalModel
from raycrafter.core.analysis import trace_fast
from raycrafter.core.project_file import (
    FORMAT_VERSION,
    ProjectFileError,
    load_project,
    project_from_dict,
    project_to_dict,
    save_project,
)
from raycrafter.core.state import BeamField, state_from_dict, state_to_dict
from raycrafter.core.zmx_import import load_zmx


@pytest.fixture
def model() -> OpticalModel:
    """A design using everything a project file holds and a ZMX file cannot.

    A catalogue doublet, a flipped plano-convex lens (it has a flat surface), a Plossl pair,
    a paraxial lens and two fields, the second selected and given its own colour. The gap
    behind the first lens, and the gap to the image plane, are locked.
    """
    bench_model = OpticalModel()
    bench_model.add_lens("LA1509-A")
    bench_model.flip_lens(1)
    for _ in range(2):
        bench_model.add_lens("AC254-050-A")
    bench_model.make_assembly(2, 3, "Plossl pair")  # as make_plossl names it
    bench_model.insert_lens(bench_model.make_paraxial_lens(-80.0, z=900.0))
    bench_model.state.image_z = 1000.0
    bench_model.set_gap_locked(0, True)
    bench_model.set_gap_locked(4, True)  # the last lens: the gap to the image plane
    bench_model.state.fields.append(
        BeamField(diameter_mm=4.0, angle_deg=2.5, vergence_dioptres=-1.0, offset_mm=0.5,
                  wavelength_nm=850.0, colour="#123456")
    )  # fmt: skip
    bench_model.state.selected_field_index = 1
    return bench_model


@pytest.fixture
def imported_model(model, tmp_path) -> OpticalModel:
    """The design after a ZMX save and Import ZMX, so it carries its source file."""
    zmx_path = tmp_path / "design.zmx"
    model.save_zmx(str(zmx_path))
    model.load_zmx(str(zmx_path))
    return model


def saved_and_loaded(state, tmp_path):
    project_path = tmp_path / "design.raycrafter"
    save_project(state, project_path)
    return load_project(project_path)


# ----------------------------------------------------------------- the design survives


def test_a_saved_project_loads_back_identical(model, tmp_path):
    assert saved_and_loaded(model.state, tmp_path) == model.state


def test_act508_back_focal_length_survives_a_save_and_load(tmp_path):
    loaded = saved_and_loaded(OpticalModel().state, tmp_path)
    assert trace_fast(loaded).summary.back_focal_length == pytest.approx(190.6, abs=0.1)


def test_part_metadata_survives_a_save_and_load(model, tmp_path):
    """Issue #42: a catalogue lens keeps its vendor and part number."""
    loaded = saved_and_loaded(model.state, tmp_path).lenses
    assert [(lens.part_number, lens.vendor) for lens in loaded] == [
        ("ACT508-200-A", "Thorlabs"),
        ("LA1509-A", "Thorlabs"),
        ("AC254-050-A", "Thorlabs"),
        ("AC254-050-A", "Thorlabs"),
        ("Paraxial f=-80 mm", None),
    ]


def test_the_file_is_plain_json_with_flat_surfaces_as_null(model, tmp_path):
    project_path = tmp_path / "design.raycrafter"
    save_project(model.state, project_path)
    text = project_path.read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text

    data = json.loads(text)
    assert data["format_version"] == FORMAT_VERSION
    assert data["imported_from"] is None
    plano_convex = data["lenses"][1]
    assert plano_convex["flipped"] is True
    assert [surface["radius_mm"] for surface in plano_convex["surfaces"]].count(None) == 1


def test_diverging_lenses_are_recognised_on_loading(model, tmp_path):
    model.set_gap_locked(len(model.state.lenses) - 1, False)  # to add a lens behind the last
    model.add_lens("LC1060-A")
    loaded = saved_and_loaded(model.state, tmp_path)
    assert [lens.is_negative for lens in loaded.lenses] == [
        lens.is_negative for lens in model.state.lenses
    ]
    assert loaded.lenses[-1].is_negative and loaded.lenses[-2].is_negative  # LC1060, f=-80


# --------------------------------------------------------- the imported ZMX is kept


def test_an_imported_design_keeps_the_exact_file(imported_model, tmp_path):
    source = imported_model.state.imported_from
    assert source.file_format == "zmx" and source.file_name == "design.zmx"
    assert source.content() == (tmp_path / "design.zmx").read_bytes()
    assert source.imported_by.startswith("RayCrafter ")


def test_the_imported_file_survives_edits_and_a_save_and_load(imported_model, tmp_path):
    source = imported_model.state.imported_from
    imported_model.delete_lens(0)
    imported_model.add_field()
    assert saved_and_loaded(imported_model.state, tmp_path).imported_from == source


def test_the_embedded_file_imports_again_with_its_part_metadata(imported_model, tmp_path):
    loaded = saved_and_loaded(imported_model.state, tmp_path)
    copy_path = tmp_path / "copy" / loaded.imported_from.file_name
    copy_path.parent.mkdir()
    copy_path.write_bytes(loaded.imported_from.content())

    reimported = load_zmx(str(copy_path), imported_model.catalog).state
    assert [(lens.part_number, lens.vendor) for lens in reimported.lenses] == [
        (lens.part_number, lens.vendor) for lens in loaded.lenses
    ]


def test_the_design_as_imported_is_kept_and_shows_later_edits(imported_model, tmp_path):
    state = imported_model.state
    assert state.modified_since_import() is False
    imported_model.select_field(0)  # choosing a field is not a change
    assert state.modified_since_import() is False

    state.image_z += 5.0
    assert state.modified_since_import() is True
    loaded = saved_and_loaded(state, tmp_path)
    assert loaded.modified_since_import() is True
    assert loaded.imported_from.imported_design.image_z == pytest.approx(state.image_z - 5.0)
    assert loaded.imported_from.imported_design.imported_from is None


def test_a_design_not_imported_has_no_import_to_compare_with(model):
    assert model.state.modified_since_import() is None


def test_reimporting_the_embedded_file_reproduces_the_design_as_imported(imported_model, tmp_path):
    """What a regression test of the importer does: same file, same importer, same bench."""
    source = imported_model.state.imported_from
    imported_model.delete_lens(0)  # the user's edits do not matter
    copy_path = tmp_path / "copy" / source.file_name
    copy_path.parent.mkdir()
    copy_path.write_bytes(source.content())
    reimported = load_zmx(str(copy_path), imported_model.catalog).state
    assert replace(reimported, imported_from=None) == source.imported_design


def test_saving_with_forget_import_drops_the_record(imported_model, tmp_path):
    project_path = tmp_path / "design.raycrafter"
    imported_model.save_project(project_path, forget_import=True)
    assert imported_model.state.imported_from is None
    assert not imported_model.has_unsaved_changes()
    assert load_project(project_path).imported_from is None


def test_a_failed_save_keeps_the_import_record(imported_model, tmp_path):
    with pytest.raises(OSError):
        imported_model.save_project(tmp_path / "missing" / "design.raycrafter", forget_import=True)
    assert imported_model.state.imported_from is not None


def test_new_design_forgets_the_imported_file(imported_model):
    imported_model.new_design()
    assert imported_model.state.imported_from is None


def test_the_imported_file_crosses_the_agent_link(imported_model):
    copied = state_from_dict(json.loads(json.dumps(state_to_dict(imported_model.state))))
    assert copied.imported_from == imported_model.state.imported_from


# ------------------------------------------------------------- bad files are refused


def edited_project(model, change) -> dict:
    data = copy.deepcopy(project_to_dict(model.state))
    change(data)
    return data


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data.update(format="something-else"), "not a RayCrafter project"),
        (lambda data: data.update(format_version=FORMAT_VERSION + 1), "newer RayCrafter"),
        (lambda data: data.update(lock_focus=True), "unknown lock_focus"),
        (lambda data: data.pop("image_z_mm"), "missing image_z_mm"),
        (lambda data: data["lenses"][0]["part"].pop("vendor"), "missing vendor"),
        (lambda data: data["lenses"][0].update(type="mirror"), "type 'mirror'"),
        (lambda data: data["fields"].clear(), "no fields"),
        (lambda data: data.update(selected_field=5), "selected_field is 5"),
        (lambda data: data["lenses"].reverse(), "in order along the axis"),
        (lambda data: data["lenses"][0].update(flipped=1), "true or false"),
        (lambda data: data.update(image_z_mm="far"), "should be a number"),
        (lambda data: data["lenses"][-1].update(locked_to_next=True), "no lens after it"),
        (lambda data: data.update(image_gap_locked=None), "true or false"),
        (
            lambda data: data["imported_from"]["imported_design"].pop("fields"),
            "imported_design is missing fields",
        ),
        (
            lambda data: data["imported_from"]["imported_design"]["fields"].clear(),
            "imported design has no fields",
        ),
    ],
)
def test_bad_project_data_is_refused_with_a_reason(imported_model, change, message):
    with pytest.raises(ProjectFileError, match=message):
        project_from_dict(edited_project(imported_model, change))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data["assemblies"].clear(), "used by a lens but not listed"),
        (lambda data: data["lenses"][3].update(assembly=None), "two or more neighbouring"),
        (lambda data: data.update(next_assembly=0), "above every assembly id"),
        (lambda data: data["assemblies"][0].pop("name"), "missing name"),
    ],
)
def test_bad_assemblies_are_refused(model, change, message):
    with pytest.raises(ProjectFileError, match=message):
        project_from_dict(edited_project(model, change))


def test_an_unlocked_plossl_pair_is_refused(model):
    def unlock_pair(data):
        data["lenses"][2]["locked_to_next"] = False

    with pytest.raises(ProjectFileError, match="inside the assembly 'Plossl pair' must all be"):
        project_from_dict(edited_project(model, unlock_pair))


def test_a_damaged_imported_file_is_refused(imported_model):
    def damage(data):
        data["imported_from"]["content_base64"] = "AAAA" + data["imported_from"]["content_base64"]

    with pytest.raises(ProjectFileError, match="does not match its checksum"):
        project_from_dict(edited_project(imported_model, damage))


def test_infinity_and_invalid_json_are_refused(tmp_path):
    project_path = tmp_path / "bad.raycrafter"
    project_path.write_text('{"image_z_mm": Infinity}', encoding="utf-8")
    with pytest.raises(ProjectFileError, match="Infinity"):
        load_project(project_path)
    project_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProjectFileError, match="not valid JSON"):
        load_project(project_path)


def test_a_failed_save_leaves_the_earlier_file_intact(model, tmp_path):
    project_path = tmp_path / "design.raycrafter"
    save_project(model.state, project_path)
    earlier = project_path.read_bytes()

    model.state.image_z = float("nan")  # json refuses to write NaN
    with pytest.raises(ValueError):
        save_project(model.state, project_path)
    assert project_path.read_bytes() == earlier
    assert list(tmp_path.iterdir()) == [project_path]  # no temporary file left behind
