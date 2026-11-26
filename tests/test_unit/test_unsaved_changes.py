"""The model, not the GUI, knows whether the design has unsaved changes (issue #15)."""

import pytest

from raycrafter.core.model import OpticalModel


def test_new_model_counts_as_saved():
    assert not OpticalModel().has_unsaved_changes()


def test_editing_makes_changes_unsaved_and_saving_clears_them(tmp_path):
    model = OpticalModel()
    model.add_lens("AC254-050-A")
    assert model.has_unsaved_changes()

    model.save_project(tmp_path / "design.raycrafter")
    assert not model.has_unsaved_changes()

    model.state.image_z += 5.0
    assert model.has_unsaved_changes()


def test_selecting_another_field_is_not_an_unsaved_change():
    model = OpticalModel()
    model.add_field()
    model.mark_saved()
    model.select_field(0)
    assert not model.has_unsaved_changes()


def test_exporting_a_zmx_file_does_not_count_as_saving(tmp_path):
    """A ZMX file loses fields and Plossl pairs, so quitting must still ask (issue #40)."""
    model = OpticalModel()
    model.add_field()
    model.save_zmx(str(tmp_path / "design.zmx"))
    assert model.has_unsaved_changes()


def test_save_without_a_path_uses_the_project_file(tmp_path):
    model = OpticalModel()
    with pytest.raises(ValueError, match="no project file"):
        model.save_project()

    saved_path = model.save_project(tmp_path / "design")  # the extension is added
    assert saved_path == tmp_path / "design.raycrafter" and model.project_path == saved_path
    model.add_field()
    assert model.save_project() == saved_path
    assert not model.has_unsaved_changes()


def test_open_sets_the_project_file_and_new_and_import_clear_it(tmp_path):
    model = OpticalModel()
    project_path = model.save_project(tmp_path / "design.raycrafter")
    zmx_path = str(tmp_path / "design.zmx")
    model.save_zmx(zmx_path)

    model.new_design()
    assert model.project_path is None
    model.load_project(project_path)
    assert model.project_path == project_path and not model.has_unsaved_changes()
    model.load_zmx(zmx_path)
    assert model.project_path is None


def test_new_and_load_count_as_saved(tmp_path):
    model = OpticalModel()
    zmx_path = str(tmp_path / "design.zmx")
    model.save_zmx(zmx_path)

    model.add_lens("AC254-050-A")
    model.load_zmx(zmx_path)
    assert not model.has_unsaved_changes()

    model.add_lens("AC254-050-A")
    model.new_design()
    assert not model.has_unsaved_changes()
