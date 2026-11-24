"""The model, not the GUI, knows whether the design has unsaved changes (issue #15)."""

from raycrafter.core.model import OpticalModel


def test_new_model_counts_as_saved():
    assert not OpticalModel().has_unsaved_changes()


def test_editing_makes_changes_unsaved_and_saving_clears_them(tmp_path):
    model = OpticalModel()
    model.add_lens("AC254-050-A")
    assert model.has_unsaved_changes()

    model.save_zmx(str(tmp_path / "design.zmx"))
    assert not model.has_unsaved_changes()

    model.state.image_z += 5.0
    assert model.has_unsaved_changes()


def test_selecting_another_field_is_not_an_unsaved_change():
    model = OpticalModel()
    model.add_field()
    model.mark_saved()
    model.select_field(0)
    assert not model.has_unsaved_changes()


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
