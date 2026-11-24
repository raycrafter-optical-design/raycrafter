"""Smoke test of the GUI: build the window off screen and drive it from code."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QDesktopServices  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from raycrafter.gui.about_dialog import DOCS_URL, PROJECT_URL, AboutDialog  # noqa: E402
from raycrafter.gui.lens_picker import ALL_COATINGS, LensPickerDialog  # noqa: E402
from raycrafter.gui.main_window import MainWindow  # noqa: E402
from raycrafter.gui.range_slider import RangeSlider  # noqa: E402
from raycrafter.gui.style import force_light_colour_scheme  # noqa: E402


@pytest.fixture(scope="module")
def application():
    """One QApplication for all GUI tests (Qt allows only one per process)."""
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def discard_unsaved_changes(monkeypatch):
    """Answer "Discard" to "Save your changes?", which would otherwise block a headless test.

    Closing a window with unsaved changes asks this, and most tests change the bench.
    Returns the list of questions asked, for tests that check one was.
    """
    questions = []

    def answer(_parent, _title, text, *_args):
        questions.append(text)
        return QMessageBox.StandardButton.Discard

    monkeypatch.setattr(QMessageBox, "question", answer)
    return questions


def test_window_adds_pairs_and_deletes_lenses(application):
    window = MainWindow()
    window.show()
    application.processEvents()

    window.model.add_lens("ACT508-200-A")
    window.side_view.rebuild_lens_graphics()
    assert len(window.side_view.lens_graphics) == 2

    window.make_plossl(0)
    window.model.state.selected_field.angle_deg = 5.0
    window.recompute_and_redraw()
    window.update_slow_plots()
    window.mtf_tabs.setCurrentIndex(1)  # ray fan tab
    assert "EFL=" in window.status_label.text()

    window.delete_lens(1)
    window.recompute_and_redraw()
    assert len(window.side_view.lens_graphics) == 1
    window.close()


def test_light_colour_scheme_is_forced(application):
    """Button text must stay dark when the OS is in dark mode (white-on-white on Windows 11).

    The headless "offscreen" platform has no colour-scheme support and reports Unknown, so this
    checks only that the scheme is not dark and that the button text colour is dark. The real
    dark-mode case has to be checked by hand on Windows 11 or macOS in dark mode.
    """
    force_light_colour_scheme(application)
    assert application.styleHints().colorScheme() != Qt.ColorScheme.Dark
    button_text_colour = application.palette().buttonText().color()
    assert button_text_colour.lightness() < 128


def test_lens_picker_filters_and_selects(application):
    window = MainWindow()
    picker = LensPickerDialog(window.model.catalog, parent=window)
    assert picker.coating_combo.currentText() == ALL_COATINGS
    assert not picker.insert_button.isEnabled()  # nothing selected yet

    all_coatings_count = len(picker._lens_items())
    picker.coating_combo.setCurrentIndex(picker.coating_combo.findText("Uncoated"))
    assert 0 < len(picker._lens_items()) < all_coatings_count
    picker.coating_combo.setCurrentIndex(picker.coating_combo.findText(ALL_COATINGS))

    picker.search_box.setText("act508-200")
    assert {item.text(0) for item in picker._lens_items()} >= {"ACT508-200-A", "ACT508-200-B"}
    assert picker.select_part_number("ACT508-200-A")
    assert picker.selected_part_number() == "ACT508-200-A"
    assert picker.insert_button.isEnabled()

    picker.hide_positive_checkbox.setChecked(True)  # the selected lens is filtered out
    assert picker._lens_items() == []
    assert not picker.insert_button.isEnabled()
    window.close()


def test_lens_picker_focal_length_slider_narrows_the_list(application):
    window = MainWindow()
    picker = LensPickerDialog(window.model.catalog, parent=window)
    shown_before = len(picker._lens_items())
    picker.focal_length_slider.values_changed.emit(0, 500)  # the shorter half, on a log scale
    shortest, longest = picker.lens_filter.focal_length_range_mm
    assert longest < picker.slider_span_mm[1]
    focal_lengths = [abs(float(item.text(1))) for item in picker._lens_items()]
    assert 0 < len(focal_lengths) < shown_before
    assert max(focal_lengths) <= longest
    window.close()


class FakeClick:
    """Stands in for pyqtgraph's mouse click event, which cannot be made directly."""

    def __init__(self, scene_position: QPointF) -> None:
        self.scene_position = scene_position

    def button(self):
        return Qt.MouseButton.LeftButton

    def scenePos(self):
        return self.scene_position


def scene_point_on_axis(side_view, z_mm: float) -> QPointF:
    """Scene coordinates of the point on the optical axis at ``z_mm``."""
    return side_view.getPlotItem().vb.mapViewToScene(QPointF(z_mm, 0.0))


def test_placing_a_lens_with_the_ghost(application):
    window = MainWindow()
    window.show()
    application.processEvents()
    side_view = window.side_view
    existing = window.model.state.lenses[0]

    # Pointing at the existing lens snaps the ghost next to it; a click places it there.
    side_view.start_placement(window.model.make_lens("LA1050"))
    side_view.move_ghost(side_view.placement, scene_point_on_axis(side_view, existing.z + 1.0))
    assert side_view.placement.fits
    side_view.on_scene_clicked(FakeClick(scene_point_on_axis(side_view, existing.z + 1.0)))
    assert side_view.placement is None
    lenses = window.model.state.lenses
    assert [lens.part_number for lens in lenses] == ["LA1050", "ACT508-200-A"]
    assert lenses[0].back_z <= lenses[1].z + 1e-9
    assert len(side_view.lens_graphics) == 2

    # Escape (here: cancel_placement) leaves the bench unchanged.
    side_view.start_placement(window.model.make_lens("LA1050"))
    side_view.cancel_placement()
    assert len(window.model.state.lenses) == 2
    assert all(graphics.region.movable for graphics in side_view.lens_graphics)
    window.close()


def test_placing_hovering_and_dragging_a_paraxial_lens(application):
    window = MainWindow()
    window.show()
    application.processEvents()
    side_view = window.side_view

    # The focal length box sets the new lens; a click on the axis places it.
    window.paraxial_focal_length_box.setValue(-80)
    window.add_paraxial_lens_button.click()
    assert side_view.placement.lens.paraxial_focal_length_mm == -80.0
    side_view.on_scene_clicked(FakeClick(scene_point_on_axis(side_view, 40.0)))
    paraxial_lens = window.model.state.lenses[0]
    assert paraxial_lens.is_paraxial and paraxial_lens.z == pytest.approx(40.0, abs=1.0)

    # A zero-thickness lens can still be hovered a couple of pixels off its line ...
    window.recompute_and_redraw()
    pixel_mm = side_view.getPlotItem().vb.viewPixelSize()[0]
    near_line = QPointF(paraxial_lens.z + 2 * pixel_mm, 0.0)
    assert side_view.lens_index_at(near_line) == 0
    assert not window.build_lens_menu(0).actions()[0].isEnabled()  # Flip is greyed out

    # ... and dragged: the drag region is wider than the lens, centred on it.
    graphics = side_view.lens_graphics[0]
    region_start, region_end = graphics.region.getRegion()
    assert region_start < paraxial_lens.z < region_end
    graphics.region.setRegion((region_start + 10.0, region_end + 10.0))
    assert paraxial_lens.z == pytest.approx(region_start + 10.0 + graphics.grab_margin_mm)

    # A focal length of zero is not a lens, so the button is switched off.
    window.paraxial_focal_length_box.setValue(0)
    assert not window.add_paraxial_lens_button.isEnabled()
    window.close()


def test_field_window_adds_edits_and_removes_fields_live(application):
    window = MainWindow()
    window.show()
    window.add_field_button.click()
    editor = window.field_editor
    assert editor.isVisible() and editor.field_list.count() == 1
    assert not editor.remove_button.isEnabled()  # the only field cannot be removed

    # "+" adds a copy of the selected field and selects it; edits go to that field at once.
    editor.add_button.click()
    assert editor.field_list.count() == 2 and window.model.state.selected_field_index == 1
    editor.value_boxes["angle_deg"].setValue(4.0)
    editor.value_boxes["wavelength_nm"].setValue(450.0)
    assert window.model.state.fields[1].angle_deg == 4.0
    assert window.model.state.fields[0].angle_deg == 0.0
    assert window.update_timer.isActive()  # the window redraws without waiting for "OK"

    window.recompute_and_redraw()
    assert len(window.side_view.ray_curves) == len(window.spot_panel.field_scatters) == 2

    # A colour of the user's own; "From wavelength" goes back to the automatic one.
    editor.set_colour("#ff00ff")
    assert window.model.state.selected_field.display_colour() == "#ff00ff"
    editor.auto_colour_button.click()
    assert window.model.state.selected_field.colour is None

    # Picking the first field in the list fills the form with its values.
    editor.field_list.setCurrentRow(0)
    assert window.model.state.selected_field_index == 0
    assert editor.value_boxes["angle_deg"].value() == 0.0

    # "-" removes the selected fields.
    editor.field_list.item(1).setSelected(True)
    editor.field_list.item(0).setSelected(False)
    editor.remove_button.click()
    assert len(window.model.state.fields) == 1 and editor.field_list.count() == 1
    window.recompute_and_redraw()
    assert len(window.side_view.ray_curves) == 1
    window.close()


def test_field_window_shows_values_outside_its_usual_ranges(application):
    """A -20 D beam (an object 50 mm away, from a ZMX file) was shown, and edited, as -10 D."""
    window = MainWindow()
    window.model.state.selected_field.vergence_dioptres = -20.0
    window.add_field_button.click()
    vergence_box = window.field_editor.value_boxes["vergence_dioptres"]
    assert vergence_box.value() == -20.0
    vergence_box.stepUp()
    assert window.model.state.selected_field.vergence_dioptres == pytest.approx(-19.95)
    window.close()


def test_a_drag_event_for_a_deleted_lens_is_ignored(application):
    """Pressing Del mid-drag rebuilt the graphics; the next drag event raised ValueError."""
    window = MainWindow()
    window.model.add_lens("AC254-050-A")
    window.side_view.rebuild_lens_graphics()
    dragged = window.side_view.lens_graphics[1]
    window.delete_lens(1)  # rebuilds the graphics, so `dragged` is no longer shown
    window.side_view.on_lens_dragged(dragged)
    assert len(window.model.state.lenses) == 1
    window.close()


def test_picking_another_field_is_not_an_unsaved_change(application):
    """Zemax files do not store which field is selected, so selecting one changes nothing."""
    window = MainWindow()
    window.model.add_field()
    window.model.mark_saved()
    window.model.select_field(0)
    assert not window.model.has_unsaved_changes()
    window.close()


def test_range_slider_handles_follow_the_mouse_and_cannot_cross(application):
    slider = RangeSlider(0, 100)
    slider.resize(220, 30)
    slider.show()
    received = []
    slider.values_changed.connect(lambda low, high: received.append((low, high)))
    middle_y = slider.height() // 2

    # Press near the right end: the high handle is grabbed and dragged to the left.
    QTest.mousePress(slider, Qt.MouseButton.LeftButton, pos=QPoint(200, middle_y))
    QTest.mouseMove(slider, QPoint(110, middle_y))
    QTest.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=QPoint(110, middle_y))
    low, high = slider.values()
    assert low == 0 and 40 < high < 60
    assert received[-1] == (low, high)

    # Dragging the low handle past the high one stops it there.
    QTest.mousePress(slider, Qt.MouseButton.LeftButton, pos=QPoint(10, middle_y))
    QTest.mouseMove(slider, QPoint(210, middle_y))
    QTest.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=QPoint(210, middle_y))
    assert slider.values() == (high, high)
    slider.close()


def test_window_redraws_when_the_beam_is_wider_than_the_lens(application):
    """A 10 mm beam on a Ø2 mm lens crashed the field curvature plot with NaN (issue #3)."""
    window = MainWindow()
    window.delete_lens(0)
    window.model.add_lens("AC020-004-A")
    window.side_view.rebuild_lens_graphics()
    window.recompute_and_redraw()
    window.update_slow_plots()
    window.mtf_tabs.setCurrentIndex(1)  # ray fan, with its blocked (NaN) rays
    window.update_mtf_tab()
    assert "RMS=no light" not in window.status_label.text()
    window.close()


def test_drags_defer_the_slow_work_until_release(application):
    """During a drag the f-number keeps its last value and the slower plots fade; on release
    both are recomputed and the plots return to normal."""
    window = MainWindow()
    window.recompute_and_redraw()
    window.update_slow_plots()
    assert "f/20.0" in window.status_label.text()
    fade = window.mtf_panel.graphicsEffect()
    assert not fade.isEnabled()

    window.model.state.selected_field.diameter_mm = 20.0  # as if edited during a drag
    window.side_view.lens_drag_active = True  # (any drag will do)
    window.recompute_and_redraw()
    assert "f/20.0" in window.status_label.text()  # stale value kept, not recomputed
    assert fade.isEnabled() and not window.slow_timer.isActive()

    window.side_view.on_lens_drag_finished()  # mouse released
    window.recompute_and_redraw()
    assert "f/10.0" in window.status_label.text()
    window.update_slow_plots()  # what the slow timer runs
    assert not fade.isEnabled()
    window.close()


def test_side_view_fits_a_long_design_but_not_during_a_drag(application):
    window = MainWindow()
    window.show()
    application.processEvents()
    default_z_range = window.side_view.getPlotItem().vb.viewRange()[0]

    window.model.insert_lens(window.model.make_paraxial_lens(50.0, z=700.0))
    window.model.state.image_z = 760.0
    window.side_view.rebuild_lens_graphics()

    # While a lens is held the axes stay still; they are refitted on release.
    window.side_view.lens_drag_active = True
    window.recompute_and_redraw()
    assert window.side_view.getPlotItem().vb.viewRange()[0] == pytest.approx(default_z_range)

    window.side_view.lens_drag_active = False
    window.recompute_and_redraw()
    z_min, z_max = window.side_view.getPlotItem().vb.viewRange()[0]
    assert z_min <= 0.0 and z_max > 760.0
    assert window.side_view.image_max_z() > 760.0  # the image plane can be dragged on

    window.new_design()
    window.recompute_and_redraw()
    assert window.side_view.getPlotItem().vb.viewRange()[0] == pytest.approx(default_z_range)
    window.close()


def test_file_menu_new_save_and_load(application, discard_unsaved_changes, monkeypatch, tmp_path):
    window = MainWindow()
    menu_titles = [action.text() for action in window.menuBar().actions()]
    assert menu_titles == ["RayCrafter", "File", "Help"]
    assert not window.model.has_unsaved_changes()  # the start-up bench counts as saved

    # Save writes the file and marks the design saved.
    window.model.add_lens("AC254-050-A")
    window.model.flip_lens(1)
    assert window.model.has_unsaved_changes()
    zmx_path = str(tmp_path / "design.zmx")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (zmx_path, ""))
    assert window.save_zmx()
    assert not window.model.has_unsaved_changes()

    # New after a change asks first (the fixture answers "Discard"), then empties the bench.
    window.model.state.image_z += 5.0
    window.new_design()
    assert len(discard_unsaved_changes) == 1
    assert window.model.state.lenses == [] and len(window.side_view.lens_graphics) == 0

    # Load brings the saved design back, with its part numbers and orientation.
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args: (zmx_path, ""))
    window.load_zmx()
    lenses = window.model.state.lenses
    assert [lens.part_number for lens in lenses] == ["ACT508-200-A", "AC254-050-A"]
    assert lenses[1].flipped and len(window.side_view.lens_graphics) == 2
    assert not window.model.has_unsaved_changes()
    window.close()


def test_unsaved_changes_can_cancel_new(application, monkeypatch):
    window = MainWindow()
    window.model.add_lens("AC254-050-A")
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Cancel)
    window.new_design()
    assert len(window.model.state.lenses) == 2  # nothing was thrown away
    window.model.mark_saved()  # so the window can close without asking
    window.close()


def test_help_menu_opens_the_documentation(application, monkeypatch):
    window = MainWindow()
    opened_urls = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened_urls.append(url.toString()))
    help_menu = window.menuBar().actions()[-1].menu()
    help_menu.actions()[0].trigger()
    assert opened_urls == [DOCS_URL]
    window.close()


def test_about_box_shows_the_project_link(application):
    about = AboutDialog()
    assert "RayCrafter" in about.text_label.text()
    assert PROJECT_URL in about.text_label.text()
    about.close()
