"""Smoke test of the GUI: build the window off screen and drive it from code."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from raycrafter.gui.lens_picker import ALL_COATINGS, LensPickerDialog  # noqa: E402
from raycrafter.gui.main_window import MainWindow  # noqa: E402
from raycrafter.gui.range_slider import RangeSlider  # noqa: E402
from raycrafter.gui.style import force_light_colour_scheme  # noqa: E402


@pytest.fixture(scope="module")
def application():
    """One QApplication for all GUI tests (Qt allows only one per process)."""
    return QApplication.instance() or QApplication([])


def test_window_adds_pairs_and_deletes_lenses(application):
    window = MainWindow()
    window.show()
    application.processEvents()

    window.model.add_lens("ACT508-200-A")
    window.side_view.rebuild_lens_graphics()
    assert len(window.side_view.lens_graphics) == 2

    window.make_plossl(0)
    window.angle_slider.setValue(50)  # 5 degrees
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

    window.model.state.beam_diameter = 20.0  # as if the beam slider were being dragged
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
