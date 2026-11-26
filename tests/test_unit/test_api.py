"""The Python scripting API (raycrafter.api.Bench), headless and connected to a window."""

import doctest
import json
import math

import numpy as np
import pytest
from matplotlib.figure import Figure

import raycrafter
from raycrafter import api
from raycrafter.api import Bench
from raycrafter.core.model import OpticalModel
from raycrafter.core.state import state_from_dict, state_to_dict
from raycrafter.lenscatalog import selection
from raycrafter.mcp_server.link import LinkServer
from raycrafter.mcp_server.tools import LINK_ONLY_TOOL_NAMES, TOOL_NAMES, DesignTools


def test_the_examples_in_the_docstrings_are_right():
    """The docstring examples are the API's documentation, so they must actually work."""
    for module in (api, selection):
        result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
        assert result.failed == 0


def test_bench_is_importable_from_the_package():
    assert raycrafter.Bench is Bench


# ----------------------------------------------------------------------------- physics


def test_summary_gives_the_known_act508_200_a_focal_lengths():
    bench = Bench()
    bench.autofocus()
    summary = bench.summary()
    assert summary.effective_focal_length_mm == pytest.approx(200.0, abs=0.1)
    assert summary.back_focal_length_mm == pytest.approx(190.6, abs=0.2)
    assert summary.f_number == pytest.approx(20.0, abs=0.1)
    assert bench.image_z_mm == pytest.approx(bench.lenses[0].back_z_mm + 190.6, abs=0.2)


def test_analyses_are_full_resolution_numpy_arrays():
    bench = Bench()
    bench.autofocus()
    mtf = bench.mtf()
    assert isinstance(mtf.frequency, np.ndarray)
    assert len(mtf.frequency) > 25  # the agent tools thin curves to about 25 points
    assert len(bench.spots()[0].x) > 25


def test_field_index_picks_the_field_analysed_without_changing_the_selection():
    bench = Bench()
    bench.add_field(angle_deg=3.0)
    bench.select_field(0)
    on_axis, off_axis = bench.ray_fan(), bench.ray_fan(field_index=1)
    assert not np.allclose(on_axis.tangential_error, off_axis.tangential_error, equal_nan=True)
    assert bench.selected_field_index == 0
    with pytest.raises(ValueError, match="no field 5"):
        bench.mtf(field_index=5)


# ----------------------------------------------------------------------------- editing


def test_moving_a_lens_past_its_neighbour_returns_its_new_index():
    bench = Bench(empty=True)
    bench.add_lens("LA1509-A")
    bench.add_lens("LA1509-A", front_z_mm=200)
    moved = bench.move_lens(0, front_z_mm=260)
    assert moved.index == 1
    assert [lens.index for lens in bench.lenses] == [0, 1]


def test_adding_a_lens_where_there_is_no_room_puts_it_in_the_nearest_gap():
    bench = Bench()  # ACT508-200-A from z = 100 mm
    bench.add_lens("LA1509-A", front_z_mm=105)
    first, second = bench.lenses
    assert first.back_z_mm <= second.front_z_mm  # no overlap


def test_bad_requests_raise_value_error_with_a_reason():
    bench = Bench()
    with pytest.raises(ValueError, match="Unknown part number"):
        bench.add_lens("NOT-A-LENS")
    with pytest.raises(ValueError, match="no lens 3"):
        bench.move_lens(3, 50)
    with pytest.raises(ValueError, match="colour"):
        bench.set_field(0, colour="orange")


def test_a_bad_change_in_add_field_leaves_no_half_made_field():
    bench = Bench()
    with pytest.raises(ValueError):
        bench.add_field(diameter_mm=-1)
    with pytest.raises(TypeError):
        bench.add_field(no_such_property=1)
    assert len(bench.fields) == 1
    assert bench.selected_field_index == 0


def test_a_field_colour_can_be_set_and_reset_to_the_wavelengths():
    bench = Bench()
    assert bench.set_field(0, colour="#ff8800").colour_follows_wavelength is False
    reset = bench.set_field(0, colour="wavelength")
    assert reset.colour_follows_wavelength is True
    assert reset.colour != "#ff8800"


def test_snapshots_do_not_change_when_the_design_does():
    bench = Bench()
    before = bench.lenses[0]
    bench.move_lens(0, 150)
    assert before.front_z_mm == 100.0
    assert bench.lenses[0].front_z_mm == 150.0


def test_state_returns_a_copy():
    bench = Bench()
    bench.state().lenses.clear()
    assert len(bench.lenses) == 1


# ----------------------------------------------------------------------------- output


@pytest.mark.parametrize("kind", api.PLOT_KINDS)
def test_every_plot_kind_gives_a_figure(kind):
    assert isinstance(Bench().plot(kind), Figure)


def test_write_report(tmp_path):
    report_path = Bench().write_report(tmp_path / "report.html")
    assert report_path.read_text().startswith("<!DOCTYPE html>")


def test_save_and_load_zmx_round_trip(tmp_path):
    bench = Bench()
    bench.add_lens("AC254-050-A")
    saved_path = bench.save_zmx(tmp_path / "design.zmx")
    assert saved_path.is_absolute()

    loaded = Bench.from_zmx(saved_path)
    assert [lens.part_number for lens in loaded.lenses] == ["ACT508-200-A", "AC254-050-A"]
    assert loaded.has_unsaved_changes is False


def test_save_and_open_project_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bench = Bench()
    bench.add_field()
    saved_path = bench.save_project("design")
    assert saved_path == tmp_path / "design.raycrafter" and bench.project_path == saved_path

    loaded = Bench.from_project(saved_path)
    assert len(loaded.fields) == 2
    assert loaded.has_unsaved_changes is False and loaded.project_path == saved_path


def test_unsaved_work_is_kept_unless_discarding_is_asked_for():
    bench = Bench()
    bench.add_lens("AC254-050-A")
    with pytest.raises(ValueError, match="unsaved changes"):
        bench.new_design()
    bench.new_design(discard_unsaved=True)
    assert bench.lenses == []


# ----------------------------------------------------------------------------- state copying


def test_state_survives_a_trip_through_json():
    model = OpticalModel()
    model.add_lens("LA1509-A")  # plano-convex: one flat surface, radius infinite
    model.flip_lens(1)
    model.add_field().colour = "#123456"
    text = json.dumps(state_to_dict(model.state))
    assert state_from_dict(json.loads(text)) == model.state


# ----------------------------------------------------------------------------- connected


@pytest.fixture
def window(tmp_path):
    """A stand-in for a RayCrafter window: a link serving DesignTools on its own model."""
    tools = DesignTools(OpticalModel())

    def handler(tool_name, arguments):
        assert tool_name in TOOL_NAMES + LINK_ONLY_TOOL_NAMES
        return getattr(tools, tool_name)(**arguments)

    endpoint_path = tmp_path / "agent_link.json"
    server = LinkServer(handler, endpoint_path)
    server.start()
    yield tools.model, endpoint_path
    server.stop()


def test_a_connected_bench_edits_the_windows_design(window):
    window_model, endpoint_path = window
    bench = Bench.connect(endpoint_path)
    assert bench.is_connected

    placed = bench.add_lens("AC254-050-A")
    bench.set_field(0, wavelength_nm=np.float64(633))  # NumPy numbers must cross the link
    assert window_model.state.lenses[placed.index].part_number == "AC254-050-A"
    assert window_model.state.fields[0].wavelength_nm == 633


def test_a_connected_bench_sees_changes_made_in_the_window(window):
    window_model, endpoint_path = window
    bench = Bench.connect(endpoint_path)
    window_model.state.lenses[0].z = 120.0  # as if dragged by hand
    assert bench.lenses[0].front_z_mm == 120.0


def test_connected_analyses_match_headless_ones(window):
    _, endpoint_path = window
    connected, headless = Bench.connect(endpoint_path), Bench()
    connected.autofocus()
    headless.autofocus()
    assert connected.image_z_mm == pytest.approx(headless.image_z_mm)
    assert connected.summary().rms_spot_radius_um == pytest.approx(
        headless.summary().rms_spot_radius_um
    )


def test_connected_file_paths_are_made_absolute(window, tmp_path, monkeypatch):
    window_model, endpoint_path = window
    monkeypatch.chdir(tmp_path)
    saved_path = Bench.connect(endpoint_path).save_zmx("relative.zmx")
    assert saved_path == tmp_path.resolve() / "relative.zmx"
    assert saved_path.exists()


def test_connecting_with_no_window_raises_connection_error(tmp_path):
    with pytest.raises(ConnectionError, match="No RayCrafter window"):
        Bench.connect(tmp_path / "no_window.json")


def test_errors_in_the_window_reach_the_script_as_value_errors(window):
    _, endpoint_path = window
    with pytest.raises(ValueError, match="Unknown part number"):
        Bench.connect(endpoint_path).add_lens("NOT-A-LENS")


def test_summary_values_are_plain_python_numbers():
    summary = Bench().summary()
    for value in (summary.effective_focal_length_mm, summary.image_z_mm):
        assert type(value) is float
    assert not math.isnan(summary.rms_spot_radius_um[0])
