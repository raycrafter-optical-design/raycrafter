"""The agent tool layer (raycrafter.mcp_server.tools) acting on a headless model."""

import json

import pytest

from raycrafter.core.model import OpticalModel
from raycrafter.mcp_server.tools import DesignTools
from raycrafter.report.figures import PLOT_KINDS


@pytest.fixture
def tools() -> DesignTools:
    return DesignTools(OpticalModel())


def test_summary_matches_the_known_act508_200_a_focal_lengths(tools):
    summary = tools.get_summary()
    assert summary["effective_focal_length_mm"] == pytest.approx(200.0, abs=0.1)
    assert summary["back_focal_length_mm"] == pytest.approx(190.6, abs=0.2)
    assert summary["f_number"] == pytest.approx(20.0, abs=0.1)


def test_lenses_report_their_part_metadata(tools):
    lens = tools.get_design()["lenses"][0]
    assert lens["vendor"] == "Thorlabs" and lens["mount_type"] is None
    assert lens["surface_range"] == [1, 3]
    assert tools.get_lens("TRH254-040-A")["mount_type"] == "Unmounted"


def test_every_result_is_valid_json(tools):
    tools.add_field()
    results = [
        tools.get_design(),
        tools.get_summary(),
        tools.get_mtf(),
        tools.get_ray_fan(),
        tools.get_through_focus(),
        tools.get_field_curvature(),
        tools.search_lenses("AC254", limit=3),
    ]
    for result in results:
        json.dumps(result, allow_nan=False)  # raises on NaN or infinity


def test_curves_are_thinned(tools):
    assert len(tools.get_mtf()["frequency_cycles_per_mm"]) <= 25


def test_search_filters_and_orders_by_focal_length(tools):
    found = tools.search_lenses(min_focal_length_mm=40, max_focal_length_mm=60, limit=5)
    focal_lengths = [lens["focal_length_mm"] for lens in found["lenses"]]
    assert 0 < len(focal_lengths) <= 5 and found["total_matches"] >= len(focal_lengths)
    assert all(40 <= focal_length <= 60 for focal_length in focal_lengths)
    assert tools.search_lenses("no such lens")["lenses"] == []


def test_adding_moving_flipping_and_deleting_lenses(tools):
    design = tools.add_lens("AC254-050-A")
    assert [lens["part_number"] for lens in design["lenses"]] == ["ACT508-200-A", "AC254-050-A"]

    design = tools.flip_lens(1)
    assert design["lenses"][1]["flipped"]

    design = tools.insert_lens("AC254-100-A", front_z_mm=0.0)
    assert design["lenses"][0]["part_number"] == "AC254-100-A"  # nearest free gap is the first

    design = tools.delete_lens(0)
    assert len(design["lenses"]) == 2


def test_paraxial_lens_focuses_at_its_focal_length():
    tools = DesignTools(OpticalModel(add_default_lens=False))
    tools.add_paraxial_lens(focal_length_mm=100.0, front_z_mm=100.0)
    summary = tools.get_summary()
    assert summary["effective_focal_length_mm"] == pytest.approx(100.0)
    design = tools.autofocus()
    assert design["image_z_mm"] == pytest.approx(200.0, abs=0.5)


def test_set_field_changes_only_what_is_given(tools):
    design = tools.set_field(0, angle_deg=2.0, wavelength_nm=532.0)
    field = design["fields"][0]
    assert field["angle_deg"] == 2.0 and field["wavelength_nm"] == 532.0
    assert field["diameter_mm"] == 10.0


@pytest.mark.parametrize(
    "call",
    [
        lambda tools: tools.add_lens("NOT-A-LENS"),
        lambda tools: tools.move_lens(5, 100.0),
        lambda tools: tools.set_field(3, angle_deg=1.0),
        lambda tools: tools.set_field(0, diameter_mm=-1.0),
        lambda tools: tools.set_image_z(0.0),
        lambda tools: tools.make_plossl(0),
        lambda tools: tools.add_paraxial_lens(0.0, 100.0),
    ],
)
def test_bad_requests_raise_value_error_with_a_message(tools, call):
    with pytest.raises(ValueError, match=r"\w"):
        call(tools)


def test_destructive_file_tools_protect_unsaved_work(tools, tmp_path):
    zmx_path = str(tmp_path / "design.zmx")
    tools.save_zmx(zmx_path)
    tools.add_lens("AC254-050-A")
    assert tools.get_design()["has_unsaved_changes"]

    with pytest.raises(ValueError, match="unsaved"):
        tools.new_design()
    with pytest.raises(ValueError, match="unsaved"):
        tools.load_zmx(zmx_path)

    loaded = tools.load_zmx(zmx_path, discard_unsaved=True)
    assert len(loaded["lenses"]) == 1 and not loaded["has_unsaved_changes"]
    assert tools.new_design()["lenses"] == []


def test_loading_a_missing_file_is_a_value_error(tools, tmp_path):
    with pytest.raises(ValueError, match="Could not load"):
        tools.load_zmx(str(tmp_path / "missing.zmx"))


@pytest.mark.parametrize("kind", PLOT_KINDS)
def test_every_plot_is_a_png(tools, kind):
    tools.add_field()
    assert tools.get_plot(kind).startswith(b"\x89PNG")


def test_unknown_plot_names_the_choices(tools):
    with pytest.raises(ValueError, match="ray_diagram"):
        tools.get_plot("pie_chart")


def test_placing_and_moving_say_which_index_the_lens_has(tools):
    assert tools.add_lens("LA1509-A")["new_lens_index"] == 1
    assert tools.insert_lens("LA1509-A", front_z_mm=30.0)["new_lens_index"] == 0
    assert tools.move_lens(0, front_z_mm=1000.0)["moved_lens_index"] == 1  # swapped


def test_a_paraxial_lens_with_no_position_goes_after_the_last_lens(tools):
    design = tools.add_paraxial_lens(focal_length_mm=50.0)
    first, paraxial = design["lenses"]
    assert design["new_lens_index"] == 1
    assert paraxial["front_z_mm"] > first["back_z_mm"]


def test_set_field_colour(tools):
    tools.set_field(0, colour="#ff8800")
    assert tools.model.state.fields[0].colour == "#ff8800"
    tools.set_field(0, colour="wavelength")
    assert tools.model.state.fields[0].colour is None
    with pytest.raises(ValueError, match="hex colour"):
        tools.set_field(0, colour="red")
