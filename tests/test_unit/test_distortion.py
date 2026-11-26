"""Tests of grid distortion (#30).

The physics checks are a paraxial microscope, which has none, and a paraxial lens followed by
a thick flat plate, whose distortion follows from Snell's law. A real pair of doublets is
pinned to a value and compared point by point with Optiland's ``GridDistortion``.
"""

import numpy as np
import pytest
from matplotlib.figure import Figure
from optiland.analysis.grid_distortion import GridDistortion

from raycrafter.api import Bench
from raycrafter.core import OpticalModel
from raycrafter.core.distortion import compute_distortion
from raycrafter.core.examples import basic_microscope
from raycrafter.core.optic_builder import build_optic_for_object, glass
from raycrafter.core.placement import make_paraxial_lens
from raycrafter.core.plot_geometry import distortion_title, grid_polylines
from raycrafter.core.prescription import LensSurface
from raycrafter.core.state import BeamField, ObjectPoint, OpticalState, PlacedLens
from raycrafter.mcp_server.tools import DesignTools
from raycrafter.report import figures

MAGNIFICATION = -20.0  # of the example microscope
RELAY_MAX_DISTORTION_PERCENT = -0.15086


@pytest.fixture
def microscope():
    return basic_microscope()


@pytest.fixture
def relay():
    """Two achromatic doublets imaging an object 10 mm across, at a fixed image plane."""
    model = OpticalModel(add_default_lens=False)
    model.add_lens("AC254-100-A")
    model.add_lens("AC254-050-A")
    model.set_source_gap_locked(False)
    model.set_source_gap(95.0)
    model.state.fields = [ObjectPoint(height_y_mm=5.0, numerical_aperture=0.05)]
    model.state.image_z = 266.9
    return model.state


# ----------------------------------------------------------------------- physics checks


@pytest.mark.parametrize("reference", ["chief_ray", "centroid"])
def test_a_paraxial_pair_has_no_distortion_and_the_exact_magnification(microscope, reference):
    result = compute_distortion(microscope, reference)
    assert result.magnification == pytest.approx(MAGNIFICATION, rel=1e-9)
    assert np.abs(result.grid_distortion_percent).max() < 1e-7
    assert np.abs(result.height_distortion_percent).max() < 1e-7
    assert result.image_x == pytest.approx(MAGNIFICATION * result.object_x, abs=1e-9)
    assert result.image_y == pytest.approx(MAGNIFICATION * result.object_y, abs=1e-9)


def test_a_defocused_image_is_scaled_but_not_distorted(microscope):
    microscope.image_z += 20.0
    result = compute_distortion(microscope)
    assert result.magnification != pytest.approx(MAGNIFICATION, rel=1e-3)
    assert abs(result.max_distortion_percent) < 1e-7


def test_a_flat_plate_behind_the_stop_gives_the_distortion_snells_law_predicts():
    # Object 50 mm in front of a paraxial lens, then a 40 mm glass plate, with 100 mm of air
    # between the lens and the image. The chief ray crosses the middle of the lens unbent, so
    # its image height is the air path times tan(a) plus the plate times tan(a in glass).
    object_distance, plate_thickness, air_path, field_height = 50.0, 40.0, 100.0, 20.0
    lens = make_paraxial_lens(100.0, z=60.0)
    plate = PlacedLens(
        part_number="plate",
        surfaces=[
            LensSurface(radius=np.inf, thickness_after=plate_thickness, material_after="N-BK7"),
            LensSurface(radius=np.inf, thickness_after=0.0, material_after="air"),
        ],
        semi_aperture=60.0,
        z=80.0,
    )
    state = OpticalState(
        lenses=[lens, plate],
        fields=[ObjectPoint(height_y_mm=field_height)],
        source_z=lens.z - object_distance,
        source_gap_locked=False,
        image_z=lens.z + air_path + plate_thickness,
    )
    index = float(np.squeeze(glass("N-BK7").n(state.fields[0].wavelength_um)))

    result = compute_distortion(state)

    angle = np.arctan(result.heights[1:] / object_distance)
    image_height = air_path * np.tan(angle) + plate_thickness * np.tan(
        np.arcsin(np.sin(angle) / index)
    )
    magnification = -(air_path + plate_thickness / index) / object_distance
    expected = 100.0 * (image_height / (abs(magnification) * result.heights[1:]) - 1.0)
    assert result.magnification == pytest.approx(magnification, rel=1e-7)
    assert result.height_distortion_percent[1:] == pytest.approx(expected, abs=1e-5)
    assert result.max_distortion_percent == pytest.approx(expected[-1], abs=1e-5)
    assert expected[-1] < -0.5  # barrel, and large enough to be a real check


def test_a_doublet_pair_has_its_known_distortion(relay):
    result = compute_distortion(relay)
    assert result.max_distortion_percent == pytest.approx(RELAY_MAX_DISTORTION_PERCENT, abs=1e-4)


def test_the_grid_agrees_with_optiland(relay):
    optic = build_optic_for_object(relay, relay.selected_field)
    # "f-theta" is Optiland's linear reference. Its default, "f-tan", takes the tangent of an
    # object *height* as if it were an angle, which is wrong for a finite object.
    optiland_data = GridDistortion(optic, num_points=11, distortion_type="f-theta").data

    result = compute_distortion(relay)

    assert result.image_x == pytest.approx(optiland_data["xr"], abs=1e-9)
    assert result.image_y == pytest.approx(optiland_data["yr"], abs=1e-9)
    assert result.ideal_x == pytest.approx(optiland_data["xp"], abs=1e-8)
    assert result.ideal_y == pytest.approx(optiland_data["yp"], abs=1e-8)
    assert np.abs(result.grid_distortion_percent).max() == pytest.approx(
        optiland_data["max_distortion"], abs=1e-6
    )


def test_the_centroid_of_a_real_spot_is_not_quite_the_chief_ray(relay):
    chief = compute_distortion(relay, "chief_ray")
    centroid = compute_distortion(relay, "centroid")
    difference = np.abs(centroid.image_y - chief.image_y).max()
    assert 1e-6 < difference < 0.05


# ---------------------------------------------------------------------------- the grid


def test_the_grid_corners_are_as_far_out_as_the_farthest_field(microscope):
    microscope.fields.append(ObjectPoint(height_x_mm=0.6, height_y_mm=-0.8))
    result = compute_distortion(microscope, grid_points=5)
    assert result.object_x.shape == result.image_y.shape == (5, 5)
    assert np.hypot(result.object_x, result.object_y).max() == pytest.approx(1.0)
    assert result.heights[0] == 0.0 and result.heights[-1] == pytest.approx(1.0)


def test_blocked_points_are_left_out_and_said_to_be(microscope):
    microscope.lenses[1].semi_aperture = 2.0  # off-axis chief rays now miss the tube lens
    result = compute_distortion(microscope)
    assert 0 < result.blocked_point_count < result.image_y.size
    assert np.isnan(result.image_y[0, 0]) and np.isfinite(result.image_y[5, 5])
    assert np.isnan(result.height_distortion_percent[-1])
    assert np.isfinite(result.max_distortion_percent)
    assert distortion_title(result).endswith("(some points blocked)")


def beams(state):
    state.fields = [BeamField(angle_deg=2.0)]


def on_axis_only(state):
    state.fields = [ObjectPoint()]


def object_on_the_lens(state):
    state.source_z = state.lenses[0].z


def no_lens(state):
    state.lenses = []


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (beams, "finite object"),
        (on_axis_only, "off the axis"),
        (object_on_the_lens, "on the first lens"),
        (no_lens, "No lens"),
    ],
)
def test_with_nothing_to_image_the_result_says_why(microscope, change, reason):
    change(microscope)
    result = compute_distortion(microscope)
    assert not result.has_grid
    assert reason in result.unavailable_reason == distortion_title(result)
    assert np.isnan(result.max_distortion_percent)
    for kind in ("grid_distortion", "distortion"):
        assert isinstance(figures.draw_named_plot(microscope, kind), Figure)


@pytest.mark.parametrize("arguments", [{"reference": "middle"}, {"grid_points": 1}])
def test_unknown_requests_are_refused(microscope, arguments):
    with pytest.raises(ValueError):
        compute_distortion(microscope, **arguments)


def test_grid_polylines_run_along_every_row_and_column():
    x, y = np.meshgrid([0.0, 1.0], [0.0, 2.0])
    line_x, line_y = grid_polylines(x, y)
    assert np.array_equal(line_x, [0, 1, np.nan, 0, 1, np.nan, 0, 0, np.nan, 1, 1, np.nan], True)
    assert np.array_equal(line_y, [0, 0, np.nan, 2, 2, np.nan, 0, 2, np.nan, 0, 2, np.nan], True)


# --------------------------------------------------------------------- scripts and agents


def test_scripts_and_agents_get_the_distortion(microscope):
    model = OpticalModel(add_default_lens=False)
    model.state = microscope
    answer = DesignTools(model).get_distortion()
    assert answer["magnification"] == pytest.approx(MAGNIFICATION)
    assert answer["max_distortion_percent"] == pytest.approx(0.0, abs=1e-6)
    assert answer["object_height_mm"][-1] == pytest.approx(0.5)

    bench = Bench(empty=True)
    bench.add_paraxial_lens(10.0, 30.0)
    bench.set_finite_object(True)
    bench.set_field(0, height_y_mm=0.5)
    assert bench.distortion(grid_points=3).image_y.shape == (3, 3)
