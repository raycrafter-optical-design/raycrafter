"""Tests of a finite object: fields that are object points on the source plane (#29).

The physics check is the infinity-corrected microscope of ``examples.basic_microscope``: two
paraxial lenses give a magnification of exactly -f_tube / f_objective, with the image one
tube focal length behind the tube lens, whatever the gap between the lenses.
"""

import copy
import json

import numpy as np
import pytest

from raycrafter.api import Bench
from raycrafter.core import OpticalModel, analysis
from raycrafter.core.examples import basic_microscope
from raycrafter.core.fields import (
    DEFAULT_OBJECT_DISTANCE_MM,
    chief_ray_angle_deg,
    set_finite_object,
)
from raycrafter.core.optic_builder import build_optic, save_zmx
from raycrafter.core.placement import make_paraxial_lens
from raycrafter.core.project_file import (
    ProjectFileError,
    load_project,
    project_from_dict,
    project_to_dict,
    save_project,
)
from raycrafter.core.ray_launch import launch_rays, pupil_points
from raycrafter.core.state import (
    AXIS_START_Z,
    BeamField,
    ObjectPoint,
    state_from_dict,
    state_to_dict,
)
from raycrafter.mcp_server.tools import DesignTools

OBJECTIVE_F_MM = 10.0
TUBE_F_MM = 200.0
MAGNIFICATION = -TUBE_F_MM / OBJECTIVE_F_MM


@pytest.fixture
def microscope():
    return basic_microscope(OBJECTIVE_F_MM, TUBE_F_MM)


# ----------------------------------------------------------------------- physics check


@pytest.mark.parametrize("lens_gap_mm", [50.0, 100.0, 180.0])
def test_a_paraxial_microscope_magnifies_by_minus_f_tube_over_f_objective(lens_gap_mm):
    state = basic_microscope(OBJECTIVE_F_MM, TUBE_F_MM, lens_gap_mm=lens_gap_mm)
    fast_trace = analysis.trace_fast(state)
    assert fast_trace.summary.magnification == pytest.approx(MAGNIFICATION)

    # The real chief ray of the off-axis point lands at M times its height.
    object_height_mm = state.fields[1].height_y_mm
    image_x, image_y = fast_trace.chief_ray_image_points[1]
    assert image_y == pytest.approx(MAGNIFICATION * object_height_mm, abs=1e-9)
    assert image_x == pytest.approx(0.0, abs=1e-12)


def test_the_microscope_image_is_one_tube_focal_length_behind_the_tube_lens(microscope):
    fast_trace = analysis.trace_fast(microscope)
    for spot in fast_trace.spots:
        assert spot.rms_radius < 1e-9  # paraxial lenses image perfectly
    tube_lens_z = microscope.lenses[1].z
    assert analysis.find_best_focus_z(microscope) == pytest.approx(tube_lens_z + TUBE_F_MM)


def test_an_object_point_off_both_axes_images_in_both(microscope):
    microscope.fields[1] = ObjectPoint(height_x_mm=0.3, height_y_mm=-0.2, numerical_aperture=0.2)
    image_x, image_y = analysis.trace_fast(microscope).chief_ray_image_points[1]
    assert (image_x, image_y) == pytest.approx((MAGNIFICATION * 0.3, MAGNIFICATION * -0.2))


def test_the_image_side_na_is_the_object_na_over_the_magnification(microscope):
    spot = analysis.trace_fast(microscope).spot
    # A paraxial lens maps the tangent of a ray's angle, not its sine, so at NA 0.25 the
    # image-side NA is about 3% above NA / |M|.
    assert spot.numerical_aperture == pytest.approx(0.25 / abs(MAGNIFICATION), rel=0.05)


# ------------------------------------------------------------------------ the launcher


def test_an_object_points_rays_start_at_the_point_and_fill_its_cone(microscope):
    point = ObjectPoint(height_x_mm=0.2, height_y_mm=0.5, numerical_aperture=0.3)
    pupil_x, pupil_y = pupil_points("line_y", 5)
    rays = launch_rays(microscope, point, build_optic(microscope), pupil_x, pupil_y)
    assert np.allclose(rays.x, 0.2) and np.allclose(rays.y, 0.5)

    directions = np.stack([rays.L, rays.M, rays.N], axis=1)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    # The middle ray, the chief ray, is aimed at the centre of the first lens.
    distance_mm = microscope.lenses[0].z - microscope.source_z
    aim = np.array([-0.2, -0.5, distance_mm]) / np.linalg.norm([-0.2, -0.5, distance_mm])
    assert directions[2] == pytest.approx(aim)
    # The edge rays are at asin(NA) to it.
    for edge in (0, 4):
        assert np.linalg.norm(np.cross(directions[edge], aim)) == pytest.approx(0.3)


def test_the_side_view_rays_of_an_object_point_start_on_the_object_plane(microscope):
    paths = analysis.trace_fast(microscope).paths[1]
    assert np.allclose(paths.z[0], microscope.source_z)
    assert np.allclose(paths.y[0], microscope.fields[1].height_y_mm)


# ------------------------------------------------------------------- other analyses


@pytest.mark.parametrize("height_mm", [0.0, 5.0])
def test_the_mtf_of_an_object_point_is_that_of_the_same_cone_as_a_beam(height_mm):
    # An ACT508-200-A imaging a point 2f (400 mm) in front of it. A real lens is used
    # because Optiland's wavefront through a paraxial lens is wrong for anything but an
    # on-axis collimated beam.
    model = OpticalModel()
    model.set_source_gap_locked(False)
    model.set_source_gap(400.0)
    numerical_aperture = 0.02
    model.state.fields = [ObjectPoint(height_y_mm=height_mm, numerical_aperture=numerical_aperture)]
    model.autofocus()
    point_mtf = analysis.compute_mtf(model.state)

    # The same light described by hand as a beam at the lens: diverging from 400 mm away,
    # as wide as the cone is there, and aimed from the point at the lens centre.
    distance_mm = 400.0
    model.state.fields = [
        BeamField(
            diameter_mm=2 * distance_mm * np.tan(np.arcsin(numerical_aperture)),
            angle_deg=float(np.degrees(np.arctan2(-height_mm, distance_mm))),
            vergence_dioptres=-1000.0 / distance_mm,
        )
    ]
    model.set_source_gap(0.0)
    beam_mtf = analysis.compute_mtf(model.state)

    assert point_mtf.frequency.size > 0
    assert point_mtf.frequency == pytest.approx(beam_mtf.frequency, rel=1e-6)
    assert point_mtf.tangential == pytest.approx(beam_mtf.tangential, abs=1e-3)
    assert point_mtf.sagittal == pytest.approx(beam_mtf.sagittal, abs=1e-3)


def test_field_curvature_of_the_paraxial_microscope_is_flat(microscope):
    curvature = analysis.compute_field_curvature(microscope)
    assert np.all(np.isfinite(curvature.tangential_shift))
    assert np.abs(curvature.tangential_shift).max() < 1e-6
    assert np.abs(curvature.sagittal_shift).max() < 1e-6


def test_a_finite_object_has_no_f_number(microscope):
    assert analysis.compute_f_number(microscope) is None


def test_an_object_on_the_first_lens_gives_nan_rather_than_an_error(microscope):
    microscope.source_z = microscope.lenses[0].z
    fast_trace = analysis.trace_fast(microscope)
    assert np.isnan(fast_trace.summary.magnification)
    assert np.all(np.isnan(analysis.compute_field_curvature(microscope).tangential_shift))


# ------------------------------------------------------------- switching field kind


def test_switching_to_object_points_keeps_each_chief_ray_angle():
    model = OpticalModel()
    model.set_source_gap_locked(False)
    model.set_source_gap(50.0)
    model.state.fields = [BeamField(angle_deg=2.0, wavelength_nm=650.0, colour="#123456")]
    model.set_finite_object(True)

    point = model.state.fields[0]
    assert isinstance(point, ObjectPoint)
    assert point.height_y_mm == pytest.approx(-50.0 * np.tan(np.radians(2.0)))
    assert (point.wavelength_nm, point.colour) == (650.0, "#123456")
    assert chief_ray_angle_deg(model.state, point) == pytest.approx(2.0)

    model.set_finite_object(False)
    assert model.state.fields[0].angle_deg == pytest.approx(2.0)


def test_switching_with_the_source_on_the_lens_puts_the_object_at_its_front_focus():
    model = OpticalModel()
    model.state.lenses = [make_paraxial_lens(40.0, z=100.0)]
    assert model.state.source_gap_locked and model.state.source_z == 100.0
    model.set_finite_object(True)
    assert not model.state.source_gap_locked
    assert model.state.source_z == pytest.approx(60.0)
    spot = analysis.trace_fast(model.state).spot
    assert np.abs(spot.slope_y).max() < 1e-9  # collimated by the lens


@pytest.mark.parametrize(
    ("focal_length_mm", "expected_source_z"),
    [(-40.0, 100.0 - DEFAULT_OBJECT_DISTANCE_MM), (400.0, AXIS_START_Z)],
)
def test_an_object_without_a_front_focus_in_reach_goes_where_there_is_room(
    focal_length_mm, expected_source_z
):
    model = OpticalModel()
    model.state.lenses = [make_paraxial_lens(focal_length_mm, z=100.0)]
    model.set_finite_object(True)
    assert model.state.source_z == pytest.approx(expected_source_z)


def test_switching_to_the_kind_a_design_already_has_changes_nothing(microscope):
    before = copy.deepcopy(microscope)
    set_finite_object(microscope, True)
    assert microscope == before


# ----------------------------------------------------------------------- files


def test_a_finite_object_survives_a_project_save_and_load(microscope, tmp_path):
    microscope.fields[1].colour = None
    project_path = tmp_path / "microscope.raycrafter"
    save_project(microscope, project_path)
    assert load_project(project_path) == microscope
    assert json.loads(project_path.read_text())["fields"][0]["kind"] == "object_point"


def test_a_finite_object_crosses_the_agent_link(microscope):
    assert state_from_dict(json.loads(json.dumps(state_to_dict(microscope)))) == microscope


def test_a_version_1_project_loads_with_its_fields_as_beams():
    data = project_to_dict(OpticalModel().state)
    data["format_version"] = 1
    for field_data in data["fields"]:
        del field_data["kind"]
    assert isinstance(project_from_dict(data).fields[0], BeamField)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data["fields"][0].update(kind="laser"), "kind 'laser'"),
        (lambda data: data["fields"][0].update(numerical_aperture=1.5), "between 0 and 1"),
        (
            lambda data: data["fields"].append(project_to_dict(OpticalModel().state)["fields"][0]),
            "mixes beams and object points",
        ),
    ],
)
def test_bad_object_points_are_refused(microscope, change, message):
    data = project_to_dict(microscope)
    change(data)
    with pytest.raises(ProjectFileError, match=message):
        project_from_dict(data)


def test_a_finite_object_is_exported_to_zmx_at_its_distance(microscope, tmp_path):
    zmx_path = tmp_path / "microscope.zmx"
    save_zmx(microscope, str(zmx_path))
    # The importer describes a finite object as a diverging beam at the first lens.
    reloaded_model = OpticalModel()
    reloaded_model.load_zmx(str(zmx_path))
    reloaded = reloaded_model.state
    distance_mm = microscope.lenses[0].z - microscope.source_z
    assert reloaded.fields[0].vergence_dioptres == pytest.approx(-1000.0 / distance_mm)


# ------------------------------------------------------------ Python API and agent tools


def test_the_python_api_edits_and_reports_a_finite_object():
    bench = Bench()
    bench.new_design(discard_unsaved=True, example="Basic microscope")
    field = bench.set_field(1, height_y_mm=-0.25, numerical_aperture=0.1)
    assert (field.kind, field.height_y_mm, field.numerical_aperture) == ("object_point", -0.25, 0.1)
    assert field.diameter_mm is None

    summary = bench.summary()
    assert summary.magnification == pytest.approx(MAGNIFICATION)
    assert summary.f_number is None
    assert summary.chief_ray_image_mm[1][1] == pytest.approx(MAGNIFICATION * -0.25)

    with pytest.raises(ValueError, match="no diameter_mm"):
        bench.set_field(0, diameter_mm=5.0)
    assert [field.kind for field in bench.set_finite_object(False)] == ["beam", "beam"]


def test_the_agent_tools_report_each_fields_kind():
    tools = DesignTools(OpticalModel())
    design = tools.set_finite_object(True)
    assert design["fields"][0]["kind"] == "object_point"
    assert "numerical_aperture" in design["fields"][0]
    with pytest.raises(ValueError, match="no example"):
        tools.new_design(discard_unsaved=True, example="Telescope")
