"""The bench as 3D geometry (raycrafter.core.scene), for the 3D views (issues #51, #52)."""

import copy
import json

import numpy as np
import pytest

from raycrafter.core import OpticalModel, scene
from raycrafter.core.prescription import surface_sag


def objects_of(built: scene.Scene, kind: str) -> list[scene.SceneObject]:
    return [item for item in built.objects if item.kind == kind]


def signed_volume_mm3(mesh: scene.Mesh) -> float:
    """Volume inside a closed mesh; positive if the triangles' normals point outwards."""
    a, b, c = (mesh.vertices[mesh.faces[:, corner]] for corner in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def test_the_scene_has_every_lens_field_and_plane():
    model = OpticalModel()
    model.add_lens("LA1509-A")
    model.add_field()
    built = model.build_scene()
    assert [item.name for item in objects_of(built, scene.LENS)] == ["ACT508-200-A", "LA1509-A"]
    assert [item.name for item in objects_of(built, scene.RAYS)] == ["Field 0", "Field 1"]
    assert [item.name for item in objects_of(built, scene.PLANE)] == ["image"]
    assert len(objects_of(built, scene.AXIS)) == 1


def test_the_source_plane_is_shown_only_away_from_the_first_lens():
    model = OpticalModel()
    model.set_source_gap_locked(False)
    model.set_source_z(50.0)
    source, image = objects_of(model.build_scene(), scene.PLANE)
    assert (source.name, image.name) == ("source", "image")
    assert source.meshes[0].vertices[:, 2] == pytest.approx(50.0)


def test_building_a_scene_does_not_change_the_bench():
    model = OpticalModel()
    state_before = copy.deepcopy(model.state)
    model.build_scene()
    assert model.state == state_before


@pytest.mark.parametrize(("part_number", "element_count"), [("ACT508-200-A", 2), ("LA1509-A", 1)])
@pytest.mark.parametrize("flipped", [False, True])
def test_each_glass_element_is_a_solid_with_outward_normals(part_number, element_count, flipped):
    lens = OpticalModel().make_lens(part_number, 100.0)
    lens.flipped = flipped
    meshes = scene.element_meshes(lens)
    assert len(meshes) == element_count
    assert all(signed_volume_mm3(mesh) > 0 for mesh in meshes)
    points = np.concatenate([mesh.vertices for mesh in meshes])
    assert np.hypot(points[:, 0], points[:, 1]).max() == pytest.approx(lens.semi_aperture)


def test_a_plano_convex_lens_has_the_volume_of_a_cap_on_a_disc():
    """LA1509-A: a spherical cap (R = 51.51 mm) in front of nothing but its centre thickness."""
    lens = OpticalModel().make_lens("LA1509-A", 100.0)
    radius, semi_aperture, thickness = 51.51, lens.semi_aperture, lens.thickness
    rim_sag = float(surface_sag(radius, np.array(semi_aperture)))
    cap = np.pi * rim_sag**2 * (3 * radius - rim_sag) / 3
    # The glass is the cylinder to the flat back face, less what lies in front of the sphere.
    expected = np.pi * semi_aperture**2 * thickness - (np.pi * semi_aperture**2 * rim_sag - cap)
    (mesh,) = scene.element_meshes(lens)
    assert signed_volume_mm3(mesh) == pytest.approx(expected, rel=0.01)


def test_the_rays_of_a_collimated_beam_meet_at_the_focus(focused_model):
    """The 3D bundle of a 10 mm beam through the f = 200 mm doublet closes to a few um."""
    (rays,) = objects_of(focused_model.build_scene(), scene.RAYS)
    assert len(rays.lines) == 37  # the central ray and three rings
    starts = np.array([line[0] for line in rays.lines])
    ends = np.array([line[-1] for line in rays.lines])
    assert np.hypot(starts[:, 0], starts[:, 1]).max() == pytest.approx(5.0)
    assert np.hypot(ends[:, 0], ends[:, 1]).max() < 0.02
    assert ends[:, 2] == pytest.approx(focused_model.state.image_z)


def test_the_3d_rays_agree_with_the_side_view(focused_model):
    """The central ray of the 3D bundle is the side view's middle ray."""
    focused_model.state.selected_field.angle_deg = 3.0
    (rays,) = objects_of(focused_model.build_scene(), scene.RAYS)
    paths = focused_model.trace_fast().paths[0]
    central, middle = rays.lines[0], paths.y.shape[1] // 2
    assert central[:, 0] == pytest.approx(0.0, abs=1e-9)
    assert central[:, 1] == pytest.approx(paths.y[:, middle])
    assert central[:, 2] == pytest.approx(paths.z[:, middle])


def test_a_blocked_ray_ends_at_the_lens_that_blocks_it():
    model = OpticalModel()
    model.state.selected_field.diameter_mm = 80.0  # wider than the 50.8 mm lens
    (rays,) = objects_of(model.build_scene(), scene.RAYS)
    blocked = [line for line in rays.lines if line[-1, 2] < model.state.image_z - 1.0]
    assert blocked
    semi_aperture = model.state.lenses[0].semi_aperture
    for line in blocked:
        assert np.isfinite(line).all()
        assert np.hypot(*line[-1, :2]) > semi_aperture


def test_paraxial_lenses_and_assemblies_are_shown():
    model = OpticalModel(add_default_lens=False)
    model.insert_lens(model.make_lens("AC254-050-A", 100.0))
    model.make_plossl(0)
    model.insert_lens(model.make_paraxial_lens(100.0, 200.0))
    first, second, thin = objects_of(model.build_scene(), scene.LENS)
    assert first.colour == second.colour != thin.colour
    assert thin.meshes[0].vertices[:, 2] == pytest.approx(200.0)


def test_an_empty_bench_and_a_finite_object_give_a_scene():
    empty = OpticalModel(add_default_lens=False).build_scene()
    assert objects_of(empty, scene.LENS) == []

    model = OpticalModel()
    model.set_finite_object(True)
    assert objects_of(model.build_scene(), scene.PLANE)[0].name == "object"


def test_the_data_format_is_plain_json(focused_model):
    focused_model.state.selected_field.diameter_mm = 80.0  # some rays are blocked
    data = scene.scene_to_dict(focused_model.build_scene())
    text = json.dumps(data, allow_nan=False)  # NaN is not JSON
    assert json.loads(text)["format"] == "raycrafter-scene"
    for item in data["objects"]:
        for mesh in item["meshes"]:
            vertex_count = len(mesh["vertices"]) / 3
            assert len(mesh["faces"]) % 3 == 0 and 0 <= max(mesh["faces"]) < vertex_count
        assert all(len(line) % 3 == 0 for line in item["lines"])
