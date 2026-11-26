"""The bench as 3D geometry, shared by the app's 3D view and the HTML report (#52, #51).

``build_scene`` turns an ``OpticalState`` into a list of ``SceneObject``: one for each lens,
one for each field's rays, the source and image planes, and the optical axis. Each object is
nothing but triangle meshes and polylines with a colour, so a viewer only has to draw those
and knows no optics. It has no Qt code.

Coordinates are the bench's, in mm: z along the optical axis (the way the light goes), y up
in the side view and x towards the viewer's side of it, a right-handed system. Each viewer
turns these into its own idea of "up".

The data format
---------------
``scene_to_dict`` gives the scene as plain JSON-ready data, which the report embeds and its
viewer (``raycrafter/report/assets/viewer3d.js``) reads::

    {
        "format": "raycrafter-scene",
        "version": 1,
        "units": "mm",
        "objects": [
            {
                "name": "ACT508-200-A",
                "kind": "lens",
                "colour": "#3b7dd8",
                "opacity": 0.18,
                "meshes": [{"vertices": [x, y, z, ...], "faces": [i, j, k, ...]}],
                "lines": [[x, y, z, ...], ...],
            },
            ...,
        ],
    }

``kind`` is "lens", "rays", "plane" or "axis"; ``opacity`` is that of the meshes (lines are
drawn solid); a face is three vertex numbers, counted from 0 within its mesh. Change
SCENE_VERSION if this layout changes in a way an older viewer cannot read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from raycrafter import palette
from raycrafter.core.analysis import RayPaths, trace_ray_bundles
from raycrafter.core.plot_geometry import field_name, source_caption
from raycrafter.core.prescription import surface_sag
from raycrafter.core.state import AXIS_START_Z, OpticalState, PlacedLens

SCENE_FORMAT = "raycrafter-scene"
SCENE_VERSION = 1

LENS = "lens"
RAYS = "rays"
PLANE = "plane"
AXIS = "axis"

AROUND_STEPS = 64  # a circle about the axis is drawn as this many straight pieces
RADIAL_STEPS = 12  # rings of triangles from the middle of a lens surface to its rim
RAY_RINGS = 3  # each field shows its central ray and this many rings of rays about it
LENS_OPACITY = 0.18  # surfaces overlap on screen, so a lens looks several times as dense
PLANE_OPACITY = 0.12
MINIMUM_PLANE_RADIUS_MM = 12.7  # the source and image planes are at least this big
DECIMALS = 3  # coordinates in the data format are rounded to 1 um

# The viewers' standard views, as the camera's (azimuth, elevation) in degrees. The camera
# looks at the middle of the scene; azimuth is measured in the horizontal plane from the
# direction the light travels, and -90 is the side the main window's side view is seen from.
STANDARD_VIEWS = {
    "3D": (-115.0, 20.0),
    "Side": (-90.0, 0.0),
    "Top": (-90.0, 90.0),
    "End": (180.0, 0.0),  # along the axis, from the source's end
}


@dataclass(frozen=True)
class Mesh:
    """A surface made of triangles.

    Attributes
    ----------
    vertices : numpy.ndarray
        Shape (n_vertices, 3): x, y, z in mm.
    faces : numpy.ndarray
        Shape (n_faces, 3): the three vertex numbers of each triangle, ordered
        anticlockwise seen from outside, so that the normals point outwards.
    """

    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class SceneObject:
    """One thing in the scene: what it is called, how it looks, and its geometry.

    Attributes
    ----------
    name : str
        E.g. a part number, "Field 0" or "image".
    kind : str
        LENS, RAYS, PLANE or AXIS.
    colour : str
        Hex colour of its meshes and lines.
    opacity : float
        Opacity (0 to 1) of its meshes; lines are drawn solid.
    meshes : tuple of Mesh
        Its surfaces, e.g. one closed solid for each glass element of a lens.
    lines : tuple of numpy.ndarray
        Polylines, each of shape (n_points, 3): a lens's rims, or a field's rays.
    """

    name: str
    kind: str
    colour: str
    opacity: float = 1.0
    meshes: tuple[Mesh, ...] = ()
    lines: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True)
class Scene:
    """Everything the 3D views draw."""

    objects: tuple[SceneObject, ...]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """The smallest and largest x, y, z (mm) of anything in the scene."""
        points = [mesh.vertices for item in self.objects for mesh in item.meshes]
        points += [line for item in self.objects for line in item.lines if len(line)]
        everything = np.concatenate(points)
        return everything.min(axis=0), everything.max(axis=0)


# ------------------------------------------------------------------------- shapes


def circle(radius_mm: float, z_mm: float) -> np.ndarray:
    """A closed circle about the axis at ``z_mm``: AROUND_STEPS + 1 points (x, y, z)."""
    angles = np.linspace(0.0, 2.0 * np.pi, AROUND_STEPS + 1)
    return np.column_stack(
        [radius_mm * np.cos(angles), radius_mm * np.sin(angles), np.full(angles.size, z_mm)]
    )


def _revolve(profile_radius: np.ndarray, profile_z: np.ndarray) -> Mesh:
    """The surface swept out by turning a profile of (radius, z) points about the axis.

    The normals point outwards for a profile that goes from the middle of a front surface,
    out to the rim, across to the back surface and in to its middle: that is, towards -z
    where the profile moves away from the axis.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, AROUND_STEPS, endpoint=False)
    # One ring of AROUND_STEPS vertices for each point of the profile. A point on the axis
    # gives a ring of coincident vertices, which draws as the middle of a fan.
    x = np.outer(profile_radius, np.cos(angles))
    y = np.outer(profile_radius, np.sin(angles))
    z = np.repeat(profile_z[:, np.newaxis], AROUND_STEPS, axis=1)
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    numbers = np.arange(vertices.shape[0]).reshape(profile_radius.size, AROUND_STEPS)
    next_around = np.roll(numbers, -1, axis=1)
    faces = []
    for ring in range(profile_radius.size - 1):
        here, here_next = numbers[ring], next_around[ring]
        there, there_next = numbers[ring + 1], next_around[ring + 1]
        # The quadrilateral between two rings as two triangles; on the axis one of the two
        # has no area and is left out.
        if profile_radius[ring] > 0.0:
            faces.append(np.column_stack([here, here_next, there_next]))
        if profile_radius[ring + 1] > 0.0:
            faces.append(np.column_stack([here, there_next, there]))
    return Mesh(vertices, np.concatenate(faces))


def disc(radius_mm: float, z_mm: float) -> Mesh:
    """A flat disc across the axis at ``z_mm``."""
    radii = np.linspace(0.0, radius_mm, 2)
    return _revolve(radii, np.full(radii.size, z_mm))


def element_meshes(lens: PlacedLens) -> list[Mesh]:
    """A catalogue lens as solids, one for each of its glass elements.

    Each solid is bounded by its two surfaces, out to the clear aperture, and the cylinder
    joining their rims. The cemented face of a doublet therefore belongs to both solids.
    """
    radii = np.linspace(0.0, lens.semi_aperture, RADIAL_STEPS + 1)
    surfaces = lens.surfaces_in_use()
    meshes = []
    vertex_z = lens.z
    for front, back in zip(surfaces[:-1], surfaces[1:]):
        back_vertex_z = vertex_z + front.thickness_after
        if front.material_after != "air":
            front_z = vertex_z + surface_sag(front.radius, radii)
            back_z = back_vertex_z + surface_sag(back.radius, radii)
            # The three parts are meshed separately, so the rims stay sharp when shaded.
            parts = [
                _revolve(radii, front_z),
                _revolve(np.full(2, lens.semi_aperture), np.array([front_z[-1], back_z[-1]])),
                _revolve(radii[::-1], back_z[::-1]),
            ]
            meshes.append(join_meshes(parts))
        vertex_z = back_vertex_z
    return meshes


def join_meshes(meshes: list[Mesh]) -> Mesh:
    """Several meshes as one."""
    offsets = np.cumsum([0] + [len(mesh.vertices) for mesh in meshes[:-1]])
    return Mesh(
        np.concatenate([mesh.vertices for mesh in meshes]),
        np.concatenate([mesh.faces + offset for mesh, offset in zip(meshes, offsets)]),
    )


def lens_object(lens: PlacedLens) -> SceneObject:
    """One lens: its glass elements and the rim of every surface.

    A paraxial (ideal thin) lens has no thickness: it is drawn as a flat disc.
    """
    colour = palette.LENS if lens.assembly_id is None else palette.ASSEMBLY
    if lens.is_paraxial:
        meshes = [disc(lens.semi_aperture, lens.z)]
        rim_zs = [lens.z]
    else:
        meshes = element_meshes(lens)
        rim_zs = []
        vertex_z = lens.z
        for surface in lens.surfaces_in_use():
            rim_sag = surface_sag(surface.radius, np.array(lens.semi_aperture))
            rim_zs.append(vertex_z + float(rim_sag))
            vertex_z += surface.thickness_after
    return SceneObject(
        name=lens.part_number,
        kind=LENS,
        colour=colour,
        opacity=LENS_OPACITY,
        meshes=tuple(meshes),
        lines=tuple(circle(lens.semi_aperture, rim_z) for rim_z in rim_zs),
    )


def ray_lines(paths: RayPaths) -> tuple[np.ndarray, ...]:
    """Each ray of one field as a polyline, ending where the ray is blocked."""
    lines = []
    for ray_index in range(paths.z.shape[1]):
        points = np.column_stack(
            [paths.x[:, ray_index], paths.y[:, ray_index], paths.z[:, ray_index]]
        )
        blocked_from = np.flatnonzero(~np.isfinite(points).all(axis=1))
        if blocked_from.size:
            points = points[: blocked_from[0]]
        if len(points) > 1:
            lines.append(points)
    return tuple(lines)


def _plane_object(name: str, colour: str, z_mm: float, radius_mm: float) -> SceneObject:
    """The source or image plane, as a faint disc with a rim."""
    return SceneObject(
        name=name,
        kind=PLANE,
        colour=colour,
        opacity=PLANE_OPACITY,
        meshes=(disc(radius_mm, z_mm),),
        lines=(circle(radius_mm, z_mm),),
    )


# ------------------------------------------------------------------------- the scene


def build_scene(state: OpticalState) -> Scene:
    """Trace the bench and return everything the 3D views draw. The state is not changed."""
    lens_radius_mm = max(
        (lens.semi_aperture for lens in state.lenses), default=MINIMUM_PLANE_RADIUS_MM
    )
    plane_radius_mm = max(lens_radius_mm, MINIMUM_PLANE_RADIUS_MM)
    objects = [lens_object(lens) for lens in state.lenses]

    image_reach_mm = 0.0  # how far from the axis the rays meet the image plane
    for field_index, (light, paths) in enumerate(
        zip(state.fields, trace_ray_bundles(state, RAY_RINGS))
    ):
        end_radius = np.hypot(paths.x[-1], paths.y[-1])  # NaN for a blocked ray
        image_reach_mm = max(image_reach_mm, *end_radius[np.isfinite(end_radius)], 0.0)
        objects.append(
            SceneObject(
                name=field_name(field_index),
                kind=RAYS,
                colour=light.display_colour(),
                lines=ray_lines(paths),
            )
        )

    # A source plane lying on the first lens (the default for beams) would only hide it.
    source_on_first_lens = bool(state.lenses) and state.source_z == state.lenses[0].z
    if state.has_finite_object or not source_on_first_lens:
        objects.append(
            _plane_object(source_caption(state), palette.SOURCE, state.source_z, plane_radius_mm)
        )
    # The image plane widens to hold the rays, unless they spread far beyond the lenses (as
    # from a diverging system), when a huge disc would hide everything else.
    image_radius_mm = max(plane_radius_mm, min(1.1 * image_reach_mm, 3.0 * plane_radius_mm))
    objects.append(_plane_object("image", palette.IMAGE, state.image_z, image_radius_mm))

    axis_end_z = max(state.image_z, state.last_back_z())
    axis_line = np.array([[0.0, 0.0, AXIS_START_Z], [0.0, 0.0, axis_end_z]])
    objects.append(
        SceneObject(name="optical axis", kind=AXIS, colour=palette.AXIS_LINE, lines=(axis_line,))
    )
    return Scene(tuple(objects))


def scene_to_dict(scene: Scene) -> dict:
    """The scene as JSON-ready data, in the format the module docstring describes."""

    def flat(array: np.ndarray) -> list:
        # "+ 0.0" turns a rounded -0.0 into 0.0, which is shorter and tidier.
        return (np.round(array, DECIMALS) + 0.0).ravel().tolist()

    return {
        "format": SCENE_FORMAT,
        "version": SCENE_VERSION,
        "units": "mm",
        "objects": [
            {
                "name": item.name,
                "kind": item.kind,
                "colour": item.colour,
                "opacity": item.opacity,
                "meshes": [
                    {"vertices": flat(mesh.vertices), "faces": mesh.faces.ravel().tolist()}
                    for mesh in item.meshes
                ],
                "lines": [flat(line) for line in item.lines],
            }
            for item in scene.objects
        ],
    }
