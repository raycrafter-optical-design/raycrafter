"""The two kinds of field, and switching a design between them.

A design's fields are either beams (``BeamField``: an object at infinity, each field an angle
and a width) or the points of a finite object at the source plane (``ObjectPoint``: each
field a height and an NA). All the fields of a design are of one kind, as in Zemax, so the
switch (``set_finite_object``) converts every field at once.
"""

from __future__ import annotations

import numpy as np

from raycrafter.core.optic_builder import build_optic
from raycrafter.core.state import AXIS_START_Z, AnyField, BeamField, ObjectPoint, OpticalState

# Where a finite object goes, in front of the first lens, when that lens has no front focal
# point to put it at (a diverging lens, or a flat plate).
DEFAULT_OBJECT_DISTANCE_MM = 50.0


def source_to_lens_mm(state: OpticalState) -> float:
    """Distance (mm) from the source plane (a finite object's plane) to the first lens."""
    return state.first_surface_z() - state.source_z


def field_height_mm(light: AnyField) -> float:
    """How far from the axis (mm) an object point is; 0 for a beam, which has no height."""
    if isinstance(light, BeamField):
        return 0.0
    return float(np.hypot(light.height_x_mm, light.height_y_mm))


def chief_ray_angle_deg(state: OpticalState, light: AnyField) -> float:
    """The angle (degrees) of a field's chief ray to the axis, in front of the first lens.

    For a beam this is its angle. For an object point it is the angle of the line from the
    point to the centre of the first lens (its chief ray, see ``ray_launch``). Its size comes
    from the point's distance from the axis, and its sign from the height in y: positive
    (rising) for a point below the axis. It is 0 for a point on the lens's vertex plane.
    """
    if isinstance(light, BeamField):
        return light.angle_deg
    distance_mm = source_to_lens_mm(state)
    if distance_mm <= 0.0:
        return 0.0
    height_mm = field_height_mm(light)
    angle_deg = float(np.degrees(np.arctan2(height_mm, distance_mm)))
    return -angle_deg if light.height_y_mm > 0.0 else angle_deg


def set_finite_object(state: OpticalState, finite: bool) -> None:
    """Make every field an object point (``finite``) or a beam, keeping its chief ray.

    Each field keeps its wavelength and colour, and its chief ray keeps its angle (see
    ``chief_ray_angle_deg``): a beam at angle a becomes a point at height -d tan(a), d in
    front of the first lens, with the default NA; a point becomes a beam of the default
    width. Other properties (a beam's offset and vergence, a point's x height) have no
    counterpart and are not kept. Nothing changes if the fields are already of the asked kind.

    An object on the first lens would make no image, so when switching to object points with
    the source plane there (the default, a locked gap of 0 mm), the source gap is first
    unlocked and the source plane moved forward (see ``place_object_in_front``).
    """
    if finite == state.has_finite_object:
        return
    if finite and state.lenses and source_to_lens_mm(state) <= 0.0:
        place_object_in_front(state)
    distance_mm = source_to_lens_mm(state)
    converted: list[AnyField] = []
    for light in state.fields:
        angle_deg = chief_ray_angle_deg(state, light)
        if finite:
            height_mm = -distance_mm * float(np.tan(np.radians(angle_deg)))
            converted.append(
                ObjectPoint(
                    height_y_mm=height_mm,
                    wavelength_nm=light.wavelength_nm,
                    colour=light.colour,
                )
            )
        else:
            converted.append(
                BeamField(
                    angle_deg=angle_deg, wavelength_nm=light.wavelength_nm, colour=light.colour
                )
            )
    state.fields = converted


def place_object_in_front(state: OpticalState) -> None:
    """Unlock the source gap and move the source plane to the first lens's front focal plane.

    Light from an object there leaves the first lens collimated, as from an infinity-corrected
    microscope objective. A first lens with no front focal point in front of it (a diverging
    lens or a flat plate) gets the object DEFAULT_OBJECT_DISTANCE_MM in front instead. Either
    way the source plane stops at the start of the axis if there is not room.
    """
    first_lens = state.lenses[0]
    alone = OpticalState(lenses=[first_lens], image_z=first_lens.back_z + 10.0)
    # Optiland gives the front focal point relative to the lens's front vertex.
    with np.errstate(divide="ignore", invalid="ignore"):
        front_focal_distance_mm = -float(build_optic(alone).paraxial.F1())
    if not (np.isfinite(front_focal_distance_mm) and front_focal_distance_mm > 0.0):
        front_focal_distance_mm = DEFAULT_OBJECT_DISTANCE_MM
    state.source_gap_locked = False
    state.source_z = max(first_lens.z - front_focal_distance_mm, AXIS_START_Z)
