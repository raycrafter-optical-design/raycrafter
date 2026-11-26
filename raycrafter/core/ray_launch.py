"""The rays of each field (beam), generated here and handed to Optiland to trace.

Optiland can generate rays itself from an object, an entrance pupil and a list of field
angles, but all of those belong to the whole system. Our fields each have their own
diameter, angle, vergence, offset and wavelength, so instead we build the Optiland system
once and launch every field's rays ourselves. This is also fast: all the fields can be traced
together (see ``join_rays``). The MTF is the exception: it needs Optiland's own pupil model
(see ``optic_builder.build_optic_for_beam``).

Every ray starts at the start of the axis (``AXIS_START_Z``), so the side view can draw it
from there. Its height and slope are worked out from where it crosses the beam's reference
plane (the source plane, ``OpticalState.source_z``; see ``BeamField``):

- a ray at pupil point (px, py), normalised to the beam radius r, crosses the reference
  plane at x = px r, y = offset + py r;
- its slope is the beam angle plus the vergence: dy/dz = tan(angle) - V py r and
  dx/dz = -V px r, with V in 1/mm. A converging beam (V > 0) thus aims at a point 1000/V mm
  beyond the reference plane, which is how a real point object would light the lens.

``beam_at_first_lens`` gives the same beam described at the first lens instead, for the
analyses that use Optiland's own object and pupil there.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from optiland.distribution import create_distribution
from optiland.optic import Optic
from optiland.rays import RealRays

from raycrafter.core.state import AXIS_START_Z, BeamField, OpticalState


def pupil_points(distribution: str, ray_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised pupil coordinates (each from -1 to 1) of a pattern of rays across a beam.

    Parameters
    ----------
    distribution : str
        ``"hexapolar"`` (rings, for spot diagrams), ``"line_y"`` (a fan in the plane of the
        side view) or ``"line_x"`` (a fan across it).
    ray_count : int
        Number of rings for ``"hexapolar"``; number of rays for a line. Lines are ordered
        from -1 to +1, so with an odd count the middle ray is the beam's central ray.

    Returns
    -------
    tuple of numpy.ndarray
        The x and y pupil coordinates.
    """
    if distribution == "line_y":
        return np.zeros(ray_count), np.linspace(-1.0, 1.0, ray_count)
    if distribution == "line_x":
        return np.linspace(-1.0, 1.0, ray_count), np.zeros(ray_count)
    points = create_distribution(distribution)
    points.generate_points(ray_count)
    return np.asarray(points.x, dtype=float), np.asarray(points.y, dtype=float)


def launch_rays(
    state: OpticalState,
    beam: BeamField,
    optic: Optic,
    pupil_x: np.ndarray,
    pupil_y: np.ndarray,
) -> RealRays:
    """Optiland rays for one field, starting at the start of the axis.

    Parameters
    ----------
    state : OpticalState
        The bench; it gives the position of the beam's reference plane.
    beam : BeamField
        The beam to launch.
    optic : optiland.optic.Optic
        The system built from ``state`` (``optic_builder.build_optic``). Optiland's local z
        has its own origin, so the first surface's local z is used to place the rays.
    pupil_x, pupil_y : numpy.ndarray
        Normalised pupil coordinates of the rays (see ``pupil_points``).

    Returns
    -------
    optiland.rays.RealRays
        Rays ready for ``optic.surfaces.trace``.
    """
    radius_mm = beam.diameter_mm / 2.0
    vergence_per_mm = beam.vergence_dioptres / 1000.0

    # Where each ray crosses the reference plane, and its slopes there.
    x_at_reference = pupil_x * radius_mm
    y_at_reference = beam.offset_mm + pupil_y * radius_mm
    slope_x = -vergence_per_mm * pupil_x * radius_mm
    slope_y = np.tan(np.radians(beam.angle_deg)) - vergence_per_mm * pupil_y * radius_mm

    # Step back along each ray to the start of the axis. Optiland's z is anchored on its
    # first surface, which is at state.first_surface_z() on the bench.
    distance_back_mm = state.beam_reference_z() - AXIS_START_Z
    start_x = x_at_reference - slope_x * distance_back_mm
    start_y = y_at_reference - slope_y * distance_back_mm
    first_surface_local_z = float(optic.surfaces[1].geometry.cs.z)
    start_local_z = first_surface_local_z - (state.first_surface_z() - AXIS_START_Z)
    start_z = np.full(pupil_x.shape, start_local_z)

    # Direction cosines: the slope vector (dx/dz, dy/dz, 1), normalised.
    length = np.sqrt(slope_x**2 + slope_y**2 + 1.0)
    ray_count = pupil_x.size
    return RealRays(
        x=start_x,
        y=start_y,
        z=start_z,
        L=slope_x / length,
        M=slope_y / length,
        N=1.0 / length,
        intensity=np.ones(ray_count),
        wavelength=np.full(ray_count, beam.wavelength_um),
    )


def beam_at_first_lens(state: OpticalState, beam: BeamField) -> BeamField:
    """The same beam, described where it reaches the first lens instead of the source plane.

    Over the distance d from the source plane to the lens, a ray at pupil height p r rises
    by its slope, tan(a) - V p r, times d. So at the lens the centre is d tan(a) higher, the
    beam is |1 - V d| times as wide, and its vergence is V / (1 - V d): a beam converging on
    a point 1/V beyond the source is converging on one 1/V - d beyond the lens. A beam
    focused exactly on the lens has no width there.

    Returns
    -------
    BeamField
        A copy of ``beam`` with its diameter, vergence and offset at the first lens (the
        beam itself if the source plane is at the first lens).
    """
    distance_mm = state.first_surface_z() - state.source_z
    if distance_mm == 0.0:
        return beam
    vergence_per_mm = beam.vergence_dioptres / 1000.0
    width_factor = 1.0 - vergence_per_mm * distance_mm
    if width_factor == 0.0:
        return replace(beam, diameter_mm=0.0)
    return replace(
        beam,
        diameter_mm=beam.diameter_mm * abs(width_factor),
        vergence_dioptres=beam.vergence_dioptres / width_factor,
        offset_mm=beam.offset_mm + distance_mm * float(np.tan(np.radians(beam.angle_deg))),
    )


def join_rays(bundles: list[RealRays]) -> RealRays:
    """One set of rays holding all those of ``bundles``, in order, so one trace does for all.

    A trace costs much the same for a few rays as for hundreds, so tracing the bundles
    together is far faster than one at a time. Each ray keeps its own wavelength.
    """
    names = ("x", "y", "z", "L", "M", "N", "i", "w")
    joined = {name: np.concatenate([getattr(rays, name) for rays in bundles]) for name in names}
    return RealRays(
        x=joined["x"],
        y=joined["y"],
        z=joined["z"],
        L=joined["L"],
        M=joined["M"],
        N=joined["N"],
        intensity=joined["i"],
        wavelength=joined["w"],
    )


def trace_rays(optic: Optic, rays: RealRays) -> RealRays:
    """Trace launched rays through the system to the image plane, and return them.

    Optiland keeps the per-surface positions of the most recent trace on its surfaces (read
    by ``analysis._ray_paths``), so read those before tracing anything else.
    """
    optic.surfaces.trace(rays)
    return rays
