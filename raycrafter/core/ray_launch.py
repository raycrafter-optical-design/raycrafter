"""The rays of each field (beam), generated here and handed to Optiland to trace.

Optiland can generate rays itself from an object, an entrance pupil and a list of field
angles, but all of those belong to the whole system. Our fields each have their own
diameter, angle, vergence, offset and wavelength, so instead we build the Optiland system
once and launch every field's rays ourselves. This is also fast: a trace costs well under a
millisecond, a system build about 8 ms. The MTF is the exception: it needs Optiland's own
pupil model (see ``optic_builder.build_optic_for_beam``).

Every ray starts at the start of the axis (``AXIS_START_Z``), so the side view can draw it
from there. Its height and slope are worked out from where it crosses the beam's reference
plane (the first lens; see ``BeamField``):

- a ray at pupil point (px, py), normalised to the beam radius r, crosses the reference
  plane at x = px r, y = offset + py r;
- its slope is the beam angle plus the vergence: dy/dz = tan(angle) - V py r and
  dx/dz = -V px r, with V in 1/mm. A converging beam (V > 0) thus aims at a point 1000/V mm
  beyond the reference plane, which is how a real point object would light the lens.
"""

from __future__ import annotations

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

    # Step back along each ray to the start of the axis.
    distance_back_mm = state.beam_reference_z() - AXIS_START_Z
    start_x = x_at_reference - slope_x * distance_back_mm
    start_y = y_at_reference - slope_y * distance_back_mm
    reference_local_z = float(optic.surfaces[1].geometry.cs.z)
    start_z = np.full(pupil_x.shape, reference_local_z - distance_back_mm)

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


def trace_rays(optic: Optic, rays: RealRays) -> RealRays:
    """Trace launched rays through the system to the image plane, and return them.

    Optiland keeps the per-surface positions of the most recent trace on its surfaces (read
    by ``analysis._trace_ray_paths``), so read those before tracing anything else.
    """
    optic.surfaces.trace(rays)
    return rays
