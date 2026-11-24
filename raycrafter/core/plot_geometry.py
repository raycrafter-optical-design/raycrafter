"""Geometry for drawing the bench, shared by the GUI and the HTML report.

These functions turn model data into plain arrays to draw (lens outlines, rays, plot
ranges). They know nothing about Qt or matplotlib, so both front ends draw the same
picture.
"""

from __future__ import annotations

import numpy as np

from raycrafter.core.analysis import FNumberResult, RayPaths, SpotResult, airy_radius
from raycrafter.core.prescription import surface_sag
from raycrafter.core.state import OpticalState, PlacedLens

AXIS_Z_RANGE = (0.0, 450.0)  # mm of optical axis shown in the side view
AXIS_Y_RANGE = (-32.0, 32.0)  # mm either side of the axis shown in the side view
MIN_SPOT_HALF_WIDTH_UM = 5.0  # the spot plot never zooms in beyond +/-5 um


def lens_surface_arcs(lens: PlacedLens, point_count: int = 60) -> tuple[np.ndarray, list]:
    """The curve of every surface of a lens, in its current position and orientation.

    Returns
    -------
    edge_y : numpy.ndarray
        Heights from -semi_aperture to +semi_aperture (mm).
    surface_arcs_z : list of numpy.ndarray
        For each surface, front to back, the z of the surface at each height in ``edge_y``.
        The first and last arcs bound the glass body; the ones between are cemented faces.
    """
    edge_y = np.linspace(-lens.semi_aperture, lens.semi_aperture, point_count)
    surface_arcs_z = []
    vertex_z = lens.z
    for surface in lens.surfaces_in_use():
        surface_arcs_z.append(vertex_z + surface_sag(surface.radius, edge_y))
        vertex_z += surface.thickness_after
    return edge_y, surface_arcs_z


def rays_from_axis_start(state: OpticalState, paths: RayPaths) -> list[tuple]:
    """Each traced side-view ray, extended back to the start of the axis.

    Optiland's ray paths begin at the first lens. To draw the incoming beam, each ray is
    extended back to the left edge of the side view along its incoming direction. That slope
    is the beam tilt plus the vergence: a ray at height h is bent by -V * h (V in 1/mm), so
    converging beams (V > 0) are wider before the lens than at it, diverging ones narrower.

    Returns
    -------
    list of (z, y) tuples of numpy arrays, one per ray.
    """
    tilt_slope = np.tan(np.radians(state.field_angle))
    vergence_per_mm = state.beam_vergence / 1000.0
    start_z = AXIS_Z_RANGE[0]

    rays = []
    for ray_index in range(paths.z.shape[1]):
        height_at_first_surface = paths.y[0, ray_index]
        incoming_slope = tilt_slope - vergence_per_mm * height_at_first_surface
        start_y = height_at_first_surface - incoming_slope * (paths.z[0, ray_index] - start_z)
        ray_z = np.concatenate([[start_z], paths.z[:, ray_index]])
        ray_y = np.concatenate([[start_y], paths.y[:, ray_index]])
        rays.append((ray_z, ray_y))
    return rays


def spot_view_half_width_um(spot: SpotResult, on_axis_spot: SpotResult | None) -> float:
    """Half-width (um) of the square spot plot.

    The view reaches 20% beyond the spots and is never smaller than MIN_SPOT_HALF_WIDTH_UM.
    It also widens to show the Airy circle, unless that is far bigger than the spots: then
    the circle is cropped rather than shrinking the spots to a dot.
    """
    spot_coordinates = [spot.x, spot.y]
    if on_axis_spot is not None:
        spot_coordinates += [on_axis_spot.x, on_axis_spot.y]
    extent_um = largest_magnitude(*spot_coordinates) * 1000.0

    half_width = max(extent_um * 1.2, MIN_SPOT_HALF_WIDTH_UM)
    airy = airy_radius(spot.numerical_aperture)
    if airy is not None:
        half_width = max(half_width, min(airy * 1000.0 * 1.2, half_width * 3.0))
    return float(half_width)


def largest_magnitude(*arrays: np.ndarray, floor: float = 0.0) -> float:
    """The largest absolute value in the arrays, ignoring NaN, and never less than ``floor``.

    Analyses give NaN for rays or angles where no light gets through, and empty arrays when
    none does at all, so plot ranges are worked out with this rather than ``.max()``.
    """
    largest = floor
    for values in arrays:
        finite = np.abs(np.asarray(values, dtype=float))
        finite = finite[np.isfinite(finite)]
        if finite.size:
            largest = max(largest, float(finite.max()))
    return largest


def rms_radius_text(spot: SpotResult, micrometre_symbol: str = "um") -> str:
    """The spot's RMS radius for display, e.g. "1.23 um", or "no light" if no ray arrives."""
    if not spot.has_light:
        return "no light"
    return f"{spot.rms_radius * 1000:.2f} {micrometre_symbol}"


def f_number_text(result: FNumberResult) -> str:
    """The f-number for display, e.g. "f/4.0", noting when a lens rim cuts the beam.

    When the beam is clipped the f-number is based on the part that gets through, so the
    text says so, e.g. "f/1.9 (beam clipped to 10.8 mm)".
    """
    if not np.isfinite(result.f_number):
        return "f/- (no light)"
    text = f"f/{result.f_number:.1f}"
    if result.beam_is_clipped:
        text += f" (beam clipped to {result.transmitted_beam_diameter:.1f} mm)"
    return text


def ray_fan_limit_um(largest_error_mm: float) -> float:
    """Half-height (um) of the zero-centred ray fan plot.

    The range is fixed about zero so that a perfect system shows as a flat line rather than
    a zoomed-in plot of numerical noise.
    """
    return max(largest_error_mm * 1000.0 * 1.15, 1.0)
