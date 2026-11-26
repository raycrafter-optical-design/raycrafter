"""Geometry for drawing the bench, shared by the GUI and the HTML report.

These functions turn model data into plain arrays to draw (lens outlines, rays, plot
ranges). They know nothing about Qt or matplotlib, so both front ends draw the same
picture.
"""

from __future__ import annotations

import numpy as np

from raycrafter.core.analysis import FNumberResult, RayPaths, SpotResult, airy_radius
from raycrafter.core.prescription import surface_sag
from raycrafter.core.state import AXIS_START_Z, AnyField, BeamField, OpticalState, PlacedLens

DEFAULT_AXIS_Z_RANGE = (AXIS_START_Z, 450.0)  # mm of optical axis shown for a design that fits
DEFAULT_AXIS_Y_RANGE = (-32.0, 32.0)  # mm either side of the axis shown for a design that fits
IMAGE_EDGE_GAP_MM = 5.0  # the image plane is kept at least this far inside the right edge
AXIS_Z_ROOM_FRACTION = 0.1  # a design too long for the default is shown with 10% spare axis
AXIS_Y_ROOM_FRACTION = 0.15  # space above the tallest lens, as a fraction of its semi-aperture
MIN_SPOT_HALF_WIDTH_UM = 5.0  # the spot plot never zooms in beyond +/-5 um


def axis_ranges(state: OpticalState) -> tuple[tuple[float, float], tuple[float, float]]:
    """The z and y ranges (mm) of the side view that show the whole design.

    A design that fits in the default ranges (0-450 mm of axis, +/-32 mm) is shown in
    them, so the view of ordinary designs never changes. A longer design is shown up to 10%
    beyond its image plane (or last lens), which leaves room to drag the image plane
    further. A taller design is shown with some space above and below its tallest lens.
    Rays are not considered: they can diverge without limit after the last lens.

    Parameters
    ----------
    state : OpticalState
        The bench to fit.

    Returns
    -------
    z_range, y_range : tuple of (float, float)
        The (min, max) of each axis, in mm.
    """
    content_end_z = max(state.image_z, state.last_back_z())
    z_max = DEFAULT_AXIS_Z_RANGE[1]
    if content_end_z > z_max - IMAGE_EDGE_GAP_MM:
        z_max = content_end_z + AXIS_Z_ROOM_FRACTION * (content_end_z - AXIS_START_Z)

    tallest_semi_aperture = max((lens.semi_aperture for lens in state.lenses), default=0.0)
    y_half_height = max(
        DEFAULT_AXIS_Y_RANGE[1], tallest_semi_aperture * (1.0 + AXIS_Y_ROOM_FRACTION)
    )
    return (DEFAULT_AXIS_Z_RANGE[0], z_max), (-y_half_height, y_half_height)


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


def ray_polylines(paths: RayPaths) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each traced side-view ray as a (z, y) pair of arrays, from the start of the axis.

    Points after a ray is blocked are NaN; drawing code should leave gaps there.
    """
    return [(paths.z[:, ray_index], paths.y[:, ray_index]) for ray_index in range(paths.z.shape[1])]


def spot_view_half_width_um(spots: list[SpotResult], selected_spot: SpotResult) -> float:
    """Half-width (um) of the square spot plot, in which every field's spot is centred.

    The view reaches 20% beyond the spots and is never smaller than MIN_SPOT_HALF_WIDTH_UM.
    It also widens to show the selected field's Airy circle, unless that is far bigger than
    the spots: then the circle is cropped rather than shrinking the spots to a dot.
    """
    extent_um = largest_magnitude(
        *[coordinate for spot in spots for coordinate in (spot.x, spot.y)]
    )
    extent_um *= 1000.0

    half_width = max(extent_um * 1.2, MIN_SPOT_HALF_WIDTH_UM)
    airy = airy_radius(selected_spot)
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


def field_name(field_index: int) -> str:
    """What a field is called, e.g. ``"Field 0"``.

    Fields are numbered from 0, as the Python API and the agent tools number them.
    """
    return f"Field {field_index}"


def source_caption(state: OpticalState) -> str:
    """The side view's caption for the source plane: "object" for a finite object, else "source"."""
    return "object" if state.has_finite_object else "source"


def field_position_text(light: AnyField) -> str:
    """Where a field is, for plot titles: "+2.5 deg" for a beam, "y = +0.50 mm" for a point.

    An object point off the y axis gives both heights, e.g. "x = +0.30, y = -0.20 mm".
    """
    if isinstance(light, BeamField):
        return f"{light.angle_deg:+.1f} deg"
    if light.height_x_mm == 0.0:
        return f"y = {light.height_y_mm:+.2f} mm"
    return f"x = {light.height_x_mm:+.2f}, y = {light.height_y_mm:+.2f} mm"


def magnification_text(magnification: float) -> str:
    """The paraxial magnification for display, e.g. "M=-20.00x", or "M=-" if it has none."""
    if not np.isfinite(magnification):
        return "M=-"
    return f"M={magnification:.2f}x"


def image_point_text(image_x_mm: float, image_y_mm: float) -> str:
    """Where a chief ray meets the image plane, e.g. "image y=-10.000 mm".

    Both coordinates are given if x is not zero, and "image: chief ray blocked" if the ray
    does not get there (NaN).
    """
    if not (np.isfinite(image_x_mm) and np.isfinite(image_y_mm)):
        return "image: chief ray blocked"
    if abs(image_x_mm) < 5e-4:  # shows as 0.000
        return f"image y={image_y_mm:.3f} mm"
    return f"image x={image_x_mm:.3f}, y={image_y_mm:.3f} mm"


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
