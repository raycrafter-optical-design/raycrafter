"""Optical analyses of the bench: spots, through-focus, ray fans, MTF, field curvature, focus.

Every function takes an ``OpticalState``, builds the Optiland system from it and returns a
small result dataclass holding plain numpy arrays, so that the GUI and the HTML report can
draw the results without touching Optiland. Lengths are in mm unless the name says otherwise.

The spots and side-view rays are computed for every field (beam). The other analyses, and
autofocus, use the selected field (``OpticalState.selected_field``). Each field's rays are
launched by ``raycrafter.core.ray_launch`` and traced at the field's own wavelength.

Several analyses avoid re-tracing: after the last surface each ray is a straight line in air,
so its position at any other plane is position + slope * offset (see ``rms_through_focus``).

Rays blocked by a lens rim (see ``optic_builder``) are left out of every analysis (see
``transmitted``). If no light reaches the image plane at all, the results say so: the spot
has no rays (``SpotResult.has_light`` is False) and the curves are NaN or empty.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from optiland.mtf import FFTMTF
from optiland.optic import Optic
from optiland.rays import RealRays
from scipy.optimize import minimize_scalar

from raycrafter.core.optic_builder import build_optic, build_optic_for_beam
from raycrafter.core.ray_launch import launch_rays, pupil_points, trace_rays
from raycrafter.core.state import BeamField, OpticalState

SPOT_RINGS = 6  # hexapolar rings for the spot diagram (127 rays)
SIDE_VIEW_RAYS = 5  # rays across the beam diameter for the side view
MTF_RAYS = 128
FIELD_CURVATURE_MAX_ANGLE = 10.0  # degrees
FIELD_CURVATURE_POINTS = 11
FIELD_CURVATURE_FAN_RAYS = 21  # rays per fan for the field curvature estimate
RAY_FAN_RAYS = 41  # rays across the pupil in each ray fan (odd, so one is the chief ray)
PUPIL_PROBE_RAYS = 401  # rays across the beam used to find how much of it gets through
THROUGH_FOCUS_DEPTHS = 4.0  # the through-focus window spans this many depths of focus
THROUGH_FOCUS_MAX_HALF_RANGE_MM = 100.0  # ... but never more than this from the image plane
AUTOFOCUS_SEARCH_LIMIT_MM = 600.0  # autofocus looks at most this far behind the image plane


# ----------------------------------------------------------------------------- results


@dataclass
class SpotResult:
    """Spot diagram data at the image plane.

    ``x`` and ``y`` are relative to the spot centroid, for plotting. The ``position_*`` and
    ``slope_*`` arrays are the rays' absolute transverse positions and slopes (dx/dz, dy/dz)
    at the image plane. With them the rays can be propagated to other planes, which is how
    autofocus and the through-focus plot work without re-tracing. ``numerical_aperture`` is
    the sine of the half-angle of the converging cone (0 for a parallel beam), and
    ``wavelength_um`` the wavelength the rays were traced at.

    Only rays that reach the image plane are included. If none do, the arrays are empty and
    ``rms_radius`` is NaN.
    """

    x: np.ndarray
    y: np.ndarray
    rms_radius: float
    centroid_y: float
    position_x: np.ndarray
    position_y: np.ndarray
    slope_x: np.ndarray
    slope_y: np.ndarray
    numerical_aperture: float
    wavelength_um: float

    @property
    def has_light(self) -> bool:
        """True if at least one ray reaches the image plane."""
        return self.position_x.size > 0


@dataclass
class RayPaths:
    """Ray polylines for drawing: arrays of shape (n_points, n_rays) of absolute z and y.

    Each ray starts at the start of the axis and has a point at every surface. A ray blocked
    by a lens rim ends where it meets that lens; its later points are NaN.
    """

    z: np.ndarray
    y: np.ndarray


@dataclass
class SystemSummary:
    """First-order (paraxial) properties of the whole lens train."""

    effective_focal_length: float  # mm
    back_focal_length: float  # mm, last lens vertex to the back focal point


@dataclass
class FNumberResult:
    """The f-number, from the part of the beam that gets through every lens.

    When a lens rim cuts the beam, the rim sets the cone of light, so the f-number uses the
    transmitted beam diameter rather than the nominal one (see log.md).
    """

    f_number: float  # |effective focal length| / transmitted beam diameter; NaN if no light
    transmitted_beam_diameter: float  # mm of the on-axis beam that gets through every lens
    beam_is_clipped: bool  # True if a lens rim cuts the beam (transmitted < nominal)


@dataclass
class FastTrace:
    """What the GUI redraws on every change: every field's spot and rays, and the summary."""

    spots: list[SpotResult]  # one per field, in the order of OpticalState.fields
    paths: list[RayPaths]  # one per field
    summary: SystemSummary | None  # None when there are no lenses
    selected_field_index: int

    @property
    def spot(self) -> SpotResult:
        """The selected field's spot."""
        return self.spots[self.selected_field_index]


@dataclass
class ThroughFocusResult:
    """RMS spot radius against image plane position (straight-line propagation of the rays)."""

    offsets: np.ndarray  # mm from the current image plane (positive = further from the lens)
    rms_current: np.ndarray  # mm, for the selected field
    best_offset: float  # mm, where the RMS is smallest
    airy_radius: float | None  # mm; None for a parallel beam


@dataclass
class RayFanResult:
    """Transverse ray aberration at the image plane for the selected field (mm).

    Blocked rays have NaN errors; all are NaN if the chief ray itself is blocked.
    """

    pupil: np.ndarray  # normalised pupil coordinate, -1 to +1
    tangential_error: np.ndarray  # y offset from the chief ray against pupil y
    sagittal_error: np.ndarray  # x offset from the chief ray against pupil x


@dataclass
class MtfResult:
    """MTF for the selected field, with the diffraction limit for reference.

    The arrays are empty when the MTF cannot be computed because the beam is blocked
    (see ``compute_mtf``).
    """

    frequency: np.ndarray  # cycles/mm
    tangential: np.ndarray
    sagittal: np.ndarray
    diffraction_limit: np.ndarray


@dataclass
class FieldCurvatureResult:
    """Best-focus position of tangential and sagittal fans against field angle.

    Shifts are in mm relative to the current image plane (positive = behind it). They are
    NaN at angles where too few rays get through to find a focus.
    """

    angles: np.ndarray  # degrees
    tangential_shift: np.ndarray
    sagittal_shift: np.ndarray


# ------------------------------------------------------------------------- helpers


def transmitted(rays) -> np.ndarray:
    """Boolean mask of the traced rays that reach the image plane.

    Optiland marks a ray blocked by an aperture by setting its intensity to zero; it keeps
    tracing it, and its coordinates may become NaN where it misses a surface. A ray counts
    as transmitted if its intensity is positive and its position and direction are finite.
    """
    coordinates = [np.asarray(getattr(rays, name), dtype=float) for name in "xyLMN"]
    finite = np.logical_and.reduce([np.isfinite(values) for values in coordinates])
    return (np.asarray(rays.i, dtype=float) > 0) & finite


def spot_from_rays(rays, wavelength_um: float) -> SpotResult:
    """Build a SpotResult from the transmitted Optiland rays traced to the image plane."""
    keep = transmitted(rays)
    if not keep.any():
        no_rays = np.empty(0)
        return SpotResult(
            x=no_rays,
            y=no_rays,
            rms_radius=float("nan"),
            centroid_y=0.0,
            position_x=no_rays,
            position_y=no_rays,
            slope_x=no_rays,
            slope_y=no_rays,
            numerical_aperture=0.0,
            wavelength_um=wavelength_um,
        )

    x, y = np.asarray(rays.x, dtype=float)[keep], np.asarray(rays.y, dtype=float)[keep]
    direction_x = np.asarray(rays.L, dtype=float)[keep]
    direction_y = np.asarray(rays.M, dtype=float)[keep]
    direction_z = np.asarray(rays.N, dtype=float)[keep]

    centroid_x, centroid_y = x.mean(), y.mean()
    rms_radius = float(np.sqrt(np.mean((x - centroid_x) ** 2 + (y - centroid_y) ** 2)))

    # Numerical aperture: the largest angle of any ray from the mean direction of the beam
    # (using the mean direction handles tilted beams). A parallel beam gives zero.
    cone_sine = np.hypot(direction_x - direction_x.mean(), direction_y - direction_y.mean())

    return SpotResult(
        x=x - centroid_x,
        y=y - centroid_y,
        rms_radius=rms_radius,
        centroid_y=float(centroid_y),
        position_x=x,
        position_y=y,
        slope_x=direction_x / direction_z,
        slope_y=direction_y / direction_z,
        numerical_aperture=float(cone_sine.max()),
        wavelength_um=wavelength_um,
    )


def rms_through_focus(spot: SpotResult, offsets) -> np.ndarray:
    """RMS spot radius (mm) if the image plane were moved by each of ``offsets`` (mm).

    Each ray is a straight line after the last surface, so its position at a trial plane is
    position + slope * offset. This is vectorised over offsets (rows) and rays (columns).
    All NaN if no light reaches the image plane.
    """
    if not spot.has_light:
        return np.full(len(offsets), np.nan)
    offsets = np.asarray(offsets, dtype=float)[:, None]
    xs = spot.position_x[None, :] + spot.slope_x[None, :] * offsets
    ys = spot.position_y[None, :] + spot.slope_y[None, :] * offsets
    spread = (xs - xs.mean(axis=1, keepdims=True)) ** 2 + (ys - ys.mean(axis=1, keepdims=True)) ** 2
    return np.sqrt(spread.mean(axis=1))


def airy_radius(spot: SpotResult) -> float | None:
    """Radius (mm) of the first dark ring of the spot's Airy pattern, 0.61 lambda / NA.

    Returns None if the numerical aperture is effectively zero (a parallel beam).
    """
    if spot.numerical_aperture < 1e-6:
        return None
    return 0.61 * spot.wavelength_um * 1e-3 / spot.numerical_aperture


def _trace_field(
    state: OpticalState, optic: Optic, beam: BeamField, distribution: str, ray_count: int
) -> RealRays:
    """Launch one field's rays in the given pattern (see ``ray_launch.pupil_points``), trace them."""
    pupil_x, pupil_y = pupil_points(distribution, ray_count)
    return trace_rays(optic, launch_rays(state, beam, optic, pupil_x, pupil_y))


def _trace_spot(state: OpticalState, optic: Optic, beam: BeamField) -> SpotResult:
    """Trace a hexapolar spot for one field."""
    rays = _trace_field(state, optic, beam, "hexapolar", SPOT_RINGS)
    return spot_from_rays(rays, beam.wavelength_um)


def _trace_ray_paths(state: OpticalState, optic: Optic, beam: BeamField) -> RayPaths:
    """Trace a meridional fan of one field for the side view; return absolute z, y of each ray.

    Row 0 of Optiland's per-surface arrays is the (object) surface at infinity, which leaves
    the rays where they were launched, at the start of the axis; row 1 is the first lens
    surface. A ray blocked by a lens rim is drawn up to the plane of that surface's rim,
    where it would hit the lens mount, and no further.
    """
    _trace_field(state, optic, beam, "line_y", SIDE_VIEW_RAYS)

    # Optiland keeps the per-surface ray positions of the most recent trace on its surfaces.
    z_local = np.array(optic.surfaces.z, dtype=float)
    y_local = np.array(optic.surfaces.y, dtype=float)
    slope_local = np.asarray(optic.surfaces.M, dtype=float) / np.asarray(
        optic.surfaces.N, dtype=float
    )
    intensity = np.asarray(optic.surfaces.intensity, dtype=float)

    # Optiland stops a blocked ray where it meets the surface's mathematical continuation
    # beyond the rim (and gives NaN if the sphere does not reach that far). Instead, end it
    # on the plane of the rim: continue its straight line from the previous surface to there.
    for ray_index in range(intensity.shape[1]):
        blocked_at = np.flatnonzero(intensity[:, ray_index] <= 0)
        if blocked_at.size == 0:
            continue
        surface_index = int(blocked_at[0])
        surface = optic.surfaces[surface_index]
        if surface.aperture is not None:  # blocked by a rim (not, say, total internal reflection)
            rim_z = float(surface.geometry.cs.z + surface.geometry.sag(0.0, surface.aperture.r_max))
            previous = surface_index - 1
            z_local[surface_index, ray_index] = rim_z
            y_local[surface_index, ray_index] = y_local[previous, ray_index] + slope_local[
                previous, ray_index
            ] * (rim_z - z_local[previous, ray_index])
        z_local[surface_index + 1 :, ray_index] = np.nan
        y_local[surface_index + 1 :, ray_index] = np.nan

    # Shift z so that the first vertex sits at the first lens (or the start of the axis).
    first_vertex_z = float(optic.surfaces[1].geometry.cs.z)
    z_absolute = z_local - first_vertex_z + state.beam_reference_z()
    return RayPaths(z=z_absolute, y=y_local)


def _system_summary(state: OpticalState, optic: Optic) -> SystemSummary | None:
    """Effective and back focal length from Optiland's paraxial properties.

    Optiland reports the back focal point relative to the image surface, so the back focal
    length is measured from the last lens vertex using the image plane position. Both are
    at the selected field's wavelength (the primary wavelength of ``build_optic``).
    """
    if not state.lenses:
        return None
    paraxial = optic.paraxial
    # An afocal system (e.g. two lenses a typed gap apart that makes a beam expander) has no
    # focal point: both lengths are then infinite, which is the answer rather than a fault.
    with np.errstate(divide="ignore"):
        back_focal_point_z = state.image_z + float(paraxial.F2())
        effective_focal_length = float(paraxial.f2())
    return SystemSummary(
        effective_focal_length=effective_focal_length,
        back_focal_length=float(back_focal_point_z - state.last_back_z()),
    )


# ------------------------------------------------------------------------ analyses


def trace_fast(state: OpticalState) -> FastTrace:
    """Trace what the GUI redraws on every change, building the Optiland system only once.

    Each field gets a hexapolar spot and a fan of rays for the side view. Each fan's
    per-surface positions are read straight after it is traced, before the next trace
    replaces them.
    """
    optic = build_optic(state)
    spots, paths = [], []
    for beam in state.fields:
        spots.append(_trace_spot(state, optic, beam))
        paths.append(_trace_ray_paths(state, optic, beam))
    return FastTrace(spots, paths, _system_summary(state, optic), state.selected_field_index)


def trace_spot(state: OpticalState) -> SpotResult:
    """Trace the spot of the selected field only."""
    return _trace_spot(state, build_optic(state), state.selected_field)


def compute_through_focus(state: OpticalState, fast_trace: FastTrace) -> ThroughFocusResult:
    """RMS spot radius of the selected field for image planes either side of the current one.

    The window is a few depths of focus (lambda / NA^2) wide, spans both the current image
    plane and the best focus so both are always visible, and cannot reach back into the
    last lens.
    """
    spot = fast_trace.spot
    if not spot.has_light:
        offsets = np.linspace(-1.0, 1.0, 161)
        return ThroughFocusResult(
            offsets=offsets,
            rms_current=np.full(offsets.size, np.nan),
            best_offset=0.0,
            airy_radius=None,
        )
    max_offset = THROUGH_FOCUS_MAX_HALF_RANGE_MM
    min_offset = max(state.last_back_z() - state.image_z + 0.5, -max_offset)

    # A wide, coarse scan finds the best focus wherever it is; a fine scan then refines it.
    coarse_offsets = np.linspace(min_offset, max_offset, 400)
    coarse_best = coarse_offsets[np.argmin(rms_through_focus(spot, coarse_offsets))]
    coarse_step = coarse_offsets[1] - coarse_offsets[0]
    fine_offsets = np.linspace(coarse_best - coarse_step, coarse_best + coarse_step, 201)
    best_offset = float(fine_offsets[np.argmin(rms_through_focus(spot, fine_offsets))])

    wavelength_mm = spot.wavelength_um * 1e-3
    if spot.numerical_aperture > 1e-4:
        half_range = THROUGH_FOCUS_DEPTHS * wavelength_mm / spot.numerical_aperture**2
    else:
        half_range = max_offset
    half_range = float(np.clip(half_range, 0.5, max_offset))

    low = max(min(0.0, best_offset) - half_range, min_offset)
    high = min(max(0.0, best_offset) + half_range, max_offset)
    offsets = np.linspace(low, high, 161)
    return ThroughFocusResult(
        offsets=offsets,
        rms_current=rms_through_focus(spot, offsets),
        best_offset=best_offset,
        airy_radius=airy_radius(spot),
    )


def compute_ray_fan(state: OpticalState) -> RayFanResult:
    """Transverse ray aberration of the selected field at the image plane.

    Two fans of rays across the beam are traced: one along y (tangential) and one along x
    (sagittal). For each ray the error is its image-plane position minus that of the chief
    ray, which is the middle ray of the fan. The fans run from -1 to +1 in the normalised
    pupil coordinate.
    """
    optic = build_optic(state)
    beam = state.selected_field
    fans = {
        distribution: _trace_field(state, optic, beam, distribution, RAY_FAN_RAYS)
        for distribution in ("line_y", "line_x")
    }

    def errors_from_chief_ray(fan, coordinate: str) -> np.ndarray:
        """Image-plane position minus the chief ray's, NaN for blocked rays."""
        position = np.where(
            transmitted(fan), np.asarray(getattr(fan, coordinate), dtype=float), np.nan
        )
        return position - position[RAY_FAN_RAYS // 2]

    return RayFanResult(
        pupil=np.linspace(-1.0, 1.0, RAY_FAN_RAYS),
        tangential_error=errors_from_chief_ray(fans["line_y"], "y"),
        sagittal_error=errors_from_chief_ray(fans["line_x"], "x"),
    )


def compute_f_number(state: OpticalState) -> FNumberResult | None:
    """The f-number from the transmitted beam diameter; None when there are no lenses.

    The beam is the selected field, put on axis (see ``transmitted_beam_diameter``). This
    needs an extra trace, so it is not part of ``trace_fast``: the GUI computes it only
    once nothing is being dragged.
    """
    if not state.lenses:
        return None
    optic = build_optic(state)
    transmitted_diameter_mm = _measure_transmitted_diameter(state, optic)
    effective_focal_length = float(optic.paraxial.f2())
    if transmitted_diameter_mm > 0.0:
        f_number = abs(effective_focal_length) / transmitted_diameter_mm
    else:
        f_number = float("nan")
    return FNumberResult(
        f_number=f_number,
        transmitted_beam_diameter=transmitted_diameter_mm,
        beam_is_clipped=transmitted_diameter_mm < state.selected_field.diameter_mm,
    )


def transmitted_beam_diameter(state: OpticalState) -> float:
    """Diameter (mm) of the part of the selected field's beam that gets through every lens.

    The beam is taken on axis (no angle or offset), so this measures what the lens rims
    allow, not vignetting. See ``_measure_transmitted_diameter``.
    """
    return _measure_transmitted_diameter(state, build_optic(state))


def _measure_transmitted_diameter(state: OpticalState, optic: Optic) -> float:
    """Transmitted on-axis beam diameter (mm), measured on an already built system.

    A dense fan of rays across the selected field's beam, put on axis, is traced, and the
    widest ray that gets through sets the diameter (0 if none does). The fan is symmetric,
    so the larger edge is used.
    """
    on_axis_beam = replace(state.selected_field, angle_deg=0.0, offset_mm=0.0)
    fan = _trace_field(state, optic, on_axis_beam, "line_y", PUPIL_PROBE_RAYS)
    pupil = np.linspace(-1.0, 1.0, PUPIL_PROBE_RAYS)
    getting_through = transmitted(fan)
    if not getting_through.any():
        return 0.0
    return float(np.abs(pupil[getting_through]).max() * on_axis_beam.diameter_mm)


def compute_mtf(state: OpticalState) -> MtfResult:
    """Diffraction MTF (FFT method) of the selected field, up to the cutoff frequency.

    Returns tangential and sagittal curves plus the ideal diffraction-limited curve for the
    same cutoff, so the user can see how far the system is from perfect.

    This uses Optiland's own object and pupil model (``build_optic_for_beam``), which cannot
    describe a field's offset: the MTF is for the beam centred on the first lens.

    Optiland's MTF needs the ray through the edge of the pupil, so when a lens rim cuts the
    beam the MTF is computed for the part of the beam that gets through on axis (which is
    what the lens actually transmits). If the beam is cut further at the field's angle
    (vignetting), Optiland cannot compute the MTF and the result is empty.
    """
    no_result = MtfResult(np.empty(0), np.empty(0), np.empty(0), np.empty(0))
    beam = state.selected_field
    beam_diameter_mm = min(beam.diameter_mm, transmitted_beam_diameter(state))
    if beam_diameter_mm <= 0.0:
        return no_result
    try:
        optic = build_optic_for_beam(state, beam, beam_diameter_mm=beam_diameter_mm)
        mtf = FFTMTF(optic, num_rays=MTF_RAYS)
    except ValueError:
        # Optiland raises "Working F/# could not be calculated due to raytrace errors" when
        # the edge-of-pupil ray is blocked at this field angle.
        return no_result
    cutoff = float(mtf.max_freq)
    frequency = np.asarray(mtf.freq_tang[0], dtype=float)
    keep = frequency <= cutoff

    # MTF of an aberration-free circular pupil.
    ratio = np.clip(frequency[keep] / cutoff, 0.0, 1.0)
    ideal = (2 / np.pi) * (np.arccos(ratio) - ratio * np.sqrt(1 - ratio**2))
    return MtfResult(
        frequency=frequency[keep],
        tangential=np.asarray(mtf.mtf[0][0], dtype=float)[keep],
        sagittal=np.asarray(mtf.mtf[0][1], dtype=float)[keep],
        diffraction_limit=ideal,
    )


def _best_focus_offset(position: np.ndarray, slope: np.ndarray) -> float:
    """Offset along z that minimises the variance of position + slope * offset.

    Setting the derivative of var(position + slope * t) to zero gives the closed form
    t = -cov(position, slope) / var(slope). NaN if fewer than three rays are given (too few
    to define a focus; blocked rays are left out by the caller).
    """
    if position.size < 3:
        return float("nan")
    slope_variance = np.var(slope)
    if slope_variance < 1e-18:
        return 0.0
    return float(-np.cov(position, slope, bias=True)[0, 1] / slope_variance)


def compute_field_curvature(state: OpticalState) -> FieldCurvatureResult:
    """Best-focus shift of the tangential and sagittal fans for field angles 0 to 10 degrees.

    The selected field's beam is swung through the angles. At each angle a fan of rays
    across the beam is traced to the image plane (a line in y for tangential, in x for
    sagittal), and its best focus found in closed form (see ``_best_focus_offset``). Zero
    means the image plane is in focus. This is used instead of Optiland's
    ``FieldCurvature`` class (see log.md).
    """
    angles = np.linspace(0.0, FIELD_CURVATURE_MAX_ANGLE, FIELD_CURVATURE_POINTS)
    optic = build_optic(state)

    tangential, sagittal = [], []
    for angle in angles:
        beam = replace(state.selected_field, angle_deg=float(angle))
        fan_y = _trace_field(state, optic, beam, "line_y", FIELD_CURVATURE_FAN_RAYS)
        fan_x = _trace_field(state, optic, beam, "line_x", FIELD_CURVATURE_FAN_RAYS)
        keep_y, keep_x = transmitted(fan_y), transmitted(fan_x)
        tangential.append(
            _best_focus_offset(
                np.asarray(fan_y.y)[keep_y], (np.asarray(fan_y.M) / np.asarray(fan_y.N))[keep_y]
            )
        )
        sagittal.append(
            _best_focus_offset(
                np.asarray(fan_x.x)[keep_x], (np.asarray(fan_x.L) / np.asarray(fan_x.N))[keep_x]
            )
        )

    return FieldCurvatureResult(angles, np.array(tangential), np.array(sagittal))


def find_best_focus_z(state: OpticalState) -> float:
    """The image plane z that minimises the RMS spot radius of the selected field.

    The rays are traced once to the current image plane, then propagated in straight lines
    (they are in air) to trial planes, so no re-tracing is needed. A coarse scan comes first
    so the minimiser cannot land in a local minimum. The result is never in front of the
    back of the last lens. If no light reaches the image plane, the image plane stays put.
    """
    spot = trace_spot(state)
    if not spot.has_light:
        return state.image_z
    min_z = state.last_back_z()
    offsets = np.linspace(min_z - state.image_z, AUTOFOCUS_SEARCH_LIMIT_MM, 400)
    coarse_best_offset = offsets[np.argmin(rms_through_focus(spot, offsets))]
    step = offsets[1] - offsets[0]
    refined = minimize_scalar(
        lambda offset: float(rms_through_focus(spot, [offset])[0]),
        bounds=(coarse_best_offset - step, coarse_best_offset + step),
        method="bounded",
    )
    return max(state.image_z + refined.x, min_z)
