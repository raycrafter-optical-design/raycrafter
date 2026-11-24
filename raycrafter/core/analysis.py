"""Optical analyses of the bench: spots, through-focus, ray fans, MTF, field curvature, focus.

Every function takes an ``OpticalState``, builds the Optiland system from it and returns a
small result dataclass holding plain numpy arrays, so that the GUI and the HTML report can
draw the results without touching Optiland. Lengths are in mm unless the name says otherwise.

Several analyses avoid re-tracing: after the last surface each ray is a straight line in air,
so its position at any other plane is position + slope * offset (see ``rms_through_focus``).

Rays blocked by a lens rim (see ``optic_builder``) are left out of every analysis (see
``transmitted``). If no light reaches the image plane at all, the results say so: the spot
has no rays (``SpotResult.has_light`` is False) and the curves are NaN or empty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from optiland.mtf import FFTMTF
from optiland.optic import Optic
from scipy.optimize import minimize_scalar

from raycrafter.core.optic_builder import WAVELENGTH_UM, build_optic
from raycrafter.core.state import AXIS_START_Z, OpticalState

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
    the sine of the half-angle of the converging cone (0 for a parallel beam).

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

    @property
    def has_light(self) -> bool:
        """True if at least one ray reaches the image plane."""
        return self.position_x.size > 0


@dataclass
class RayPaths:
    """Ray polylines for drawing: arrays of shape (n_surfaces, n_rays) of absolute z and y.

    A ray blocked by a lens rim ends where it meets that lens; its later points are NaN.
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
    """What the GUI redraws on every change: both spots, the side-view rays and the summary."""

    spot: SpotResult  # at the current beam angle
    on_axis_spot: SpotResult | None  # at 0 degrees; None when the beam angle is already 0
    paths: RayPaths
    summary: SystemSummary | None  # None when there are no lenses


@dataclass
class ThroughFocusResult:
    """RMS spot radius against image plane position (straight-line propagation of the rays)."""

    offsets: np.ndarray  # mm from the current image plane (positive = further from the lens)
    rms_current: np.ndarray  # mm, at the current beam angle
    rms_on_axis: np.ndarray | None  # mm, at 0 degrees; None when the beam angle is already 0
    best_offset: float  # mm, where the current-angle RMS is smallest
    airy_radius: float | None  # mm; None for a parallel beam


@dataclass
class RayFanResult:
    """Transverse ray aberration at the image plane for the current beam angle (mm).

    Blocked rays have NaN errors; all are NaN if the chief ray itself is blocked.
    """

    pupil: np.ndarray  # normalised pupil coordinate, -1 to +1
    tangential_error: np.ndarray  # y offset from the chief ray against pupil y
    sagittal_error: np.ndarray  # x offset from the chief ray against pupil x


@dataclass
class MtfResult:
    """MTF at the current field angle, with the diffraction limit for reference.

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


def spot_from_rays(rays) -> SpotResult:
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


def airy_radius(numerical_aperture: float) -> float | None:
    """Radius (mm) of the first dark ring of the Airy pattern, 0.61 lambda / NA.

    Returns None if the numerical aperture is effectively zero (a parallel beam).
    """
    if numerical_aperture < 1e-6:
        return None
    return 0.61 * WAVELENGTH_UM * 1e-3 / numerical_aperture


def field_y_sign(field_angle: float) -> float:
    """Optiland field coordinates are normalised to the largest field; the sign picks +/-."""
    return -1.0 if field_angle < 0 else 1.0


def _trace_spot(optic: Optic, normalised_field: float) -> SpotResult:
    """Trace a hexapolar spot at one normalised field."""
    rays = optic.trace(
        Hx=0,
        Hy=normalised_field,
        wavelength=WAVELENGTH_UM,
        num_rays=SPOT_RINGS,
        distribution="hexapolar",
    )
    return spot_from_rays(rays)


def _trace_ray_paths(state: OpticalState, optic: Optic, normalised_field: float) -> RayPaths:
    """Trace a meridional fan for the side view and return absolute z, y at each surface.

    A ray blocked by a lens rim is drawn up to the plane of that surface's rim, where it
    would hit the lens mount, and no further. Row 0 of Optiland's arrays is where the rays
    start, before the first surface (row 1).
    """
    optic.trace(
        Hx=0,
        Hy=normalised_field,
        wavelength=WAVELENGTH_UM,
        num_rays=SIDE_VIEW_RAYS,
        distribution="line_y",
    )
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

    # Shift z so that the first vertex sits at the first lens (or at z = 0 with no lenses).
    first_vertex_z = float(optic.surfaces[1].geometry.cs.z)
    first_z = state.lenses[0].z if state.lenses else AXIS_START_Z
    z_absolute = z_local - first_vertex_z + first_z
    return RayPaths(z=z_absolute[1:], y=y_local[1:])


def _system_summary(state: OpticalState, optic: Optic) -> SystemSummary | None:
    """Effective and back focal length from Optiland's paraxial properties.

    Optiland reports the back focal point relative to the image surface, so the back focal
    length is measured from the last lens vertex using the image plane position.
    """
    if not state.lenses:
        return None
    paraxial = optic.paraxial
    back_focal_point_z = state.image_z + float(paraxial.F2())
    return SystemSummary(
        effective_focal_length=float(paraxial.f2()),
        back_focal_length=float(back_focal_point_z - state.last_back_z()),
    )


# ------------------------------------------------------------------------ analyses


def trace_fast(state: OpticalState) -> FastTrace:
    """Trace what the GUI redraws on every change, building the Optiland system only once.

    The system gets two fields, 0 degrees and the current beam angle (just one if they are
    the same). Optiland's normalised field coordinate Hy = 0 selects the first and Hy = +/-1
    the second. The side-view ray fan is traced last because the per-surface ray positions
    read afterwards belong to the most recent trace.
    """
    angle = state.field_angle
    has_off_axis_field = angle != 0.0
    optic = build_optic(state, field_angles=[0.0, abs(angle)] if has_off_axis_field else [0.0])
    sign = field_y_sign(angle)

    on_axis_spot = _trace_spot(optic, 0.0)
    spot = _trace_spot(optic, sign) if has_off_axis_field else on_axis_spot

    paths = _trace_ray_paths(state, optic, sign if has_off_axis_field else 0.0)
    summary = _system_summary(state, optic)
    return FastTrace(spot, on_axis_spot if has_off_axis_field else None, paths, summary)


def trace_spot(state: OpticalState) -> SpotResult:
    """Trace the spot at the current beam angle only."""
    return _trace_spot(build_optic(state), field_y_sign(state.field_angle))


def compute_through_focus(state: OpticalState, fast_trace: FastTrace) -> ThroughFocusResult:
    """RMS spot radius for image planes either side of the current one.

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
            rms_on_axis=None,
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

    wavelength_mm = WAVELENGTH_UM * 1e-3
    if spot.numerical_aperture > 1e-4:
        half_range = THROUGH_FOCUS_DEPTHS * wavelength_mm / spot.numerical_aperture**2
    else:
        half_range = max_offset
    half_range = float(np.clip(half_range, 0.5, max_offset))

    low = max(min(0.0, best_offset) - half_range, min_offset)
    high = min(max(0.0, best_offset) + half_range, max_offset)
    offsets = np.linspace(low, high, 161)

    on_axis_spot = fast_trace.on_axis_spot
    return ThroughFocusResult(
        offsets=offsets,
        rms_current=rms_through_focus(spot, offsets),
        rms_on_axis=(
            rms_through_focus(on_axis_spot, offsets) if on_axis_spot is not None else None
        ),
        best_offset=best_offset,
        airy_radius=airy_radius(spot.numerical_aperture),
    )


def compute_ray_fan(state: OpticalState) -> RayFanResult:
    """Transverse ray aberration at the image plane for the current beam angle.

    Two fans of rays across the pupil are traced: one along y (tangential) and one along x
    (sagittal). For each ray the error is its image-plane position minus that of the chief
    ray, which is the middle ray of the fan. Optiland orders the fan from -1 to +1 in the
    normalised pupil coordinate.
    """
    optic = build_optic(state)
    normalised_field = field_y_sign(state.field_angle)
    fans = {}
    for distribution in ("line_y", "line_x"):
        fans[distribution] = optic.trace(
            Hx=0,
            Hy=normalised_field,
            wavelength=WAVELENGTH_UM,
            num_rays=RAY_FAN_RAYS,
            distribution=distribution,
        )

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

    This needs an extra trace, so it is not part of ``trace_fast``: the GUI computes it only
    once nothing is being dragged.
    """
    if not state.lenses:
        return None
    optic = build_optic(state, field_angles=[0.0])
    transmitted_diameter_mm = _measure_transmitted_diameter(state, optic)
    effective_focal_length = float(optic.paraxial.f2())
    if transmitted_diameter_mm > 0.0:
        f_number = abs(effective_focal_length) / transmitted_diameter_mm
    else:
        f_number = float("nan")
    return FNumberResult(
        f_number=f_number,
        transmitted_beam_diameter=transmitted_diameter_mm,
        beam_is_clipped=transmitted_diameter_mm < state.beam_diameter,
    )


def transmitted_beam_diameter(state: OpticalState) -> float:
    """Diameter (mm) of the part of the on-axis beam that gets through every lens.

    When the beam is wider than a lens, the lens rim limits it. See
    ``_measure_transmitted_diameter``.
    """
    return _measure_transmitted_diameter(state, build_optic(state, field_angles=[0.0]))


def _measure_transmitted_diameter(state: OpticalState, optic: Optic) -> float:
    """Transmitted on-axis beam diameter (mm), measured on an already built system.

    A dense fan of rays across the beam is traced on axis (normalised field 0, which is
    0 degrees in every system built here), and the widest ray that gets through sets the
    diameter (0 if none does). The fan is symmetric, so the larger edge is used.
    """
    fan = optic.trace(
        Hx=0, Hy=0, wavelength=WAVELENGTH_UM, num_rays=PUPIL_PROBE_RAYS, distribution="line_y"
    )
    pupil = np.linspace(-1.0, 1.0, PUPIL_PROBE_RAYS)
    getting_through = transmitted(fan)
    if not getting_through.any():
        return 0.0
    return float(np.abs(pupil[getting_through]).max() * state.beam_diameter)


def compute_mtf(state: OpticalState) -> MtfResult:
    """Diffraction MTF (FFT method) at the current beam angle, up to the cutoff frequency.

    Returns tangential and sagittal curves plus the ideal diffraction-limited curve for the
    same cutoff, so the user can see how far the system is from perfect.

    Optiland's MTF needs the ray through the edge of the pupil, so when a lens rim cuts the
    beam the MTF is computed for the part of the beam that gets through on axis (which is
    what the lens actually transmits). If the beam is cut further at the current angle
    (vignetting), Optiland cannot compute the MTF and the result is empty.
    """
    no_result = MtfResult(np.empty(0), np.empty(0), np.empty(0), np.empty(0))
    beam_diameter_mm = min(state.beam_diameter, transmitted_beam_diameter(state))
    if beam_diameter_mm <= 0.0:
        return no_result
    try:
        mtf = FFTMTF(build_optic(state, beam_diameter_mm=beam_diameter_mm), num_rays=MTF_RAYS)
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
    """Best-focus shift of the tangential and sagittal ray fans for field angles 0 to 10 degrees.

    For each field angle a fan of rays across the beam is traced to the image plane (a line
    in y for tangential, in x for sagittal), and its best focus found in closed form (see
    ``_best_focus_offset``). Zero means the image plane is in focus. This is used instead of
    Optiland's ``FieldCurvature`` class (see log.md).
    """
    angles = np.linspace(0.0, FIELD_CURVATURE_MAX_ANGLE, FIELD_CURVATURE_POINTS)
    optic = build_optic(state, field_angles=list(angles))

    tangential, sagittal = [], []
    for angle in angles:
        normalised_field = angle / FIELD_CURVATURE_MAX_ANGLE
        trace_settings = dict(
            Hx=0,
            Hy=normalised_field,
            wavelength=WAVELENGTH_UM,
            num_rays=FIELD_CURVATURE_FAN_RAYS,
        )
        fan_y = optic.trace(distribution="line_y", **trace_settings)
        fan_x = optic.trace(distribution="line_x", **trace_settings)
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
    """The image plane z that minimises the RMS spot radius at the current beam angle.

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
