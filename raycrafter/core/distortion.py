"""Grid distortion of a finite object: how far the image of a square grid is from a scaled copy.

A square grid of object points on the source plane is imaged, and each image point is compared
with the *ideal* one, magnification times the object point. The field of view is set by the
design's fields: the corners of the grid are as far from the axis as the farthest object point
(as in Zemax and Optiland). The distortion of a point, in percent, is

    100 (r_image - r_ideal) / r_ideal

with r the distance from the axis, so it is positive for pincushion and negative for barrel
distortion.

The magnification of the ideal grid is that of a point very close to the axis, measured on the
image plane itself (as Zemax and Optiland do). At the paraxial focus this is the paraxial
magnification. Elsewhere it is not, and using the paraxial one would report a defocused image
as distorted all over, even next to the axis.

Where an object point's image is can be taken from its chief ray (the real ray aimed at the
centre of the first lens) or from the centroid of its spot. Only a finite object is covered:
beams (an object at infinity) have no object grid.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from optiland.optic import Optic

from raycrafter.core.analysis import SPOT_RINGS, transmitted
from raycrafter.core.fields import field_height_mm, source_to_lens_mm
from raycrafter.core.optic_builder import build_optic
from raycrafter.core.ray_launch import join_rays, launch_from_point, pupil_points, trace_rays
from raycrafter.core.state import ObjectPoint, OpticalState

DISTORTION_REFERENCES = ("chief_ray", "centroid")  # how an image point is located
GRID_POINTS = 11  # object points along each side of the grid
HEIGHT_CURVE_POINTS = 21  # object heights of the distortion curve, from the axis outwards
# The reference magnification is measured at this fraction of the field of view. Distortion
# grows as the square of the height, so there it is about 1e-8 of that at the edge.
NEAR_AXIS_FRACTION = 1e-4


@dataclass
class DistortionResult:
    """The image of a grid of object points, and the distortion against object height.

    The grid arrays have shape (n, n), with y changing along the first axis and x along the
    second. Image points are NaN where no light gets through. All the arrays are empty if
    there is nothing to image; ``unavailable_reason`` then says why.
    """

    reference: str  # "chief_ray" or "centroid": how the image points were located
    magnification: float  # of the ideal grid: image over object position next to the axis
    object_x: np.ndarray  # mm on the object (source) plane
    object_y: np.ndarray
    image_x: np.ndarray  # mm on the image plane
    image_y: np.ndarray
    heights: np.ndarray  # object heights (mm) of the curve, from 0 to the grid's corners
    height_distortion_percent: np.ndarray  # at each of ``heights``
    unavailable_reason: str = ""

    @property
    def has_grid(self) -> bool:
        """True if there is a grid to show (even if its light is blocked)."""
        return self.object_x.size > 0

    @property
    def ideal_x(self) -> np.ndarray:
        """x (mm) of the undistorted image of each grid point."""
        return self.magnification * self.object_x

    @property
    def ideal_y(self) -> np.ndarray:
        """y (mm) of the undistorted image of each grid point."""
        return self.magnification * self.object_y

    @property
    def grid_distortion_percent(self) -> np.ndarray:
        """Distortion (%) of each grid point; 0 on the axis, NaN where blocked."""
        return _distortion_percent(
            np.hypot(self.image_x, self.image_y), np.hypot(self.ideal_x, self.ideal_y)
        )

    @property
    def max_distortion_percent(self) -> float:
        """The distortion (%) of largest size, with its sign; NaN if there is none.

        Both the grid and the curve are searched, so a maximum part way out is found too.
        """
        values = np.concatenate(
            [self.grid_distortion_percent.ravel(), self.height_distortion_percent]
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float("nan")
        return float(values[np.argmax(np.abs(values))])

    @property
    def blocked_point_count(self) -> int:
        """How many grid points have no image because their light is blocked."""
        return int(np.count_nonzero(~np.isfinite(self.image_y)))


def compute_distortion(
    state: OpticalState, reference: str = "chief_ray", grid_points: int = GRID_POINTS
) -> DistortionResult:
    """Image a square grid of object points and measure how distorted it is.

    The selected field gives the wavelength, and the NA used for centroids. All the points
    are traced together, one ray each for chief rays and a spot's worth each for centroids.

    Parameters
    ----------
    state : OpticalState
        The bench. Its fields must be object points, at least one off the axis.
    reference : str
        ``"chief_ray"`` (default) or ``"centroid"``, the centroid of the point's spot.
    grid_points : int
        Object points along each side of the grid, at least 2.

    Returns
    -------
    DistortionResult
        With empty arrays and an ``unavailable_reason`` if there is no grid to image.

    Raises
    ------
    ValueError
        If ``reference`` or ``grid_points`` is not allowed.
    """
    if reference not in DISTORTION_REFERENCES:
        raise ValueError(
            f"Unknown reference {reference!r}; choose one of: {', '.join(DISTORTION_REFERENCES)}"
        )
    if grid_points < 2:
        raise ValueError("A grid needs at least 2 points along each side")
    reason = _why_unavailable(state)
    if reason:
        return _no_result(reference, reason)

    field_radius_mm = _field_of_view_radius_mm(state)
    # The square whose corners are on the circle of that radius.
    half_width_mm = field_radius_mm / np.sqrt(2.0)
    side = np.linspace(-half_width_mm, half_width_mm, grid_points)
    object_x, object_y = np.meshgrid(side, side)
    heights = np.linspace(0.0, field_radius_mm, HEIGHT_CURVE_POINTS)
    near_axis_mm = NEAR_AXIS_FRACTION * field_radius_mm

    # One trace for the near-axis point, then the curve's points (up the y axis), then the grid.
    all_x = np.concatenate([[0.0], np.zeros(heights.size), object_x.ravel()])
    all_y = np.concatenate([[near_axis_mm], heights, object_y.ravel()])
    image_x, image_y = _image_points(state, build_optic(state), all_x, all_y, reference)

    magnification = float(image_y[0] / near_axis_mm)
    curve = slice(1, 1 + heights.size)
    grid = slice(curve.stop, None)
    return DistortionResult(
        reference=reference,
        magnification=magnification,
        object_x=object_x,
        object_y=object_y,
        image_x=image_x[grid].reshape(object_x.shape),
        image_y=image_y[grid].reshape(object_y.shape),
        heights=heights,
        height_distortion_percent=_distortion_percent(
            np.hypot(image_x[curve], image_y[curve]), np.abs(magnification) * heights
        ),
    )


def _field_of_view_radius_mm(state: OpticalState) -> float:
    """How far from the axis (mm) the farthest of the design's object points is."""
    return max(field_height_mm(light) for light in state.fields)


def _why_unavailable(state: OpticalState) -> str:
    """Why the design has no grid to image, in a few words for a plot title ("" if it has)."""
    if not state.has_finite_object:
        return "Needs a finite object (object points)"
    if not state.lenses:
        return "No lens on the bench"
    if source_to_lens_mm(state) <= 0.0:
        return "The object is on the first lens"
    if _field_of_view_radius_mm(state) == 0.0:
        return "Move an object point off the axis"
    return ""


def _no_result(reference: str, reason: str) -> DistortionResult:
    """A result with nothing in it but the reason why."""
    no_grid = np.empty((0, 0))
    return DistortionResult(
        reference=reference,
        magnification=float("nan"),
        object_x=no_grid,
        object_y=no_grid,
        image_x=no_grid,
        image_y=no_grid,
        heights=np.empty(0),
        height_distortion_percent=np.empty(0),
        unavailable_reason=reason,
    )


def _image_points(
    state: OpticalState, optic: Optic, object_x: np.ndarray, object_y: np.ndarray, reference: str
) -> tuple[np.ndarray, np.ndarray]:
    """Where each object point (mm) is imaged: (x, y) in mm, NaN if no light gets through.

    Each point sends out either its chief ray or a spot's pattern of rays, and its image is
    the mean position of those that reach the image plane (for one ray, just its position).
    """
    if reference == "chief_ray":
        pupil_x, pupil_y = np.zeros(1), np.zeros(1)
    else:
        pupil_x, pupil_y = pupil_points("hexapolar", SPOT_RINGS)

    selected = state.selected_field
    assert isinstance(selected, ObjectPoint)  # checked by _why_unavailable
    bundles = [
        launch_from_point(
            state,
            replace(selected, height_x_mm=float(x), height_y_mm=float(y)),
            optic,
            pupil_x,
            pupil_y,
        )
        for x, y in zip(object_x, object_y)
    ]
    rays = trace_rays(optic, join_rays(bundles))

    # One row for each object point, one column for each of its rays.
    shape = (object_x.size, pupil_x.size)
    getting_through = transmitted(rays).reshape(shape)
    ray_count = getting_through.sum(axis=1)
    image = []
    for positions in (rays.x, rays.y):
        positions = np.where(
            getting_through, np.asarray(positions, dtype=float).reshape(shape), 0.0
        )
        with np.errstate(invalid="ignore"):  # 0 / 0, giving NaN, where no ray gets through
            image.append(positions.sum(axis=1) / ray_count)
    return image[0], image[1]


def _distortion_percent(image_radius: np.ndarray, ideal_radius: np.ndarray) -> np.ndarray:
    """100 (image - ideal) / ideal, for distances from the axis (mm).

    On the axis this is 0 / 0; its limit there, 0, is given (unless the point is blocked).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        percent = 100.0 * (image_radius - ideal_radius) / ideal_radius
    on_axis = (ideal_radius == 0.0) & np.isfinite(image_radius)
    return np.where(on_axis, 0.0, percent)
