"""Air gaps: the distances along the axis between one lens and the next, and to the image.

A gap is measured vertex to vertex: from the back vertex of a lens to the front vertex of the
next one, or, behind the last lens, to the image plane. Each gap is identified by the index of
the lens in front of it. It has no Qt code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from raycrafter.core.prescription import surface_sag
from raycrafter.core.state import OpticalState, PlacedLens

MIN_IMAGE_GAP_MM = 1.0  # the image plane stays at least this far behind the last lens
SURFACE_SAMPLE_COUNT = 200  # heights at which two facing surfaces are compared


@dataclass(frozen=True)
class AirGap:
    """The air gap behind one lens.

    Attributes
    ----------
    lens_index : int
        The lens in front of the gap.
    start_z : float
        z of that lens's back vertex (mm).
    end_z : float
        z of the next lens's front vertex, or of the image plane (mm).
    to_image : bool
        True for the gap between the last lens and the image plane.
    in_plossl_pair : bool
        True for the gap inside a Plossl pair, which cannot be changed.
    """

    lens_index: int
    start_z: float
    end_z: float
    to_image: bool
    in_plossl_pair: bool

    @property
    def width_mm(self) -> float:
        """The length of the gap (mm)."""
        return self.end_z - self.start_z


def air_gaps(state: OpticalState) -> list[AirGap]:
    """The gap behind every lens, left to right; empty if there are no lenses."""
    lenses = state.lenses
    gaps = []
    for index, lens in enumerate(lenses):
        to_image = index == len(lenses) - 1
        next_lens = None if to_image else lenses[index + 1]
        gaps.append(
            AirGap(
                lens_index=index,
                start_z=lens.back_z,
                end_z=state.image_z if next_lens is None else next_lens.z,
                to_image=to_image,
                in_plossl_pair=next_lens is not None
                and lens.group_id is not None
                and lens.group_id == next_lens.group_id,
            )
        )
    return gaps


def closest_sag_difference_mm(left: PlacedLens, right: PlacedLens) -> float:
    """How much closer than their vertices two facing surfaces come, as a number <= 0 (mm).

    The facing surfaces are the back of ``left`` and the front of ``right`` in their current
    orientations. The separation along z at height h is the vertex gap plus
    ``front_sag(h) - back_sag(h)``; this returns the smallest value of that difference across
    the smaller clear aperture. It is zero for two convex or flat surfaces (closest at the
    vertices) and negative if the surfaces approach each other towards the rim.
    """
    heights = np.linspace(0.0, min(left.semi_aperture, right.semi_aperture), SURFACE_SAMPLE_COUNT)
    # A paraxial lens has no surfaces: it is flat.
    back_radius = np.inf if left.is_paraxial else left.surfaces_in_use()[-1].radius
    front_radius = np.inf if right.is_paraxial else right.surfaces_in_use()[0].radius
    difference = surface_sag(front_radius, heights) - surface_sag(back_radius, heights)
    return float(difference.min())


def minimum_gap_mm(state: OpticalState, lens_index: int) -> float:
    """The smallest vertex-to-vertex gap allowed behind a lens (mm).

    Between two lenses it is the gap at which their facing surfaces just touch, which is more
    than zero if either surface is concave. Behind the last lens it is MIN_IMAGE_GAP_MM.
    """
    _check_lens_index(state, lens_index)
    if lens_index == len(state.lenses) - 1:
        return MIN_IMAGE_GAP_MM
    left, right = state.lenses[lens_index], state.lenses[lens_index + 1]
    return max(0.0, -closest_sag_difference_mm(left, right))


def set_gap(state: OpticalState, lens_index: int, gap_mm: float) -> None:
    """Set the gap behind a lens, shifting everything downstream so no other gap changes.

    The later lenses and the image plane move together. Behind the last lens only the image
    plane moves.

    Parameters
    ----------
    state : OpticalState
        The bench to change.
    lens_index : int
        The lens in front of the gap.
    gap_mm : float
        Back vertex to the next front vertex (or the image plane), in mm.

    Raises
    ------
    IndexError
        If there is no such lens.
    ValueError
        If the gap is inside a Plossl pair, is not a number, or is smaller than
        ``minimum_gap_mm``. Nothing is changed then.
    """
    _check_lens_index(state, lens_index)
    gap = air_gaps(state)[lens_index]
    if gap.in_plossl_pair:
        raise ValueError("The gap inside a Plossl pair is fixed: split the pair to change it")
    if not math.isfinite(gap_mm):
        raise ValueError("A gap must be a finite number of mm")

    minimum_mm = minimum_gap_mm(state, lens_index)
    if gap_mm < minimum_mm - 1e-9:  # allow for rounding in a value computed to be the minimum
        shown_minimum = math.ceil(minimum_mm * 1000.0) / 1000.0  # rounded up, so it is accepted
        reason = (
            "to keep the image plane behind the lens"
            if gap.to_image
            else "or the two lenses would overlap"
        )
        raise ValueError(f"This gap must be at least {shown_minimum:g} mm, {reason}")

    shift_mm = gap_mm - gap.width_mm
    for downstream in state.lenses[lens_index + 1 :]:
        downstream.z += shift_mm
    state.image_z += shift_mm


def _check_lens_index(state: OpticalState, lens_index: int) -> None:
    if not 0 <= lens_index < len(state.lenses):
        raise IndexError(f"There is no lens {lens_index}")
