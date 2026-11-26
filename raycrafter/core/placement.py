"""Where lenses and the source plane may go on the axis: making, adding, inserting, deleting.

Every function works on an ``OpticalState`` and has no Qt code. ``OpticalModel``
(``raycrafter.core.model``) offers them as methods; the rules are documented here.

Two limits hold at the front of the bench. The first lens stays FIRST_LENS_MIN_Z from the
start of the axis, so it can still be dragged. The source plane, where the fields are
defined (``OpticalState.source_z``), is never behind the first lens. While the gap from the
source plane is locked, the source plane keeps its distance in front of whichever lens is
first (``keep_source_gap``).
"""

from __future__ import annotations

import numpy as np

from raycrafter.core import assemblies, gaps
from raycrafter.core.prescription import prescription_from_zmx_text
from raycrafter.core.state import AXIS_START_Z, OpticalState, PlacedLens
from raycrafter.lenscatalog.catalog import CatalogEntry

INITIAL_LENS_Z = 100.0  # mm, front vertex of the first lens
LENS_GAP_MM = 10.0  # air gap left between a new lens and its left neighbour
IMAGE_CLEARANCE_MM = 50.0  # a new lens pushes the image plane at least this far behind it
FIRST_LENS_MIN_Z = AXIS_START_Z + 20.0  # lenses cannot be dragged closer to the axis start
PARAXIAL_SEMI_APERTURE_MM = 25.4  # paraxial lenses are by default the size of a 2-inch optic


# ---------------------------------------------------------------------- making lenses


def make_lens(entry: CatalogEntry, front_z: float = INITIAL_LENS_Z) -> PlacedLens:
    """A catalogue lens at the given front-vertex z, not yet on the bench."""
    return PlacedLens(
        part_number=entry.part_number,
        surfaces=prescription_from_zmx_text(entry.zmx_text),
        semi_aperture=entry.semi_aperture,
        z=front_z,
        is_negative=entry.is_negative,
        vendor=entry.vendor,
        mount_type=entry.mount_type,
    )


def make_paraxial_lens(
    focal_length_mm: float,
    z: float = INITIAL_LENS_Z,
    diameter_mm: float = 2.0 * PARAXIAL_SEMI_APERTURE_MM,
) -> PlacedLens:
    """An ideal thin lens of the given focal length and clear diameter at ``z``.

    The lens is not yet on the bench. A paraxial lens focuses perfectly (no aberrations) and
    has zero thickness. By default it is as wide as a 2-inch optic.

    Raises
    ------
    ValueError
        If the focal length is zero or the diameter is not positive.
    """
    lens = PlacedLens(
        part_number="",
        surfaces=[],
        semi_aperture=PARAXIAL_SEMI_APERTURE_MM,
        z=z,
        is_negative=False,
    )
    _set_paraxial_properties(lens, focal_length_mm, diameter_mm)
    return lens


def set_paraxial_lens(
    state: OpticalState, lens_index: int, focal_length_mm: float, diameter_mm: float
) -> None:
    """Give a paraxial lens on the bench a new focal length and clear diameter.

    It stays where it is: a paraxial lens has no thickness, so neither change moves it or
    its neighbours.

    Raises
    ------
    ValueError
        If the lens is not paraxial, the focal length is zero or the diameter not positive.
    """
    lens = state.lenses[lens_index]
    if not lens.is_paraxial:
        raise ValueError(f"{lens.part_number} is not a paraxial lens")
    _set_paraxial_properties(lens, focal_length_mm, diameter_mm)


def _set_paraxial_properties(lens: PlacedLens, focal_length_mm: float, diameter_mm: float) -> None:
    """Set a paraxial lens's focal length and diameter, and the name that shows them."""
    if focal_length_mm == 0.0:
        raise ValueError("A paraxial lens needs a non-zero focal length")
    if diameter_mm <= 0.0:
        raise ValueError("A paraxial lens needs a diameter greater than 0 mm")
    lens.part_number = f"Paraxial f={focal_length_mm:+g} mm"
    lens.paraxial_focal_length_mm = focal_length_mm
    lens.is_negative = focal_length_mm < 0
    lens.semi_aperture = diameter_mm / 2.0


# ---------------------------------------------------------------------- the front limits


def source_gap_mm(state: OpticalState) -> float:
    """The gap from the source plane to the first lens (mm); 0 with no lenses."""
    gap = gaps.source_gap(state)
    return 0.0 if gap is None else gap.width_mm


def keep_source_gap(state: OpticalState, gap_mm: float) -> None:
    """After the first lens changed: with the source gap locked, put the source plane
    ``gap_mm`` in front of the new first lens (if there is one)."""
    if state.source_gap_locked and state.lenses:
        state.source_z = state.lenses[0].z - gap_mm


def lowest_first_front_z(state: OpticalState) -> float:
    """The smallest z (mm) the front vertex of the first lens may have.

    It must stay FIRST_LENS_MIN_Z from the start of the axis. With the source gap unlocked
    it cannot pass the source plane; with it locked, the source plane, which moves with it,
    cannot pass the start of the axis.
    """
    if state.source_gap_locked:
        return max(FIRST_LENS_MIN_Z, AXIS_START_Z + source_gap_mm(state))
    return max(FIRST_LENS_MIN_Z, state.source_z)


def refuse_if_image_gap_locked(state: OpticalState, action: str) -> None:
    """Raise ValueError, saying the user cannot do ``action``, if the image gap is locked."""
    if state.lenses and state.image_gap_locked:
        raise ValueError(f"The gap to the image plane is locked: unlock it to {action}")


def set_source_gap(state: OpticalState, gap_mm: float) -> None:
    """Set the gap from the source plane to the first lens (see ``gaps.set_source_gap``).

    Raises
    ------
    ValueError
        As ``gaps.set_source_gap``, and also if the first lens would end up closer than
        FIRST_LENS_MIN_Z to the start of the axis (where it could not be dragged).
    """
    if state.lenses and state.source_z + gap_mm < FIRST_LENS_MIN_Z - 1e-9:
        smallest_mm = FIRST_LENS_MIN_Z - state.source_z
        raise ValueError(
            f"This gap must be at least {smallest_mm:g} mm, to keep the first lens "
            f"{FIRST_LENS_MIN_Z - AXIS_START_Z:g} mm from the start of the axis; move the "
            "source plane forward first"
        )
    gaps.set_source_gap(state, gap_mm)


def set_source_z(state: OpticalState, source_z: float) -> None:
    """Move the source plane, where every field is defined, to ``source_z`` (mm).

    Raises
    ------
    ValueError
        If the gap from the source plane is locked, or ``source_z`` is in front of the
        start of the axis or behind the first lens.
    """
    if state.source_gap_locked:
        raise ValueError("The gap from the source plane is locked: unlock it to move it")
    highest_z = state.lenses[0].z if state.lenses else np.inf
    if not AXIS_START_Z <= source_z <= highest_z + 1e-9:
        raise ValueError(
            f"The source plane must be between the start of the axis ({AXIS_START_Z:g} mm) "
            f"and the first lens ({highest_z:g} mm)"
        )
    state.source_z = min(source_z, highest_z)


# ---------------------------------------------------------------------- adding and removing


def append_lens(state: OpticalState, new_lens: PlacedLens) -> None:
    """Put a lens just to the right of the rightmost lens (at INITIAL_LENS_Z on an empty bench).

    The lens's own z is replaced. The image plane is pushed back if the new lens would
    otherwise end up beyond it.

    Raises
    ------
    ValueError
        If the gap to the image plane is locked, as the new lens would split it.
    """
    refuse_if_image_gap_locked(state, "add a lens behind the last one")
    lenses = state.lenses
    if lenses:
        new_lens.z = lenses[-1].back_z + LENS_GAP_MM
    else:
        new_lens.z = max(INITIAL_LENS_Z, lowest_first_front_z(state))
    lenses.append(new_lens)
    if len(lenses) == 1:
        keep_source_gap(state, 0.0)  # a locked source plane starts at the first lens
    state.image_z = max(state.image_z, new_lens.back_z + IMAGE_CLEARANCE_MM)


def free_front_z(
    state: OpticalState, thickness_mm: float, requested_front_z: float, max_back_z: float = np.inf
) -> float | None:
    """The front-vertex z nearest the request at which a new lens fits without overlap.

    The free stretches of axis are the gaps between the lenses on the bench, plus the
    stretch before the first lens (from ``lowest_first_front_z``) and after the last one (up
    to ``max_back_z``). A locked gap (such as the one inside a Plossl pair) is not free, as
    a new lens would split it; nor is the stretch behind the last lens if the gap to the
    image plane is locked. A new lens may touch its neighbours, as a dragged lens may.

    Parameters
    ----------
    state : OpticalState
        The bench.
    thickness_mm : float
        Front-to-back vertex thickness of the new lens.
    requested_front_z : float
        Where the front vertex is wanted (e.g. under the pointer).
    max_back_z : float
        The new lens's back vertex must not go beyond this z.

    Returns
    -------
    float or None
        The nearest allowed front-vertex z, or None if the lens fits nowhere.
    """
    lenses = state.lenses

    # Occupied stretches of axis, one per block of lenses joined by locked gaps.
    occupied = [
        (lenses[members[0]].z, lenses[members[-1]].back_z) for members in gaps.blocks(state)
    ]

    gap_starts = [lowest_first_front_z(state)] + [back for _front, back in occupied]
    gap_ends = [block_front for block_front, _back in occupied] + [max_back_z]
    if lenses and state.image_gap_locked:
        gap_starts, gap_ends = gap_starts[:-1], gap_ends[:-1]

    # In each gap the lens is wide enough for, the closest position to the request; then the
    # closest of those.
    candidates = []
    for gap_start, gap_end in zip(gap_starts, gap_ends):
        latest_front_z = gap_end - thickness_mm
        if latest_front_z >= gap_start:
            candidates.append(float(np.clip(requested_front_z, gap_start, latest_front_z)))
    if not candidates:
        return None
    return min(candidates, key=lambda front_z: abs(front_z - requested_front_z))


def insert_lenses(
    state: OpticalState,
    new_lenses: list[PlacedLens],
    max_image_z: float = np.inf,
    assembly_name: str | None = None,
) -> list[int]:
    """Put a run of lenses (one lens, or a user part) on the bench at their own z, as one block.

    The lenses must follow each other without overlapping; the caller chooses a free
    position (see ``free_front_z``). Two or more lenses become an assembly called
    ``assembly_name``. If the lenses land behind the image plane, the image plane is pushed
    back to IMAGE_CLEARANCE_MM behind them, but not beyond ``max_image_z`` (unless even that
    is too close to the lens, in which case it goes just behind the lens).

    Returns
    -------
    list of int
        The indices of the new lenses.

    Raises
    ------
    ValueError
        If the lenses would land in a locked gap or overlap a lens on the bench; nothing is
        changed then.
    """
    lenses = state.lenses
    new_index = sum(1 for lens in lenses if lens.z < new_lenses[0].z)
    if new_index == 0 and new_lenses[0].z < lowest_first_front_z(state) - 1e-9:
        raise ValueError("A lens cannot go in front of the source plane or the axis start")
    if new_index == len(lenses):
        if lenses:
            refuse_if_image_gap_locked(state, "add a lens behind the last one")
    elif new_index > 0 and lenses[new_index - 1].locked_to_next:
        raise ValueError("That gap is locked: unlock it to put a lens in it")
    # Allow for rounding, as a lens may be placed exactly touching its neighbours.
    previous_back_z = lenses[new_index - 1].back_z if new_index > 0 else -np.inf
    next_front_z = lenses[new_index].z if new_index < len(lenses) else np.inf
    if new_lenses[0].z < previous_back_z - 1e-9 or new_lenses[-1].back_z > next_front_z + 1e-9:
        raise ValueError("The new lenses would overlap a lens on the bench")

    gap_from_source_mm = source_gap_mm(state)
    lenses[new_index:new_index] = new_lenses
    if new_index == 0:
        keep_source_gap(state, gap_from_source_mm)
    new_indices = list(range(new_index, new_index + len(new_lenses)))
    if len(new_lenses) > 1:
        assemblies.make_assembly(state, new_indices[0], new_indices[-1], assembly_name)

    minimum_image_z = state.last_back_z() + gaps.MIN_IMAGE_GAP_MM
    if state.image_z < minimum_image_z:
        pushed_image_z = min(new_lenses[-1].back_z + IMAGE_CLEARANCE_MM, max_image_z)
        state.image_z = max(pushed_image_z, minimum_image_z)
    return new_indices


def delete_lens(state: OpticalState, lens_index: int) -> None:
    """Remove one lens, and the locks on the gaps either side of it.

    Its assembly, if any, is dissolved; the other gaps inside it stay locked. The gap from
    the source plane is the exception: if it is locked, the source plane keeps the same
    distance in front of whichever lens is now first.
    """
    assemblies.split_assembly(state, lens_index, unlock=False)
    gap_from_source_mm = source_gap_mm(state)
    lenses = state.lenses
    if lens_index > 0:
        lenses[lens_index - 1].locked_to_next = False
    if lens_index == len(lenses) - 1:
        state.image_gap_locked = False
    del lenses[lens_index]
    if lens_index == 0:
        keep_source_gap(state, gap_from_source_mm)
