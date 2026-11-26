"""Plossl pairs: making one out of a single lens, and replacing both its lenses.

A Plossl pair is two identical positive lenses facing each other symmetrically. For doublets
the positive (crown) elements are on the inside; for singlets the more strongly curved faces
are. It is an assembly (``raycrafter.core.assemblies``) named PLOSSL_NAME, so it moves and
flips as one, and "Split assembly" turns it back into two separate lenses. It has no Qt code.

Where the pair goes
-------------------
``make_plossl`` duplicates a lens. The pair is centred on the optical centre of the original
lens, taken as the point midway between its two principal planes. A locked gap on either side
keeps its length: the lenses joined by it (and the source or image plane, if locked to them)
move outwards with that end of the pair. Nothing else moves, except that the image plane is
pushed back if the pair would reach it. If the pair would overlap a neighbour or the start of
the bench, it is refused and nothing changes.
"""

from __future__ import annotations

import copy

from raycrafter.core import assemblies, gaps, placement, replacement
from raycrafter.core.state import OpticalState, PlacedLens

PLOSSL_GAP_MM = 1.0  # closest distance between the two facing surfaces of a Plossl pair
OVERLAP_TOLERANCE_MM = 1e-9  # surfaces may touch, allowing for rounding


def reason_not_possible(state: OpticalState, lens_index: int) -> str | None:
    """Why this lens cannot be made into a Plossl pair, or None if it can be tried.

    It must be a real (not paraxial), positive lens that is in no assembly. Whether the pair
    also fits between the neighbours is only found out by ``make_plossl``.
    """
    lens = state.lenses[lens_index]
    if lens.is_paraxial:
        return "A paraxial lens cannot be made into a Plossl pair"
    if lens.is_negative:
        return "A Plossl pair needs a positive lens"
    if lens.assembly_id is not None:
        return "This lens is already in an assembly: split it first"
    return None


def make_plossl(state: OpticalState, lens_index: int) -> None:
    """Turn a lens into a Plossl pair: two copies of it, facing each other.

    The pair takes the lens's place, as the module docstring describes, and becomes an
    assembly named PLOSSL_NAME. It occupies lenses ``lens_index`` and ``lens_index + 1``;
    the later lenses' indices go up by one.

    Parameters
    ----------
    state : OpticalState
        The bench to change.
    lens_index : int
        The lens to duplicate.

    Raises
    ------
    ValueError
        If the lens cannot form a Plossl pair (``reason_not_possible``, or it has no power),
        or the pair does not fit. Nothing is changed then.
    """
    reason = reason_not_possible(state, lens_index)
    if reason is not None:
        raise ValueError(reason)
    lenses = state.lenses
    original = lenses[lens_index]
    planes = replacement.principal_planes_z([original], state.selected_field.wavelength_nm)
    if planes is None:
        raise ValueError(f"{original.part_number} has no power, so it cannot form a Plossl pair")
    optical_centre_z = 0.5 * (planes[0] + planes[1])

    left, right = copy.deepcopy(original), copy.deepcopy(original)
    face_as_plossl(left, right)
    # The pair is mirror symmetric, so its own optical centre is midway between its ends.
    shift = optical_centre_z - 0.5 * (left.z + right.back_z)
    left.z += shift
    right.z += shift
    front_shift = left.z - original.z  # how far the front end moves (negative: forwards)
    back_shift = right.back_z - original.back_z

    # The lenses joined to this one by locked gaps move with the nearer end of the pair.
    block = gaps.block_indices(state, lens_index)
    first_moving, last_moving = block[0], block[-1]
    in_front = lenses[first_moving:lens_index]
    behind = lenses[lens_index + 1 : last_moving + 1]
    image_moves_too = last_moving == len(lenses) - 1 and state.image_gap_locked
    gap_from_source_mm = placement.source_gap_mm(state)
    lowest_front_z = placement.lowest_first_front_z(state)

    # Everything is done on copies first, so that a refusal leaves the bench as it was.
    moved_front = [copy.deepcopy(lens) for lens in in_front]
    moved_behind = [copy.deepcopy(lens) for lens in behind]
    for lens in moved_front:
        lens.z += front_shift
    for lens in moved_behind:
        lens.z += back_shift
    new_lenses = (
        lenses[:first_moving]
        + moved_front
        + [left, right]
        + moved_behind
        + lenses[last_moving + 1 :]
    )

    # Refuse if the surfaces would now overlap in front of or behind the pair, or at either
    # end of the lenses that moved with it. ``new_last`` allows for the extra lens.
    new_last = last_moving + 1
    if first_moving == 0 and new_lenses[0].z < lowest_front_z - OVERLAP_TOLERANCE_MM:
        raise ValueError(
            f"A Plossl pair of {original.part_number} does not fit here: it would reach past "
            "the start of the bench. Move the lens back first"
        )
    for gap_index in {first_moving - 1, lens_index - 1, lens_index + 1, new_last}:
        if 0 <= gap_index < len(new_lenses) - 1:
            _refuse_overlap(new_lenses[gap_index], new_lenses[gap_index + 1], original)

    state.lenses = new_lenses
    assemblies.make_assembly(state, lens_index, lens_index + 1, assemblies.PLOSSL_NAME)
    # make_assembly leaves the gap behind the pair unlocked: it keeps the original's lock.
    right.locked_to_next = original.locked_to_next

    if image_moves_too:
        state.image_z += back_shift
    else:
        state.image_z = max(state.image_z, state.last_back_z() + gaps.MIN_IMAGE_GAP_MM)
    if first_moving == 0:
        placement.keep_source_gap(state, gap_from_source_mm)


def _refuse_overlap(front: PlacedLens, back: PlacedLens, original: PlacedLens) -> None:
    """Raise ValueError if the facing surfaces of two neighbouring lenses would overlap."""
    gap_mm = back.z - front.back_z
    # The surfaces touch when the gap makes up for the sag of the closest points.
    clearance_mm = gap_mm + min(0.0, gaps.closest_sag_difference_mm(front, back))
    if clearance_mm < -OVERLAP_TOLERANCE_MM:
        raise ValueError(
            f"A Plossl pair of {original.part_number} does not fit here: it needs "
            f"{-clearance_mm:.1f} mm more room. Move the neighbouring lenses away first"
        )


def face_as_plossl(left: PlacedLens, right: PlacedLens) -> None:
    """Turn two identical lenses to face each other as a Plossl pair, PLOSSL_GAP_MM apart.

    The left lens is flipped if needed so its inner face (``_inner_face_is_first``) is at
    its back, and the right lens so that it is at its front. The right lens is then moved so
    that the closest points of the facing surfaces are PLOSSL_GAP_MM apart. The gap is
    measured along z across the smaller clear aperture: at the vertices for two convex
    surfaces, nearer the rim if either is concave.
    """
    left.flipped = _inner_face_is_first(left)  # moves the inner face to the back
    right.flipped = not _inner_face_is_first(right)  # keeps or puts it at the front
    right.z = left.back_z + PLOSSL_GAP_MM - gaps.closest_sag_difference_mm(left, right)


def _inner_face_is_first(lens: PlacedLens) -> bool:
    """True if the face that belongs inside a Plossl pair is the catalogue's front face.

    For a cemented or multi-element lens it is the end with the positive (crown) element.
    Both ends of a singlet are the same element, so there it is the more strongly curved
    face (the curved face of a plano-convex lens).
    """
    glass_count = sum(surface.material_after != "air" for surface in lens.surfaces[:-1])
    if glass_count == 1:
        return abs(1.0 / lens.surfaces[0].radius) >= abs(1.0 / lens.surfaces[-1].radius)
    return lens.positive_element_is_first()


def replace_plossl_pair(state: OpticalState, lens_index: int, new_lens: PlacedLens) -> None:
    """Replace both lenses of this lens's Plossl pair with two copies of ``new_lens``.

    The new pair is set up as ``make_plossl`` does (facing each other, PLOSSL_GAP_MM apart)
    and placed as a replaced lens is (``raycrafter.core.replacement``), with the pair's own
    rear principal plane where the old pair's was. It stays a Plossl pair.

    Raises
    ------
    ValueError
        If the lens is in no Plossl pair, the new lens is paraxial or diverging, or the new
        pair would overlap a neighbour. Nothing is changed then.
    """
    if not assemblies.in_plossl_pair(state, lens_index):
        raise ValueError("This lens is not one of a Plossl pair")
    if new_lens.is_negative or new_lens.is_paraxial:
        raise ValueError(
            f"A Plossl pair needs a positive catalogue lens, and {new_lens.part_number} is not"
        )
    left, right = copy.deepcopy(new_lens), copy.deepcopy(new_lens)
    left.z = 0.0
    face_as_plossl(left, right)
    members = assemblies.assembly_indices(state, lens_index)
    replacement.replace_lenses(state, members[0], members[-1], [left, right])
