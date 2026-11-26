"""Assemblies: runs of neighbouring lenses grouped so that they move and flip as one.

An assembly is a named run of two or more neighbouring lenses that share an
``assembly_id`` (``raycrafter.core.state``). The gaps inside it are always locked, so it
moves as one block (``raycrafter.core.gaps``). It can be flipped as a whole, which reverses
the order of its lenses and turns each one round, and it can be saved as a user part
(``raycrafter.core.user_parts``) to be inserted again later. A Plossl pair is an assembly
named PLOSSL_NAME, made by ``make_plossl``. Flipping a single lens (``flip_lens``) is here
too. It has no Qt code.
"""

from __future__ import annotations

from raycrafter.core import gaps
from raycrafter.core.state import OpticalState, PlacedLens

PLOSSL_NAME = "Plossl pair"
PLOSSL_GAP_MM = 1.0  # closest distance between the two facing surfaces of a Plossl pair
PLOSSL_SEARCH_WIDTHS = 2.0  # a Plossl partner must be within this many lens thicknesses


def assembly_indices(state: OpticalState, lens_index: int) -> list[int]:
    """The indices of the lenses in this lens's assembly, in order (just itself if none)."""
    assembly_id = state.lenses[lens_index].assembly_id
    if assembly_id is None:
        return [lens_index]
    return [index for index, lens in enumerate(state.lenses) if lens.assembly_id == assembly_id]


def assembly_name(state: OpticalState, lens_index: int) -> str | None:
    """The name of this lens's assembly, or None if it is in none."""
    assembly_id = state.lenses[lens_index].assembly_id
    return None if assembly_id is None else state.assembly_names[assembly_id]


def make_assembly(
    state: OpticalState, first_index: int, last_index: int, name: str | None = None
) -> int:
    """Group the lenses from ``first_index`` to ``last_index`` (inclusive) into an assembly.

    The gaps between them are locked. Assemblies wholly inside the run are absorbed into the
    new one. Nothing is moved.

    Parameters
    ----------
    state : OpticalState
        The bench to change.
    first_index, last_index : int
        The first and last lens of the run.
    name : str, optional
        The assembly's name; "Assembly <id>" if not given.

    Returns
    -------
    int
        The new assembly's id.

    Raises
    ------
    ValueError
        If the run has fewer than two lenses, goes beyond the bench, or would cut an
        existing assembly in two. Nothing is changed then.
    """
    lenses = state.lenses
    if not 0 <= first_index < last_index < len(lenses):
        raise ValueError(
            f"An assembly needs two or more neighbouring lenses; lenses {first_index} to "
            f"{last_index} are not that"
        )
    for edge_index in (first_index, last_index):
        members = assembly_indices(state, edge_index)
        if members[0] < first_index or members[-1] > last_index:
            raise ValueError(
                f"Lens {edge_index} is in an assembly that reaches beyond lenses "
                f"{first_index} to {last_index}: include all of it, or split it first"
            )

    for absorbed_id in {lens.assembly_id for lens in lenses[first_index : last_index + 1]}:
        if absorbed_id is not None:
            del state.assembly_names[absorbed_id]

    assembly_id = state.next_assembly_id
    state.next_assembly_id += 1
    state.assembly_names[assembly_id] = name if name else f"Assembly {assembly_id}"
    for index in range(first_index, last_index + 1):
        lenses[index].assembly_id = assembly_id
        lenses[index].locked_to_next = index < last_index
    return assembly_id


def split_assembly(state: OpticalState, lens_index: int, unlock: bool = True) -> None:
    """Dissolve this lens's assembly; its lenses stay where they are.

    Parameters
    ----------
    state : OpticalState
        The bench to change.
    lens_index : int
        Any lens of the assembly. Nothing happens if it is in none.
    unlock : bool
        If True (the default), also unlock the gaps inside the assembly, so that its lenses
        move independently again.
    """
    assembly_id = state.lenses[lens_index].assembly_id
    if assembly_id is None:
        return
    members = assembly_indices(state, lens_index)
    for index in members:
        state.lenses[index].assembly_id = None
        if unlock and index < members[-1]:
            state.lenses[index].locked_to_next = False
    del state.assembly_names[assembly_id]


def flip_assembly(state: OpticalState, lens_index: int) -> list[int]:
    """Turn this lens's assembly round as a whole, in the same stretch of the axis.

    The order of the lenses is reversed and each catalogue lens is flipped, so the assembly
    is the mirror image of what it was, with the gaps inside it in reverse order. A paraxial
    lens is the same both ways round, so only its position changes. Flipping twice restores
    the assembly. A lens in no assembly is flipped on its own.

    Returns
    -------
    list of int
        The new order as a list of old lens indices, so the caller can reorder anything it
        keeps per lens.
    """
    lenses = state.lenses
    members = assembly_indices(state, lens_index)
    first, last = members[0], members[-1]
    start_z, end_z = lenses[first].z, lenses[last].back_z
    gap_after_locked = lenses[last].locked_to_next  # the gap behind the whole assembly

    # Mirror each lens about the middle of the stretch: its back vertex lands where the
    # mirror image of its front vertex was.
    for index in members:
        lens = lenses[index]
        lens.z = start_z + end_z - lens.back_z
        if not lens.is_paraxial:
            lens.flipped = not lens.flipped

    lenses[first : last + 1] = lenses[first : last + 1][::-1]
    # A lock belongs to a gap, not to a lens. The gaps inside the assembly are all locked,
    # and the gap behind it is unchanged, so set the flags by position again.
    for index in members:
        lenses[index].locked_to_next = index < last or gap_after_locked
    return list(range(first)) + members[::-1] + list(range(last + 1, len(lenses)))


def flip_lens(state: OpticalState, lens_index: int) -> None:
    """Turn one lens around.

    Lenses in an assembly cannot be flipped on their own (see ``flip_assembly``), and a
    paraxial lens is the same both ways round, so neither changes.
    """
    lens = state.lenses[lens_index]
    if lens.assembly_id is None and not lens.is_paraxial:
        lens.flipped = not lens.flipped


# ---------------------------------------------------------------------- Plossl pairs


def in_plossl_pair(state: OpticalState, lens_index: int) -> bool:
    """True if the lens is one of a Plossl pair."""
    return assembly_name(state, lens_index) == PLOSSL_NAME


def plossl_partner(state: OpticalState, lens_index: int) -> int | None:
    """Index of a lens this one could form a Plossl pair with, or None.

    The partner must be an immediate neighbour, be the same part, and be in no assembly;
    the gap between the two must be at most PLOSSL_SEARCH_WIDTHS lens thicknesses. Only
    positive catalogue lenses can form a Plossl pair (a Plossl is made of doublets, so
    paraxial lenses are excluded).
    """
    lenses = state.lenses
    lens = lenses[lens_index]
    if lens.assembly_id is not None or lens.is_negative or lens.is_paraxial:
        return None
    for other_index in (lens_index - 1, lens_index + 1):
        if not 0 <= other_index < len(lenses):
            continue
        other = lenses[other_index]
        left, right = sorted((lens, other), key=lambda item: item.z)
        gap = right.z - left.back_z
        if (
            other.part_number == lens.part_number
            and other.assembly_id is None
            and gap <= PLOSSL_SEARCH_WIDTHS * lens.thickness
        ):
            return other_index
    return None


def make_plossl(state: OpticalState, lens_index: int) -> None:
    """Snap the lens and its partner (``plossl_partner``) into a Plossl pair and group them.

    A Plossl pair is two identical doublets facing each other symmetrically, with their
    positive (crown) elements on the inside (``face_as_plossl``). The right lens is moved so
    the closest points of the two facing surfaces are PLOSSL_GAP_MM apart, and everything
    downstream (later lenses and the image plane) shifts with it, even if the gap between
    the two was locked. The two then form an assembly named PLOSSL_NAME. Does nothing if the
    lens has no partner.
    """
    partner_index = plossl_partner(state, lens_index)
    if partner_index is None:
        return
    left_index, right_index = sorted((lens_index, partner_index))
    lenses = state.lenses
    left, right = lenses[left_index], lenses[right_index]

    original_right_z = right.z
    face_as_plossl(left, right)
    shift = right.z - original_right_z
    for downstream in lenses[right_index + 1 :]:
        downstream.z += shift
    minimum_image_z = state.last_back_z() + gaps.MIN_IMAGE_GAP_MM
    state.image_z = max(state.image_z + shift, minimum_image_z)

    make_assembly(state, left_index, right_index, PLOSSL_NAME)


def face_as_plossl(left: PlacedLens, right: PlacedLens) -> None:
    """Turn two identical doublets to face each other as a Plossl pair, PLOSSL_GAP_MM apart.

    The left lens is flipped if needed so its positive element is at its back, and the right
    lens so its positive element is at its front. The right lens is then moved so that the
    closest points of the facing surfaces are PLOSSL_GAP_MM apart. The gap is measured along
    z across the smaller clear aperture: at the vertices for two convex surfaces, nearer the
    rim if either is concave.
    """
    left.flipped = left.positive_element_is_first()  # moves the positive element to the back
    right.flipped = not right.positive_element_is_first()  # keeps or puts it at the front
    right.z = left.back_z + PLOSSL_GAP_MM - gaps.closest_sag_difference_mm(left, right)
