"""Assemblies: runs of neighbouring lenses grouped so that they move and flip as one.

An assembly is a named run of two or more neighbouring lenses that share an
``assembly_id`` (``raycrafter.core.state``). The gaps inside it are always locked, so it
moves as one block (``raycrafter.core.gaps``). It can be flipped as a whole, which reverses
the order of its lenses and turns each one round, and it can be saved as a user part
(``raycrafter.core.user_parts``) to be inserted again later. A Plossl pair is an assembly
named PLOSSL_NAME. It has no Qt code.
"""

from __future__ import annotations

from raycrafter.core.state import OpticalState

PLOSSL_NAME = "Plossl pair"


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
