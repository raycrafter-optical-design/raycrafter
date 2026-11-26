"""Moving lenses along the axis, as dragging them does, and swapping neighbouring blocks.

A lens moves with its *block*: the lenses joined to it by locked gaps (``gaps.block_indices``).
Dragged past the centre of a neighbouring block, the two swap places. Every function works
on an ``OpticalState`` and has no Qt code; ``OpticalModel`` offers them as methods.
"""

from __future__ import annotations

import numpy as np

from raycrafter.core import gaps
from raycrafter.core.placement import lowest_first_front_z
from raycrafter.core.state import OpticalState


def block_centre_z(state: OpticalState, members: list[int]) -> float:
    """z of the middle of a block of adjacent lenses, from its first front to last back."""
    lenses = state.lenses
    return 0.5 * (lenses[members[0]].z + lenses[members[-1]].back_z)


def move_lens(state: OpticalState, lens_index: int, requested_front_z: float) -> list[int] | None:
    """Move a lens (and its block) so this lens's front vertex is near the request.

    The first lens cannot pass ``placement.lowest_first_front_z``, and with the source gap
    locked the source plane moves with it. If the request carries the block's centre past
    the centre of the neighbouring block, the two swap places (``swap_neighbouring_blocks``).
    Otherwise the block stays between its neighbours (nothing overlaps), and the last lens
    stays before the image plane; if the allowed range is empty the lens does not move. If
    the gap to the image plane is locked, the image plane moves with the last block instead.

    Returns
    -------
    list of int or None
        After a swap, the new order as a list of old indices, so the caller can reorder
        anything it keeps per lens. None when the order is unchanged.
    """
    lenses = state.lenses
    members = gaps.block_indices(state, lens_index)
    first, last = members[0], members[-1]
    is_last_block = last == len(lenses) - 1

    shift_wanted = requested_front_z - lenses[lens_index].z
    wanted_centre_z = block_centre_z(state, members) + shift_wanted

    # Dragged past the centre of the next block to the right, or the previous one?
    if not is_last_block:
        right_members = gaps.block_indices(state, last + 1)
        if wanted_centre_z > block_centre_z(state, right_members):
            return swap_neighbouring_blocks(state, members, right_members)
    if first > 0:
        left_members = gaps.block_indices(state, first - 1)
        if wanted_centre_z < block_centre_z(state, left_members):
            return swap_neighbouring_blocks(state, left_members, members)

    block_width = lenses[last].back_z - lenses[first].z
    lowest_front_z = lenses[first - 1].back_z if first > 0 else lowest_first_front_z(state)
    image_moves_too = is_last_block and state.image_gap_locked
    if image_moves_too:
        next_front_z = np.inf
    elif not is_last_block:
        next_front_z = lenses[last + 1].z
    else:
        next_front_z = state.image_z - gaps.MIN_IMAGE_GAP_MM
    highest_front_z = next_front_z - block_width

    shift_min = lowest_front_z - lenses[first].z
    shift_max = highest_front_z - lenses[first].z
    if shift_max < shift_min:
        return None
    shift = float(np.clip(shift_wanted, shift_min, shift_max))
    for index in members:
        lenses[index].z += shift
    if image_moves_too:
        state.image_z += shift
    if first == 0 and state.source_gap_locked:
        state.source_z += shift  # the source plane moves with the first lens
    return None


def swap_neighbouring_blocks(
    state: OpticalState, left_members: list[int], right_members: list[int]
) -> list[int]:
    """Exchange two adjacent blocks of lenses, so the left one ends up on the right.

    Each block keeps its internal spacing and locks. The air gap between them is kept, and
    the two together occupy exactly the same stretch of axis as before, so nothing
    downstream (later lenses, the image plane) has to move.

    Returns
    -------
    list of int
        The new order as a list of old lens indices.
    """
    lenses = state.lenses
    left_block = [lenses[index] for index in left_members]
    right_block = [lenses[index] for index in right_members]

    left_start = left_block[0].z
    right_start = right_block[0].z
    gap = right_start - left_block[-1].back_z
    right_width = right_block[-1].back_z - right_start

    # The old right block now starts where the old left one did; the old left one follows.
    for lens in right_block:
        lens.z += left_start - right_start
    for lens in left_block:
        lens.z += right_width + gap

    low, high = left_members[0], right_members[-1]
    lenses[low : high + 1] = right_block + left_block
    return list(range(low)) + right_members + left_members + list(range(high + 1, len(lenses)))
