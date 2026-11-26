"""Replacing lenses in place: swap a lens (or a run of lenses) for another on the bench.

This is how a design sketched with paraxial lenses is turned into real parts, or one part is
compared with another of the same focal length (issue #63). It has no Qt code.

Where the new lens goes
-----------------------
A real lens has thickness, so "in place" needs a rule. The new lens is placed so that its
rear principal plane is where the old lens's was. For a paraxial (thin) lens that plane is
the lens itself. A collimated beam then comes to a focus about where it did before, so what
changes is the aberrations, not the focus. Locked gaps take precedence:

- If the gap in front is locked, the front vertex stays where it is.
- Otherwise, if the gap behind is locked, the back vertex stays where it is.
- If both are locked, the front vertex stays, and the lenses joined to it behind (and the
  image plane, if its gap is locked too) shift by the change in length, so every locked
  gap keeps its length.

If the new lens would then overlap a neighbour, the replacement is refused and nothing
changes. Behind the last lens, the image plane is pushed back if the new lens would reach
it. A locked gap from the source plane keeps its length.

A lens of a Plossl pair is not replaced on its own unless the pair is split first
(``replace_lens`` with ``split_plossl_pair``); ``plossl.replace_plossl_pair`` replaces both.
"""

from __future__ import annotations

import copy
from dataclasses import fields, replace

import numpy as np

from raycrafter.core import assemblies, gaps, optic_builder, placement
from raycrafter.core.state import BeamField, OpticalState, PlacedLens

SAME_FOCAL_LENGTH_TOLERANCE = 0.10  # "same f" lists lenses within ±10% of the focal length
OVERLAP_TOLERANCE_MM = 1e-9  # lenses may touch, allowing for rounding


def _alone(lenses: list[PlacedLens], wavelength_nm: float) -> OpticalState:
    """A bench holding only copies of these lenses, traced at one wavelength."""
    copies = copy.deepcopy(lenses)
    return OpticalState(
        lenses=copies,
        fields=[BeamField(wavelength_nm=wavelength_nm)],
        image_z=copies[-1].back_z + 10.0,
    )


def focal_length_mm(lens: PlacedLens, wavelength_nm: float) -> float:
    """The effective focal length (mm) of one lens on its own, at this wavelength.

    Negative for a diverging lens; infinite for one with no power (a flat plate).
    """
    if lens.paraxial_focal_length_mm is not None:
        return lens.paraxial_focal_length_mm
    return float(optic_builder.build_optic(_alone([lens], wavelength_nm)).paraxial.f2())


def principal_planes_z(
    lenses: list[PlacedLens], wavelength_nm: float
) -> tuple[float, float] | None:
    """z (mm) of the front and rear principal planes of a run of lenses, taken on their own.

    Returns None if the run has no power (it is afocal), as it then has no principal planes.
    """
    if len(lenses) == 1 and lenses[0].is_paraxial:
        return lenses[0].z, lenses[0].z  # a thin lens is its own principal plane
    state = _alone(lenses, wavelength_nm)
    paraxial = optic_builder.build_optic(state).paraxial
    if not np.isfinite(float(paraxial.f2())):
        return None
    # Optiland gives the front principal plane relative to the first surface, and the rear
    # one relative to the image surface.
    front_z = state.lenses[0].z + float(paraxial.P1())
    rear_z = state.image_z + float(paraxial.P2())
    return front_z, rear_z


def anchored_front_z(
    old_lenses: list[PlacedLens], new_lenses: list[PlacedLens], wavelength_nm: float
) -> float:
    """Front-vertex z for the new run so its rear principal plane is where the old one's was.

    If either run is afocal, the new run is centred where the old one was instead.
    """
    old_planes = principal_planes_z(old_lenses, wavelength_nm)
    new_planes = principal_planes_z(new_lenses, wavelength_nm)
    if old_planes is None or new_planes is None:
        old_centre_z = 0.5 * (old_lenses[0].z + old_lenses[-1].back_z)
        new_length = new_lenses[-1].back_z - new_lenses[0].z
        return old_centre_z - 0.5 * new_length
    return new_lenses[0].z + (old_planes[1] - new_planes[1])


def replace_lens(
    state: OpticalState, lens_index: int, new_lens: PlacedLens, split_plossl_pair: bool = False
) -> None:
    """Put ``new_lens`` in place of a lens, as the module docstring describes.

    The new lens takes over the old one's place in an assembly, the lock on the gap behind
    it and, if both are catalogue lenses, whether it is flipped. It is copied; its z does not
    matter.

    Parameters
    ----------
    state : OpticalState
        The bench to change.
    lens_index : int
        The lens to replace.
    new_lens : PlacedLens
        The replacement.
    split_plossl_pair : bool
        If the lens is one of a Plossl pair, the pair must be split first (its gap is then
        unlocked); True does that. To replace both, use ``plossl.replace_plossl_pair``.

    Raises
    ------
    ValueError
        If the lens is one of a Plossl pair and ``split_plossl_pair`` is False, or the new
        lens would overlap a neighbour. Nothing is changed then (the pair is not split).
    """
    in_pair = assemblies.in_plossl_pair(state, lens_index)
    if in_pair and not split_plossl_pair:
        raise ValueError(
            "This lens is one of a Plossl pair: replace both lenses, or split the pair first"
        )
    old_lens = state.lenses[lens_index]
    new_lens = copy.deepcopy(new_lens)
    new_lens.flipped = old_lens.flipped and not (old_lens.is_paraxial or new_lens.is_paraxial)

    # Split and replace on a copy, so that a refusal leaves the pair as it was.
    trial = copy.deepcopy(replace(state, imported_from=None))  # the import record is large
    if in_pair:
        assemblies.split_assembly(trial, lens_index)
    replace_lenses(trial, lens_index, lens_index, [new_lens])
    for state_field in fields(OpticalState):
        if state_field.name != "imported_from":
            setattr(state, state_field.name, getattr(trial, state_field.name))


def replace_lenses(
    state: OpticalState, first_index: int, last_index: int, new_lenses: list[PlacedLens]
) -> None:
    """Put ``new_lenses`` on the bench in place of lenses ``first_index`` to ``last_index``.

    The new run is placed as the module docstring describes. It takes over the old run's
    assembly and the lock on the gap behind it; the gaps inside a new run of two or more
    lenses are locked (it is meant to be one assembly, such as a Plossl pair). Orientation
    is the caller's business.

    Parameters
    ----------
    state : OpticalState
        The bench to change.
    first_index, last_index : int
        The first and last lens to replace (inclusive).
    new_lenses : list of PlacedLens
        The replacement, in axis order, with the spacing it should have; its own z values
        are only relative. The lenses are copied, so the caller's objects are not changed.

    Raises
    ------
    ValueError
        If the new lenses would overlap a neighbour. Nothing is changed then.
    """
    lenses = state.lenses
    old_run = lenses[first_index : last_index + 1]
    run_is_last = last_index == len(lenses) - 1
    locked_before = first_index > 0 and lenses[first_index - 1].locked_to_next
    locked_after = state.image_gap_locked if run_is_last else old_run[-1].locked_to_next

    new_run = copy.deepcopy(new_lenses)
    new_length = new_run[-1].back_z - new_run[0].z
    if locked_before:
        front_z = old_run[0].z
    elif locked_after:
        front_z = old_run[-1].back_z - new_length
    else:
        front_z = anchored_front_z(old_run, new_run, state.selected_field.wavelength_nm)
    shift = front_z - new_run[0].z
    for lens in new_run:
        lens.z += shift
    growth_mm = new_run[-1].back_z - old_run[-1].back_z  # how far the back vertex moves

    # With the gap behind locked, the lenses joined to the run behind it move with its back.
    followers = []
    if locked_after and not run_is_last:
        followers = [index for index in gaps.block_indices(state, last_index) if index > last_index]
    end_index = followers[-1] if followers else last_index  # the last lens that moves
    end_back_z = new_run[-1].back_z if not followers else lenses[end_index].back_z + growth_mm
    end_is_last = end_index == len(lenses) - 1
    image_moves_too = end_is_last and state.image_gap_locked

    # Refuse before changing anything if the new lenses would overlap a neighbour.
    name = new_run[0].part_number
    if first_index > 0:
        lowest_z = lenses[first_index - 1].back_z
    else:
        lowest_z = placement.lowest_first_front_z(state)
    if front_z < lowest_z - OVERLAP_TOLERANCE_MM:
        neighbour = "the lens in front" if first_index > 0 else "the start of the bench"
        raise ValueError(f"{name} does not fit here: it would overlap {neighbour}")
    if not end_is_last and end_back_z > lenses[end_index + 1].z + OVERLAP_TOLERANCE_MM:
        raise ValueError(f"{name} does not fit here: it would overlap the lens behind")

    following_lenses = [lenses[index] for index in followers]
    gap_from_source_mm = placement.source_gap_mm(state)
    for lens in new_run:
        lens.assembly_id = old_run[0].assembly_id
        lens.locked_to_next = True
    new_run[-1].locked_to_next = old_run[-1].locked_to_next
    lenses[first_index : last_index + 1] = new_run

    for lens in following_lenses:
        lens.z += growth_mm
    if image_moves_too:
        state.image_z += growth_mm
    elif end_is_last:
        # The image plane stays put unless the last lens would now reach it.
        state.image_z = max(state.image_z, lenses[-1].back_z + gaps.MIN_IMAGE_GAP_MM)
    if first_index == 0:
        placement.keep_source_gap(state, gap_from_source_mm)
