"""The lens bench model: the one object that front ends (GUI, report, scripts) talk to.

``OpticalModel`` owns the bench state (``raycrafter.core.state``) and the list of lenses
that can be placed. It provides the editing operations: add, insert, delete, move, swap and
replace lenses (``raycrafter.core.replacement``), set and lock the air gaps between them
(and from the source plane, where the beams are defined, to the first lens), group lenses
into assemblies (such as Plossl pairs), flip them and save them as user parts (``raycrafter.core.user_parts``), add or
remove fields, and undo and redo. It also gives access to the analyses
(``raycrafter.core.analysis``), to saving and opening project files
(``raycrafter.core.project_file``) and to Zemax import and export, applied to its current
state. It has no Qt code.

Example
-------
>>> from raycrafter.core.model import OpticalModel
>>> model = OpticalModel()  # starts with one ACT508-200-A on the axis
>>> model.autofocus()  # doctest: +SKIP
>>> model.trace_fast().spot.rms_radius  # doctest: +SKIP
"""

from __future__ import annotations

import copy
from collections.abc import Hashable
from dataclasses import replace
from pathlib import Path

import numpy as np

from raycrafter.core import (
    analysis,
    assemblies,
    gaps,
    optic_builder,
    project_file,
    replacement,
    user_parts,
    zmx_import,
)
from raycrafter.core.history import UndoHistory, same_design
from raycrafter.core.prescription import prescription_from_zmx_text
from raycrafter.core.state import AXIS_START_Z, BeamField, OpticalState, PlacedLens
from raycrafter.core.user_parts import UserPart
from raycrafter.lenscatalog import thorlabs
from raycrafter.lenscatalog.catalog import CatalogEntry

DEFAULT_LENS_PART = "ACT508-200-A"  # the lens placed when the model is created
INITIAL_LENS_Z = 100.0  # mm, front vertex of the first lens
LENS_GAP_MM = 10.0  # air gap left between a new lens and its left neighbour
IMAGE_CLEARANCE_MM = 50.0  # a new lens pushes the image plane at least this far behind it
FIRST_LENS_MIN_Z = AXIS_START_Z + 20.0  # lenses cannot be dragged closer to the axis start
PLOSSL_GAP_MM = 1.0  # closest distance between the two facing surfaces of a Plossl pair
PLOSSL_SEARCH_WIDTHS = 2.0  # a Plossl partner must be within this many lens thicknesses
PARAXIAL_SEMI_APERTURE_MM = 25.4  # paraxial lenses are the size of a 2-inch optic


class OpticalModel:
    """A row of lenses on an optical axis, a beam and an image plane.

    Parameters
    ----------
    catalog : dict, optional
        Part number -> CatalogEntry for the lenses that can be placed. Defaults to the
        packaged Thorlabs catalogue.
    add_default_lens : bool
        If True (the default), start with one DEFAULT_LENS_PART on the axis.

    Attributes
    ----------
    state : OpticalState
        The lenses, fields (beams) and image plane. Front ends may set a field's properties
        and ``image_z`` directly; lenses, and which fields exist, should be changed through
        the methods here. ``undo``, ``redo``, ``new_design``, ``load_project`` and
        ``load_zmx`` replace this object, so do not keep a reference to it.
    catalog : dict
        Part number -> CatalogEntry.
    project_path : Path or None
        The project file last saved or opened, where ``save_project`` saves by default;
        None for a new or imported design.
    parts_folder : Path
        Where user parts are saved and read (``user_parts.default_parts_folder()`` unless
        changed).

    Notes
    -----
    Undo steps are not recorded automatically: a front end calls ``record_undo_step`` when a
    user action is complete.
    """

    def __init__(
        self, catalog: dict[str, CatalogEntry] | None = None, add_default_lens: bool = True
    ) -> None:
        self.state = OpticalState()
        self.catalog = catalog if catalog is not None else thorlabs.load_lenses()
        if add_default_lens and DEFAULT_LENS_PART in self.catalog:
            self.add_lens(DEFAULT_LENS_PART)
        self.saved_state: OpticalState = copy.deepcopy(
            self.state
        )  # the start-up bench counts as saved
        self.project_path: Path | None = None  # the project file last saved or opened
        self.parts_folder: Path = user_parts.default_parts_folder()
        self.history = UndoHistory(self.state)

    # --------------------------------------------------------------- adding and removing

    def make_lens(self, part_number: str, front_z: float = INITIAL_LENS_Z) -> PlacedLens:
        """A catalogue lens at the given front-vertex z, not yet on the bench.

        Front ends use this to preview a lens (e.g. the ghost that follows the pointer)
        before ``insert_lens`` puts it on the bench.
        """
        entry = self.catalog[part_number]
        return PlacedLens(
            part_number=part_number,
            surfaces=prescription_from_zmx_text(entry.zmx_text),
            semi_aperture=entry.semi_aperture,
            z=front_z,
            is_negative=entry.is_negative,
            vendor=entry.vendor,
            mount_type=entry.mount_type,
        )

    @staticmethod
    def make_paraxial_lens(focal_length_mm: float, z: float = INITIAL_LENS_Z) -> PlacedLens:
        """An ideal thin lens of the given focal length at ``z``, not yet on the bench.

        A paraxial lens focuses perfectly (no aberrations) and has zero thickness. Its clear
        aperture is PARAXIAL_SEMI_APERTURE_MM. Put it on the bench with ``insert_lens``.

        Raises
        ------
        ValueError
            If the focal length is zero.
        """
        if focal_length_mm == 0.0:
            raise ValueError("A paraxial lens needs a non-zero focal length")
        return PlacedLens(
            part_number=f"Paraxial f={focal_length_mm:+g} mm",
            surfaces=[],
            semi_aperture=PARAXIAL_SEMI_APERTURE_MM,
            z=z,
            is_negative=focal_length_mm < 0,
            paraxial_focal_length_mm=focal_length_mm,
        )

    def add_lens(self, part_number: str) -> PlacedLens:
        """Place a catalogue lens just to the right of the rightmost lens.

        The first lens goes at INITIAL_LENS_Z. The image plane is pushed back if the new
        lens would otherwise end up beyond it.

        Returns
        -------
        PlacedLens
            The new lens.

        Raises
        ------
        ValueError
            If the gap to the image plane is locked, as the new lens would split it.
        """
        self._refuse_if_image_gap_locked("add a lens behind the last one")
        lenses = self.state.lenses
        if lenses:
            front_z = lenses[-1].back_z + LENS_GAP_MM
        else:
            front_z = max(INITIAL_LENS_Z, self.lowest_first_front_z())
        new_lens = self.make_lens(part_number, front_z)
        lenses.append(new_lens)
        if len(lenses) == 1:
            self._keep_source_gap(0.0)  # a locked source plane starts at the first lens
        self.state.image_z = max(self.state.image_z, new_lens.back_z + IMAGE_CLEARANCE_MM)
        return new_lens

    def free_front_z(
        self, thickness_mm: float, requested_front_z: float, max_back_z: float = np.inf
    ) -> float | None:
        """The front-vertex z nearest the request at which a new lens fits without overlap.

        The free stretches of axis are the gaps between the lenses on the bench, plus the
        stretch before the first lens (from ``lowest_first_front_z``) and after the last one
        (up to ``max_back_z``). A locked gap (such as the one inside a Plossl pair) is not free, as
        a new lens would split it; nor is the stretch behind the last lens if the gap to the
        image plane is locked. A new lens may touch its neighbours, as a dragged lens may.

        Parameters
        ----------
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
        lenses = self.state.lenses

        # Occupied stretches of axis, one per block of lenses joined by locked gaps.
        occupied = [
            (lenses[members[0]].z, lenses[members[-1]].back_z)
            for members in gaps.blocks(self.state)
        ]

        gap_starts = [self.lowest_first_front_z()] + [back for _front, back in occupied]
        gap_ends = [block_front for block_front, _back in occupied] + [max_back_z]
        if lenses and self.state.image_gap_locked:
            gap_starts, gap_ends = gap_starts[:-1], gap_ends[:-1]

        # In each gap the lens is wide enough for, the closest position to the request;
        # then the closest of those.
        candidates = []
        for gap_start, gap_end in zip(gap_starts, gap_ends):
            latest_front_z = gap_end - thickness_mm
            if latest_front_z >= gap_start:
                candidates.append(float(np.clip(requested_front_z, gap_start, latest_front_z)))
        if not candidates:
            return None
        return min(candidates, key=lambda front_z: abs(front_z - requested_front_z))

    def insert_lens(self, new_lens: PlacedLens, max_image_z: float = np.inf) -> int:
        """Put a lens (from ``make_lens``) on the bench at its own z, in axis order.

        The caller is responsible for choosing a free position (see ``free_front_z``). If
        the lens lands behind the image plane, the image plane is pushed back to
        IMAGE_CLEARANCE_MM behind it, but not beyond ``max_image_z`` (unless even that is
        too close to the lens, in which case it goes just behind the lens).

        Returns
        -------
        int
            The index of the new lens.

        Raises
        ------
        ValueError
            If the lens would land in a locked gap, splitting it; nothing is changed then.
        """
        return self.insert_lenses([new_lens], max_image_z)[0]

    def insert_lenses(
        self,
        new_lenses: list[PlacedLens],
        max_image_z: float = np.inf,
        assembly_name: str | None = None,
    ) -> list[int]:
        """Put a run of lenses (e.g. a user part) on the bench at their own z, as one block.

        As ``insert_lens``, for lenses that follow each other without overlapping. Two or
        more lenses become an assembly called ``assembly_name``.

        Returns
        -------
        list of int
            The indices of the new lenses.

        Raises
        ------
        ValueError
            If the lenses would land in a locked gap or overlap a lens on the bench; nothing
            is changed then.
        """
        lenses = self.state.lenses
        new_index = sum(1 for lens in lenses if lens.z < new_lenses[0].z)
        if new_index == 0 and new_lenses[0].z < self.lowest_first_front_z() - 1e-9:
            raise ValueError("A lens cannot go in front of the source plane or the axis start")
        if new_index == len(lenses):
            if lenses:
                self._refuse_if_image_gap_locked("add a lens behind the last one")
        elif new_index > 0 and lenses[new_index - 1].locked_to_next:
            raise ValueError("That gap is locked: unlock it to put a lens in it")
        # Allow for rounding, as a lens may be placed exactly touching its neighbours.
        previous_back_z = lenses[new_index - 1].back_z if new_index > 0 else -np.inf
        next_front_z = lenses[new_index].z if new_index < len(lenses) else np.inf
        if new_lenses[0].z < previous_back_z - 1e-9 or new_lenses[-1].back_z > next_front_z + 1e-9:
            raise ValueError("The new lenses would overlap a lens on the bench")

        source_gap_mm = self._source_gap_mm()
        lenses[new_index:new_index] = new_lenses
        if new_index == 0:
            self._keep_source_gap(source_gap_mm)
        new_indices = list(range(new_index, new_index + len(new_lenses)))
        if len(new_lenses) > 1:
            assemblies.make_assembly(self.state, new_indices[0], new_indices[-1], assembly_name)

        minimum_image_z = self.state.last_back_z() + gaps.MIN_IMAGE_GAP_MM
        if self.state.image_z < minimum_image_z:
            pushed_image_z = min(new_lenses[-1].back_z + IMAGE_CLEARANCE_MM, max_image_z)
            self.state.image_z = max(pushed_image_z, minimum_image_z)
        return new_indices

    def delete_lens(self, lens_index: int) -> None:
        """Remove one lens, and the locks on the gaps either side of it.

        Its assembly, if any, is dissolved; the other gaps inside it stay locked. The gap
        from the source plane is the exception: if it is locked, the source plane keeps the
        same distance in front of whichever lens is now first.
        """
        assemblies.split_assembly(self.state, lens_index, unlock=False)
        source_gap_mm = self._source_gap_mm()
        lenses = self.state.lenses
        if lens_index > 0:
            lenses[lens_index - 1].locked_to_next = False
        if lens_index == len(lenses) - 1:
            self.state.image_gap_locked = False
        del lenses[lens_index]
        if lens_index == 0:
            self._keep_source_gap(source_gap_mm)

    def flip_lens(self, lens_index: int) -> None:
        """Turn one lens around.

        Lenses in an assembly cannot be flipped on their own (see ``flip_assembly``), and a
        paraxial lens is the same both ways round, so neither changes.
        """
        lens = self.state.lenses[lens_index]
        if lens.assembly_id is None and not lens.is_paraxial:
            lens.flipped = not lens.flipped

    # ---------------------------------------------------------------- replacing lenses

    def lens_focal_length_mm(self, lens_index: int) -> float:
        """The focal length (mm) of one lens on its own, at the selected field's wavelength."""
        wavelength_nm = self.state.selected_field.wavelength_nm
        return replacement.focal_length_mm(self.state.lenses[lens_index], wavelength_nm)

    def in_plossl_pair(self, lens_index: int) -> bool:
        """True if the lens is one of a Plossl pair."""
        return self.assembly_name(lens_index) == assemblies.PLOSSL_NAME

    def replace_lens(
        self, lens_index: int, new_lens: PlacedLens, split_plossl_pair: bool = False
    ) -> None:
        """Put ``new_lens`` (from ``make_lens`` or ``make_paraxial_lens``) in place of a lens.

        It goes where the old lens's rear principal plane was, unless locked gaps say
        otherwise (see ``raycrafter.core.replacement``). It takes over the old lens's place
        in an assembly, the lock on the gap behind it and, if both are catalogue lenses,
        whether it is flipped.

        Parameters
        ----------
        lens_index : int
            The lens to replace.
        new_lens : PlacedLens
            The replacement; its z does not matter. It is copied.
        split_plossl_pair : bool
            If the lens is one of a Plossl pair, the pair must be split first (its gap is
            then unlocked); True does that. To replace both, use ``replace_plossl_pair``.

        Raises
        ------
        ValueError
            If the lens is one of a Plossl pair and ``split_plossl_pair`` is False, or the
            new lens would overlap a neighbour. Nothing is changed then.
        """
        if self.in_plossl_pair(lens_index) and not split_plossl_pair:
            raise ValueError(
                "This lens is one of a Plossl pair: replace both lenses, or split the pair first"
            )
        old_lens = self.state.lenses[lens_index]
        new_lens = copy.deepcopy(new_lens)
        new_lens.flipped = old_lens.flipped and not (old_lens.is_paraxial or new_lens.is_paraxial)

        state_before = copy.deepcopy(self.state)
        if self.in_plossl_pair(lens_index):
            assemblies.split_assembly(self.state, lens_index)
        try:
            self._replace_run(lens_index, lens_index, [new_lens])
        except ValueError:
            self.state = state_before  # put the pair back: a refusal changes nothing
            raise

    def replace_plossl_pair(self, lens_index: int, part_number: str) -> None:
        """Replace both lenses of this lens's Plossl pair with two of ``part_number``.

        The new pair is set up as ``make_plossl`` does (facing each other, PLOSSL_GAP_MM
        apart) and placed as ``replace_lens`` places one lens, with the pair's own rear
        principal plane where the old pair's was. It stays a Plossl pair.

        Raises
        ------
        ValueError
            If the lens is in no Plossl pair, the new part is a diverging lens, or the new
            pair would overlap a neighbour. Nothing is changed then.
        """
        if not self.in_plossl_pair(lens_index):
            raise ValueError("This lens is not one of a Plossl pair")
        left, right = self.make_lens(part_number, 0.0), self.make_lens(part_number, 0.0)
        if left.is_negative:
            raise ValueError(f"A Plossl pair needs a positive lens, and {part_number} is not")
        self._turn_to_face_as_plossl(left, right)
        right.z = self._plossl_front_z(left, right)
        members = self.assembly_indices(lens_index)
        self._replace_run(members[0], members[-1], [left, right])

    def _replace_run(self, first_index: int, last_index: int, new_lenses: list[PlacedLens]) -> None:
        """``replacement.replace_lenses``, keeping a locked gap from the source plane."""
        source_gap_mm = self._source_gap_mm()
        replacement.replace_lenses(
            self.state, first_index, last_index, new_lenses, self.lowest_first_front_z()
        )
        if first_index == 0:
            self._keep_source_gap(source_gap_mm)

    # ---------------------------------------------------------------- air gaps

    def air_gaps(self) -> list[gaps.AirGap]:
        """The air gap behind every lens (see ``gaps.air_gaps``)."""
        return gaps.air_gaps(self.state)

    def set_gap(self, lens_index: int, gap_mm: float) -> None:
        """Set the gap behind a lens, moving everything downstream (see ``gaps.set_gap``)."""
        gaps.set_gap(self.state, lens_index, gap_mm)

    def set_gap_locked(self, lens_index: int, locked: bool) -> None:
        """Lock or unlock the gap behind a lens (see ``gaps.set_gap_locked``)."""
        gaps.set_gap_locked(self.state, lens_index, locked)

    # ---------------------------------------------------------------- the source plane

    def source_gap(self) -> gaps.AirGap | None:
        """The gap from the source plane to the first lens (see ``gaps.source_gap``)."""
        return gaps.source_gap(self.state)

    def set_source_gap(self, gap_mm: float) -> None:
        """Set the gap from the source plane to the first lens (see ``gaps.set_source_gap``).

        Raises
        ------
        ValueError
            As ``gaps.set_source_gap``, and also if the first lens would end up closer than
            FIRST_LENS_MIN_Z to the start of the axis (where it could not be dragged).
        """
        if self.state.lenses and self.state.source_z + gap_mm < FIRST_LENS_MIN_Z - 1e-9:
            smallest_mm = FIRST_LENS_MIN_Z - self.state.source_z
            raise ValueError(
                f"This gap must be at least {smallest_mm:g} mm, to keep the first lens "
                f"{FIRST_LENS_MIN_Z - AXIS_START_Z:g} mm from the start of the axis; move the "
                "source plane forward first"
            )
        gaps.set_source_gap(self.state, gap_mm)

    def set_source_gap_locked(self, locked: bool) -> None:
        """Lock or unlock the gap from the source plane to the first lens.

        While it is locked the source plane moves with the first lens, keeping its distance.
        """
        self.state.source_gap_locked = locked

    def set_source_z(self, source_z: float) -> None:
        """Move the source plane, where every field is defined, to ``source_z`` (mm).

        Raises
        ------
        ValueError
            If the gap from the source plane is locked, or ``source_z`` is in front of the
            start of the axis or behind the first lens.
        """
        if self.state.source_gap_locked:
            raise ValueError("The gap from the source plane is locked: unlock it to move it")
        highest_z = self.state.lenses[0].z if self.state.lenses else np.inf
        if not AXIS_START_Z <= source_z <= highest_z + 1e-9:
            raise ValueError(
                f"The source plane must be between the start of the axis ({AXIS_START_Z:g} mm) "
                f"and the first lens ({highest_z:g} mm)"
            )
        self.state.source_z = min(source_z, highest_z)

    def lowest_first_front_z(self) -> float:
        """The smallest z (mm) the front vertex of the first lens may have.

        It must stay FIRST_LENS_MIN_Z from the start of the axis. With the source gap
        unlocked it cannot pass the source plane; with it locked, the source plane, which
        moves with it, cannot pass the start of the axis.
        """
        if self.state.source_gap_locked:
            return max(FIRST_LENS_MIN_Z, AXIS_START_Z + self._source_gap_mm())
        return max(FIRST_LENS_MIN_Z, self.state.source_z)

    def _source_gap_mm(self) -> float:
        """The gap from the source plane to the first lens (mm); 0 with no lenses."""
        gap = gaps.source_gap(self.state)
        return 0.0 if gap is None else gap.width_mm

    def _keep_source_gap(self, source_gap_mm: float) -> None:
        """After the first lens changed: with the source gap locked, put the source plane
        ``source_gap_mm`` in front of the new first lens (if there is one)."""
        if self.state.source_gap_locked and self.state.lenses:
            self.state.source_z = self.state.lenses[0].z - source_gap_mm

    def _refuse_if_image_gap_locked(self, action: str) -> None:
        """Raise ValueError, saying the user cannot do ``action``, if the image gap is locked."""
        if self.state.lenses and self.state.image_gap_locked:
            raise ValueError(f"The gap to the image plane is locked: unlock it to {action}")

    # ---------------------------------------------------------------- moving

    def block_indices(self, lens_index: int) -> list[int]:
        """Indices of the lenses that move with this one: those joined to it by locked gaps."""
        return gaps.block_indices(self.state, lens_index)

    def _block_centre_z(self, members: list[int]) -> float:
        """z of the middle of a block of adjacent lenses, from its first front to last back."""
        lenses = self.state.lenses
        return 0.5 * (lenses[members[0]].z + lenses[members[-1]].back_z)

    def move_lens(self, lens_index: int, requested_front_z: float) -> list[int] | None:
        """Move a lens (and its block) so this lens's front vertex is near the request.

        The block is the lenses joined to this one by locked gaps (``block_indices``), which
        move as one. The first lens cannot pass ``lowest_first_front_z``, and with the source
        gap locked the source plane moves with it. If the request carries the block's centre past the centre of the
        neighbouring block, the two swap places (see ``swap_neighbouring_blocks``).
        Otherwise the block stays between its neighbours (nothing overlaps), the first lens
        stays to the right of the start of the axis, and the last lens stays before the
        image plane; if the allowed range is empty the lens does not move. If the gap to the
        image plane is locked, the image plane moves with the last block instead.

        Returns
        -------
        list of int or None
            After a swap, the new order as a list of old indices, so the caller can reorder
            anything it keeps per lens. None when the order is unchanged.
        """
        lenses = self.state.lenses
        members = self.block_indices(lens_index)
        first, last = members[0], members[-1]
        is_last_block = last == len(lenses) - 1

        shift_wanted = requested_front_z - lenses[lens_index].z
        wanted_centre_z = self._block_centre_z(members) + shift_wanted

        # Dragged past the centre of the next block to the right, or the previous one?
        if not is_last_block:
            right_members = self.block_indices(last + 1)
            if wanted_centre_z > self._block_centre_z(right_members):
                return self.swap_neighbouring_blocks(members, right_members)
        if first > 0:
            left_members = self.block_indices(first - 1)
            if wanted_centre_z < self._block_centre_z(left_members):
                return self.swap_neighbouring_blocks(left_members, members)

        block_width = lenses[last].back_z - lenses[first].z
        lowest_front_z = lenses[first - 1].back_z if first > 0 else self.lowest_first_front_z()
        image_moves_too = is_last_block and self.state.image_gap_locked
        if image_moves_too:
            next_front_z = np.inf
        elif not is_last_block:
            next_front_z = lenses[last + 1].z
        else:
            next_front_z = self.state.image_z - gaps.MIN_IMAGE_GAP_MM
        highest_front_z = next_front_z - block_width

        shift_min = lowest_front_z - lenses[first].z
        shift_max = highest_front_z - lenses[first].z
        if shift_max < shift_min:
            return None
        shift = float(np.clip(shift_wanted, shift_min, shift_max))
        for index in members:
            lenses[index].z += shift
        if image_moves_too:
            self.state.image_z += shift
        if first == 0 and self.state.source_gap_locked:
            self.state.source_z += shift  # the source plane moves with the first lens
        return None

    def swap_neighbouring_blocks(
        self, left_members: list[int], right_members: list[int]
    ) -> list[int]:
        """Exchange two adjacent blocks of lenses, so the left one ends up on the right.

        Each block keeps its internal spacing and locks. The air gap between them is kept, and the two
        together occupy exactly the same stretch of axis as before, so nothing downstream
        (later lenses, the image plane) has to move.

        Returns
        -------
        list of int
            The new order as a list of old lens indices.
        """
        lenses = self.state.lenses
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

    # ---------------------------------------------------------------- assemblies

    def assembly_indices(self, lens_index: int) -> list[int]:
        """Indices of the lenses in this lens's assembly (just itself if it is in none)."""
        return assemblies.assembly_indices(self.state, lens_index)

    def assembly_name(self, lens_index: int) -> str | None:
        """The name of this lens's assembly, or None if it is in none."""
        return assemblies.assembly_name(self.state, lens_index)

    def make_assembly(self, first_index: int, last_index: int, name: str | None = None) -> int:
        """Group a run of neighbouring lenses into an assembly (see ``assemblies``)."""
        return assemblies.make_assembly(self.state, first_index, last_index, name)

    def split_assembly(self, lens_index: int) -> None:
        """Dissolve this lens's assembly and unlock the gaps inside it."""
        assemblies.split_assembly(self.state, lens_index)

    def flip_assembly(self, lens_index: int) -> list[int]:
        """Turn this lens's assembly round as a whole; returns the new order of old indices."""
        return assemblies.flip_assembly(self.state, lens_index)

    def save_assembly_as_part(self, lens_index: int, name: str, overwrite: bool = False) -> Path:
        """Save this lens's assembly as a user part in ``parts_folder``; return the file.

        Raises
        ------
        ValueError
            If the lens is in no assembly, the name is blank, or (unless ``overwrite``)
            there is already a part of that name.
        """
        if self.state.lenses[lens_index].assembly_id is None:
            raise ValueError("Only an assembly can be saved as a user part")
        members = self.assembly_indices(lens_index)
        part = user_parts.part_from_lenses(name, [self.state.lenses[index] for index in members])
        return user_parts.save_part(part, self.parts_folder, overwrite)

    def user_parts(self) -> list[UserPart]:
        """The user parts in ``parts_folder``, sorted by name (unreadable files are skipped)."""
        return user_parts.load_parts(self.parts_folder)[0]

    def user_part(self, name: str) -> UserPart:
        """The user part with this name.

        Raises
        ------
        ValueError
            If there is none.
        """
        for part in self.user_parts():
            if part.name == name:
                return part
        raise ValueError(f"There is no user part called {name!r}")

    # ---------------------------------------------------------------- Plossl pairs

    def plossl_partner(self, lens_index: int) -> int | None:
        """Index of a lens this one could form a Plossl pair with, or None.

        The partner must be an immediate neighbour, be the same part, and be in no
        assembly; the gap between the two must be at most PLOSSL_SEARCH_WIDTHS lens
        thicknesses. Only positive catalogue lenses can form a Plossl pair (a Plossl is made
        of doublets, so paraxial lenses are excluded).
        """
        lenses = self.state.lenses
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

    def make_plossl(self, lens_index: int) -> None:
        """Snap the lens and its partner into a Plossl pair and group them.

        A Plossl pair is two identical doublets facing each other symmetrically, with their
        positive (crown) elements on the inside. The left lens is flipped if needed so its
        positive element is at its back, and the right lens so its positive element is at
        its front. The right lens is then moved so the closest points of the two facing
        surfaces are PLOSSL_GAP_MM apart, and everything downstream (later lenses and the
        image plane) shifts with it, even if the gap between the two was locked. The two
        then form an assembly named ``assemblies.PLOSSL_NAME``. Does nothing if the lens has
        no partner.
        """
        partner_index = self.plossl_partner(lens_index)
        if partner_index is None:
            return
        left_index, right_index = sorted((lens_index, partner_index))
        lenses = self.state.lenses
        left, right = lenses[left_index], lenses[right_index]
        self._turn_to_face_as_plossl(left, right)

        original_right_z = right.z
        right.z = self._plossl_front_z(left, right)
        shift = right.z - original_right_z
        for downstream in lenses[right_index + 1 :]:
            downstream.z += shift
        minimum_image_z = self.state.last_back_z() + gaps.MIN_IMAGE_GAP_MM
        self.state.image_z = max(self.state.image_z + shift, minimum_image_z)

        assemblies.make_assembly(self.state, left_index, right_index, assemblies.PLOSSL_NAME)

    @staticmethod
    def _turn_to_face_as_plossl(left: PlacedLens, right: PlacedLens) -> None:
        """Flip two identical doublets as needed so their positive elements face each other."""
        left.flipped = left.positive_element_is_first()  # moves the positive element to the back
        right.flipped = not right.positive_element_is_first()  # keeps or puts it at the front

    @staticmethod
    def _plossl_front_z(left: PlacedLens, right: PlacedLens) -> float:
        """Front-vertex z for ``right`` so the facing surfaces are PLOSSL_GAP_MM apart at closest.

        The gap is measured along z across the smaller clear aperture, and its smallest value
        (at the vertices for two convex surfaces, nearer the rim if either is concave) is set
        to the required gap.
        """
        return left.back_z + PLOSSL_GAP_MM - gaps.closest_sag_difference_mm(left, right)

    # ---------------------------------------------------------------- fields

    def add_field(self) -> BeamField:
        """Add a field: a copy of the selected one, which then becomes the selected field.

        Starting from a copy means a new field is already sensible for the bench, and the
        user changes only what should differ. It gets the colour of its wavelength, even if
        the copied field had a colour of its own, so the two can be told apart once edited.

        Returns
        -------
        BeamField
            The new field (the last in ``state.fields``).
        """
        new_field = replace(self.state.selected_field, colour=None)
        self.state.fields.append(new_field)
        self.state.selected_field_index = len(self.state.fields) - 1
        return new_field

    def remove_fields(self, field_indices: list[int]) -> None:
        """Remove the given fields, always keeping at least one.

        If every field is listed, the first field is kept. The selection moves to the field
        just before the first one removed (or to the first field).

        Raises
        ------
        IndexError
            If any index is not a field; nothing is removed then.
        """
        fields = self.state.fields
        for field_index in field_indices:
            if not 0 <= field_index < len(fields):
                raise IndexError(f"There is no field {field_index}")
        to_remove = sorted(set(field_indices))
        if len(to_remove) >= len(fields):
            to_remove = to_remove[1:]  # keep the first field
        if not to_remove:
            return
        for index in reversed(to_remove):
            del fields[index]
        self.state.selected_field_index = min(max(to_remove[0] - 1, 0), len(fields) - 1)

    def select_field(self, field_index: int) -> None:
        """Make this field the one that autofocus and the single-field analyses use."""
        if not 0 <= field_index < len(self.state.fields):
            raise IndexError(f"There is no field {field_index}")
        self.state.selected_field_index = field_index

    # ---------------------------------------------------------------- optics

    def autofocus(self) -> float:
        """Move the image plane to the smallest RMS spot of the selected field, and return its z.

        Raises
        ------
        ValueError
            If the gap to the image plane is locked.
        """
        self._refuse_if_image_gap_locked("focus")
        self.state.image_z = analysis.find_best_focus_z(self.state)
        return self.state.image_z

    def trace_fast(self) -> analysis.FastTrace:
        """Every field's spot and rays, and the first-order summary (``analysis.trace_fast``)."""
        return analysis.trace_fast(self.state)

    def compute_f_number(self) -> analysis.FNumberResult | None:
        """f-number from the transmitted beam (see ``analysis.compute_f_number``)."""
        return analysis.compute_f_number(self.state)

    def compute_through_focus(self, fast_trace: analysis.FastTrace) -> analysis.ThroughFocusResult:
        """RMS radius against image plane position (see ``analysis.compute_through_focus``)."""
        return analysis.compute_through_focus(self.state, fast_trace)

    def compute_ray_fan(self) -> analysis.RayFanResult:
        """Transverse ray aberration (see ``analysis.compute_ray_fan``)."""
        return analysis.compute_ray_fan(self.state)

    def compute_mtf(self) -> analysis.MtfResult:
        """FFT MTF of the selected field (see ``analysis.compute_mtf``)."""
        return analysis.compute_mtf(self.state)

    def compute_field_curvature(self) -> analysis.FieldCurvatureResult:
        """Best focus against field angle (see ``analysis.compute_field_curvature``)."""
        return analysis.compute_field_curvature(self.state)

    # ---------------------------------------------------------------- undo and redo

    def record_undo_step(self, merge_key: Hashable | None = None) -> None:
        """Make the changes since the last step undoable as one (see ``UndoHistory.record``)."""
        self.history.record(self.state, merge_key)

    def amend_undo_step(self) -> None:
        """Take the bench as it is into the current step, without making a new one."""
        self.history.amend(self.state)

    def clear_undo_history(self) -> None:
        """Forget every undo step: the bench as it is now becomes the starting point."""
        self.history.reset(self.state)

    def undo(self) -> bool:
        """Go back one step. Returns False (changing nothing) if there is none."""
        return self._show_history_state(self.history.undo())

    def redo(self) -> bool:
        """Redo the step last undone. Returns False (changing nothing) if there is none."""
        return self._show_history_state(self.history.redo())

    def _show_history_state(self, state: OpticalState | None) -> bool:
        if state is None:
            return False
        self.state = state
        return True

    # ---------------------------------------------------------------- files

    def has_unsaved_changes(self) -> bool:
        """True if the design differs from the one last saved, opened, imported or started.

        Which field is selected is saved in a project file, but choosing another one does
        not count as a change. Nor does an image plane moved by less than
        ``history.IMAGE_Z_TOLERANCE_MM``, as "Lock focus" refocusing an unchanged bench does.
        """
        return not same_design(self.state, self.saved_state)

    def mark_saved(self) -> None:
        """Remember the current design as the saved one.

        Called by ``save_project``, ``load_project``, ``load_zmx`` and ``new_design``. A
        front end calls it itself after changes that count as part of starting up (e.g. the
        first autofocus).
        """
        self.saved_state = copy.deepcopy(self.state)

    def new_design(self) -> None:
        """Start again: no lenses, one default field, no project file.

        Counts as saved; clears undo.
        """
        self.state = OpticalState()
        self.project_path = None
        self.mark_saved()
        self.clear_undo_history()

    def save_project(
        self, file_path: str | Path | None = None, forget_import: bool = False
    ) -> Path:
        """Save the whole design as a RayCrafter project file, and count it as saved.

        Parameters
        ----------
        file_path : str or Path, optional
            Where to save. Defaults to ``project_path``, the file last saved or opened.
            ``project_file.FILE_EXTENSION`` is added if the name has no extension.
        forget_import : bool
            If True, drop the record of the file the design was imported from
            (``OpticalState.imported_from``), so that from now on it is a design of its own
            rather than an import. The record is dropped only if the save succeeds.

        Returns
        -------
        Path
            The file written, which becomes ``project_path``.

        Raises
        ------
        ValueError
            If no path is given and the design has no project file yet.
        """
        if file_path is None:
            if self.project_path is None:
                raise ValueError("The design has no project file yet, so give a file path.")
            file_path = self.project_path
        target = Path(file_path)
        if not target.suffix:
            target = target.with_suffix(project_file.FILE_EXTENSION)
        state_to_save = replace(self.state, imported_from=None) if forget_import else self.state
        project_file.save_project(state_to_save, target)
        self.state.imported_from = state_to_save.imported_from
        if forget_import:
            self.history.forget_imported_source()  # or undo would bring the record back
        self.project_path = target
        self.mark_saved()
        return target

    def load_project(self, file_path: str | Path) -> None:
        """Replace the design with the one in a project file, clearing the undo history.

        Raises
        ------
        project_file.ProjectFileError
            If the file cannot be read or is not a valid project. The design is then left
            as it was.
        """
        self.state = project_file.load_project(file_path)
        self.project_path = Path(file_path)
        self.mark_saved()
        self.clear_undo_history()

    def save_zmx(self, file_path: str) -> None:
        """Export the current system to a Zemax .zmx file.

        This does not count as saving the design: a ZMX file keeps only the selected field
        and no assemblies or part metadata, so the design would be lost on quitting.
        """
        optic_builder.save_zmx(self.state, file_path)

    def load_zmx(self, file_path: str) -> list[str]:
        """Replace the bench with the system in a Zemax .zmx file, clearing the undo history.

        The imported design counts as saved, and has no project file until it is saved.

        Returns
        -------
        list of str
            Notes on anything that was simplified (see ``zmx_import.load_zmx``).

        Raises
        ------
        zmx_import.ZmxImportError
            If the file cannot be read or describes something RayCrafter cannot show. The
            bench is then left as it was.
        """
        imported = zmx_import.load_zmx(file_path, self.catalog)
        self.state = imported.state
        self.project_path = None
        self.mark_saved()
        self.clear_undo_history()
        return imported.notes
