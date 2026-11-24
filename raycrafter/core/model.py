"""The lens bench model: the one object that front ends (GUI, report, scripts) talk to.

``OpticalModel`` owns the bench state (``raycrafter.core.state``) and the list of lenses
that can be placed. It provides the editing operations: add, insert, delete, move and swap
lenses, make or split Plossl pairs, and add or remove fields. It also gives access to the
analyses (``raycrafter.core.analysis``) and to Zemax import and export, applied to its
current state. It has no Qt code.

Example
-------
>>> from raycrafter.core.model import OpticalModel
>>> model = OpticalModel()  # starts with one ACT508-200-A on the axis
>>> model.autofocus()  # doctest: +SKIP
>>> model.trace_fast().spot.rms_radius  # doctest: +SKIP
"""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from raycrafter.core import analysis, optic_builder, zmx_import
from raycrafter.core.prescription import prescription_from_zmx_text, surface_sag
from raycrafter.core.state import AXIS_START_Z, BeamField, OpticalState, PlacedLens
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
        the methods here.
    catalog : dict
        Part number -> CatalogEntry.
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
        """
        lenses = self.state.lenses
        front_z = lenses[-1].back_z + LENS_GAP_MM if lenses else INITIAL_LENS_Z
        new_lens = self.make_lens(part_number, front_z)
        lenses.append(new_lens)
        self.state.image_z = max(self.state.image_z, new_lens.back_z + IMAGE_CLEARANCE_MM)
        return new_lens

    def free_front_z(
        self, thickness_mm: float, requested_front_z: float, max_back_z: float = np.inf
    ) -> float | None:
        """The front-vertex z nearest the request at which a new lens fits without overlap.

        The free stretches of axis are the gaps between the lenses on the bench, plus the
        stretch before the first lens (from FIRST_LENS_MIN_Z) and after the last one (up to
        ``max_back_z``). The gap inside a Plossl pair is not free: a pair is never split.
        A new lens may touch its neighbours, as a dragged lens may.

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

        # Occupied stretches of axis, one per block (a single lens or a Plossl pair).
        occupied = []
        lens_index = 0
        while lens_index < len(lenses):
            members = self.group_indices(lens_index)
            occupied.append((lenses[members[0]].z, lenses[members[-1]].back_z))
            lens_index = members[-1] + 1

        gap_starts = [FIRST_LENS_MIN_Z] + [block_back for _front, block_back in occupied]
        gap_ends = [block_front for block_front, _back in occupied] + [max_back_z]

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
        """
        lenses = self.state.lenses
        new_index = sum(1 for lens in lenses if lens.z < new_lens.z)
        lenses.insert(new_index, new_lens)

        minimum_image_z = self.state.last_back_z() + 1.0
        if self.state.image_z < minimum_image_z:
            pushed_image_z = min(new_lens.back_z + IMAGE_CLEARANCE_MM, max_image_z)
            self.state.image_z = max(pushed_image_z, minimum_image_z)
        return new_index

    def delete_lens(self, lens_index: int) -> None:
        """Remove one lens. Its Plossl partner, if any, becomes a normal lens."""
        self.split_plossl(lens_index)
        del self.state.lenses[lens_index]

    def flip_lens(self, lens_index: int) -> None:
        """Turn one lens around.

        Lenses in a Plossl pair cannot be flipped on their own, and a paraxial lens is the
        same both ways round, so neither changes.
        """
        lens = self.state.lenses[lens_index]
        if lens.group_id is None and not lens.is_paraxial:
            lens.flipped = not lens.flipped

    # ---------------------------------------------------------------- moving

    def group_indices(self, lens_index: int) -> list[int]:
        """Indices of the lenses that move together with this one (just itself if ungrouped)."""
        lenses = self.state.lenses
        group_id = lenses[lens_index].group_id
        if group_id is None:
            return [lens_index]
        return [index for index, lens in enumerate(lenses) if lens.group_id == group_id]

    def _block_centre_z(self, members: list[int]) -> float:
        """z of the middle of a block of adjacent lenses, from its first front to last back."""
        lenses = self.state.lenses
        return 0.5 * (lenses[members[0]].z + lenses[members[-1]].back_z)

    def move_lens(self, lens_index: int, requested_front_z: float) -> list[int] | None:
        """Move a lens (and its Plossl partner) so this lens's front vertex is near the request.

        If the request carries the group's centre past the centre of the neighbouring lens
        (or Plossl pair), the two swap places (see ``swap_neighbouring_blocks``). Otherwise
        the group stays between its neighbours (nothing overlaps), the first lens stays to
        the right of the start of the axis, and the last lens stays before the image plane;
        if the allowed range is empty the lens does not move.

        Returns
        -------
        list of int or None
            After a swap, the new order as a list of old indices, so the caller can reorder
            anything it keeps per lens. None when the order is unchanged.
        """
        lenses = self.state.lenses
        members = self.group_indices(lens_index)
        first, last = members[0], members[-1]

        shift_wanted = requested_front_z - lenses[lens_index].z
        wanted_centre_z = self._block_centre_z(members) + shift_wanted

        # Dragged past the centre of the next block to the right, or the previous one?
        if last < len(lenses) - 1:
            right_members = self.group_indices(last + 1)
            if wanted_centre_z > self._block_centre_z(right_members):
                return self.swap_neighbouring_blocks(members, right_members)
        if first > 0:
            left_members = self.group_indices(first - 1)
            if wanted_centre_z < self._block_centre_z(left_members):
                return self.swap_neighbouring_blocks(left_members, members)

        group_width = lenses[last].back_z - lenses[first].z
        lowest_front_z = lenses[first - 1].back_z if first > 0 else FIRST_LENS_MIN_Z
        if last < len(lenses) - 1:
            next_front_z = lenses[last + 1].z
        else:
            next_front_z = self.state.image_z - 1.0
        highest_front_z = next_front_z - group_width

        shift_min = lowest_front_z - lenses[first].z
        shift_max = highest_front_z - lenses[first].z
        if shift_max < shift_min:
            return None
        shift = float(np.clip(shift_wanted, shift_min, shift_max))
        for index in members:
            lenses[index].z += shift
        return None

    def swap_neighbouring_blocks(
        self, left_members: list[int], right_members: list[int]
    ) -> list[int]:
        """Exchange two adjacent lens groups, so the left one ends up on the right.

        Each block keeps its internal spacing. The air gap between them is kept, and the two
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

    # ---------------------------------------------------------------- Plossl pairs

    def plossl_partner(self, lens_index: int) -> int | None:
        """Index of a lens this one could form a Plossl pair with, or None.

        The partner must be an immediate neighbour, be the same part, and not already be
        grouped; the gap between the two must be at most PLOSSL_SEARCH_WIDTHS lens
        thicknesses. Only positive catalogue lenses can form a Plossl pair (a Plossl is made
        of doublets, so paraxial lenses are excluded).
        """
        lenses = self.state.lenses
        lens = lenses[lens_index]
        if lens.group_id is not None or lens.is_negative or lens.is_paraxial:
            return None
        for other_index in (lens_index - 1, lens_index + 1):
            if not 0 <= other_index < len(lenses):
                continue
            other = lenses[other_index]
            left, right = sorted((lens, other), key=lambda item: item.z)
            gap = right.z - left.back_z
            if (
                other.part_number == lens.part_number
                and other.group_id is None
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
        image plane) shifts with it. Does nothing if the lens has no partner.
        """
        partner_index = self.plossl_partner(lens_index)
        if partner_index is None:
            return
        left_index, right_index = sorted((lens_index, partner_index))
        lenses = self.state.lenses
        left, right = lenses[left_index], lenses[right_index]

        left.flipped = left.positive_element_is_first()  # moves the positive element to the back
        right.flipped = not right.positive_element_is_first()  # keeps or puts it at the front

        original_right_z = right.z
        right.z = self._plossl_front_z(left, right)
        shift = right.z - original_right_z
        for downstream in lenses[right_index + 1 :]:
            downstream.z += shift
        self.state.image_z = max(self.state.image_z + shift, self.state.last_back_z() + 1.0)

        left.group_id = right.group_id = self.state.next_group_id
        self.state.next_group_id += 1

    @staticmethod
    def _plossl_front_z(left: PlacedLens, right: PlacedLens) -> float:
        """Front-vertex z for ``right`` so the facing surfaces are PLOSSL_GAP_MM apart at closest.

        The facing surfaces are the back of ``left`` and the front of ``right`` in their
        current orientations. The gap is measured along z at each height across the smaller
        clear aperture, and the smallest value (at the vertices for two convex surfaces,
        nearer the rim if either is concave) is set to the required gap.
        """
        heights = np.linspace(0.0, min(left.semi_aperture, right.semi_aperture), 200)
        back_sag = surface_sag(left.surfaces_in_use()[-1].radius, heights)
        front_sag = surface_sag(right.surfaces_in_use()[0].radius, heights)
        gap_if_vertices_touched = front_sag - back_sag
        return left.back_z + PLOSSL_GAP_MM - float(gap_if_vertices_touched.min())

    def split_plossl(self, lens_index: int) -> None:
        """Dissolve the lens's group so its lenses move independently again."""
        for index in self.group_indices(lens_index):
            self.state.lenses[index].group_id = None

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
        """Move the image plane to the smallest RMS spot of the selected field, and return its z."""
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

    # ---------------------------------------------------------------- files

    def has_unsaved_changes(self) -> bool:
        """True if the design differs from the one last saved, loaded or started.

        Which field is selected is not saved in a ZMX file, so it is not compared.
        """
        current = replace(self.state, selected_field_index=0)
        return current != replace(self.saved_state, selected_field_index=0)

    def mark_saved(self) -> None:
        """Remember the current design as the saved one.

        Called by ``save_zmx``, ``load_zmx`` and ``new_design``. A front end calls it
        itself after changes that count as part of starting up (e.g. the first autofocus).
        """
        self.saved_state = copy.deepcopy(self.state)

    def new_design(self) -> None:
        """Start again: no lenses, one default field. Counts as saved."""
        self.state = OpticalState()
        self.mark_saved()

    def save_zmx(self, file_path: str) -> None:
        """Write the current system to a Zemax .zmx file."""
        optic_builder.save_zmx(self.state, file_path)
        self.mark_saved()

    def load_zmx(self, file_path: str) -> list[str]:
        """Replace the bench with the system in a Zemax .zmx file.

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
        self.mark_saved()
        return imported.notes
