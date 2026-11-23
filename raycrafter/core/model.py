"""The lens bench model: the one object that front ends (GUI, report, scripts) talk to.

``OpticalModel`` owns the bench state (``raycrafter.core.state``) and the list of lenses
that can be placed. It provides the editing operations: add, delete, move and swap lenses,
and make or split Plossl pairs. It also gives access to the analyses
(``raycrafter.core.analysis``) and to Zemax export, applied to its current state. It has
no Qt code.

Example
-------
>>> from raycrafter.core.model import OpticalModel
>>> model = OpticalModel()  # starts with one ACT508-200-A on the axis
>>> model.autofocus()  # doctest: +SKIP
>>> model.trace_fast().spot.rms_radius  # doctest: +SKIP
"""

from __future__ import annotations

import numpy as np

from raycrafter.core import analysis, optic_builder
from raycrafter.core.prescription import prescription_from_zmx_text, surface_sag
from raycrafter.core.state import AXIS_START_Z, OpticalState, PlacedLens
from raycrafter.lenscatalog import thorlabs
from raycrafter.lenscatalog.catalog import CatalogEntry

DEFAULT_LENS_PART = "ACT508-200-A"  # the lens placed when the model is created
INITIAL_LENS_Z = 100.0  # mm, front vertex of the first lens
LENS_GAP_MM = 10.0  # air gap left between a new lens and its left neighbour
IMAGE_CLEARANCE_MM = 50.0  # a new lens pushes the image plane at least this far behind it
FIRST_LENS_MIN_Z = AXIS_START_Z + 20.0  # lenses cannot be dragged closer to the axis start
PLOSSL_GAP_MM = 1.0  # closest distance between the two facing surfaces of a Plossl pair
PLOSSL_SEARCH_WIDTHS = 2.0  # a Plossl partner must be within this many lens thicknesses


class OpticalModel:
    """A row of lenses on an optical axis, a beam and an image plane.

    Parameters
    ----------
    catalog : dict, optional
        Part number -> CatalogEntry for the lenses that can be placed. Defaults to the
        packaged Thorlabs achromats.
    add_default_lens : bool
        If True (the default), start with one DEFAULT_LENS_PART on the axis.

    Attributes
    ----------
    state : OpticalState
        The lenses, beam and image plane. Front ends may set the beam properties and
        ``image_z`` directly; lens positions should be changed through the methods here.
    catalog : dict
        Part number -> CatalogEntry.
    """

    def __init__(
        self, catalog: dict[str, CatalogEntry] | None = None, add_default_lens: bool = True
    ) -> None:
        self.state = OpticalState()
        self.catalog = catalog if catalog is not None else thorlabs.load_achromats()
        if add_default_lens and DEFAULT_LENS_PART in self.catalog:
            self.add_lens(DEFAULT_LENS_PART)

    # --------------------------------------------------------------- adding and removing

    def add_lens(self, part_number: str) -> PlacedLens:
        """Place a catalogue lens just to the right of the rightmost lens.

        The first lens goes at INITIAL_LENS_Z. The image plane is pushed back if the new
        lens would otherwise end up beyond it.

        Returns
        -------
        PlacedLens
            The new lens.
        """
        entry = self.catalog[part_number]
        lenses = self.state.lenses
        front_z = lenses[-1].back_z + LENS_GAP_MM if lenses else INITIAL_LENS_Z

        new_lens = PlacedLens(
            part_number=part_number,
            surfaces=prescription_from_zmx_text(entry.zmx_text),
            semi_aperture=entry.semi_aperture,
            z=front_z,
            is_negative=entry.focal_length < 0,
        )
        lenses.append(new_lens)
        self.state.image_z = max(self.state.image_z, new_lens.back_z + IMAGE_CLEARANCE_MM)
        return new_lens

    def delete_lens(self, lens_index: int) -> None:
        """Remove one lens. Its Plossl partner, if any, becomes a normal lens."""
        self.split_plossl(lens_index)
        del self.state.lenses[lens_index]

    def flip_lens(self, lens_index: int) -> None:
        """Turn one lens around. Lenses in a Plossl pair cannot be flipped on their own."""
        lens = self.state.lenses[lens_index]
        if lens.group_id is None:
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
        thicknesses. Only positive lenses can form a Plossl pair.
        """
        lenses = self.state.lenses
        lens = lenses[lens_index]
        if lens.group_id is not None or lens.is_negative:
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

    # ---------------------------------------------------------------- optics

    def autofocus(self) -> float:
        """Move the image plane to the z with the smallest RMS spot radius, and return it."""
        self.state.image_z = analysis.find_best_focus_z(self.state)
        return self.state.image_z

    def trace_fast(self) -> analysis.FastTrace:
        """Spots, side-view rays and first-order summary (see ``analysis.trace_fast``)."""
        return analysis.trace_fast(self.state)

    def compute_through_focus(self, fast_trace: analysis.FastTrace) -> analysis.ThroughFocusResult:
        """RMS radius against image plane position (see ``analysis.compute_through_focus``)."""
        return analysis.compute_through_focus(self.state, fast_trace)

    def compute_ray_fan(self) -> analysis.RayFanResult:
        """Transverse ray aberration (see ``analysis.compute_ray_fan``)."""
        return analysis.compute_ray_fan(self.state)

    def compute_mtf(self) -> analysis.MtfResult:
        """FFT MTF at the current beam angle (see ``analysis.compute_mtf``)."""
        return analysis.compute_mtf(self.state)

    def compute_field_curvature(self) -> analysis.FieldCurvatureResult:
        """Best focus against field angle (see ``analysis.compute_field_curvature``)."""
        return analysis.compute_field_curvature(self.state)

    def save_zmx(self, file_path: str) -> None:
        """Write the current system to a Zemax .zmx file."""
        optic_builder.save_zmx(self.state, file_path)
