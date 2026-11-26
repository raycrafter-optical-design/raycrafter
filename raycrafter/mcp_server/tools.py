"""The operations an AI agent can perform on a lens bench.

``DesignTools`` wraps one ``OpticalModel``. Every method returns a dictionary of plain Python
values (numbers, strings, lists) so that it can be sent as JSON, and raises ``ValueError`` with
a message the agent can act on when a request makes no sense. This module does not import the
MCP library or Qt, so the same tools can later be pointed at the model inside a running GUI.

Conventions: lens and field indices start at 0 and follow the order along the axis (lenses) or
the order in ``get_design`` (fields); distances are in mm, angles in degrees, vergence in
dioptres (1/m), wavelengths in nm.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from raycrafter.core import analysis
from raycrafter.core.gaps import SOURCE_GAP_INDEX
from raycrafter.core.model import INITIAL_LENS_Z, LENS_GAP_MM, OpticalModel
from raycrafter.core.project_file import ProjectFileError
from raycrafter.core.state import PlacedLens, state_to_dict
from raycrafter.core.zmx_import import ZmxImportError
from raycrafter.lenscatalog.catalog import CatalogEntry
from raycrafter.lenscatalog.selection import search_catalog
from raycrafter.palette import is_hex_colour
from raycrafter.report import figures

MAX_CURVE_POINTS = 25  # curves sent to the agent are thinned to about this many points
DEFAULT_SEARCH_LIMIT = 20
WAVELENGTH_COLOUR = "wavelength"  # set_field(colour=...) value meaning "colour of its wavelength"

# The DesignTools methods that are offered to agents, in the order they are listed.
READ_ONLY_TOOL_NAMES = [
    "get_design",
    "search_lenses",
    "get_lens",
    "get_summary",
    "get_mtf",
    "get_ray_fan",
    "get_through_focus",
    "get_field_curvature",
    "get_plot",
    "list_user_parts",
]
EDITING_TOOL_NAMES = [  # these change the design, so a window must redraw after them
    "add_lens",
    "insert_lens",
    "add_paraxial_lens",
    "move_lens",
    "set_gap",
    "lock_gap",
    "set_source_gap",
    "lock_source_gap",
    "set_source_z",
    "flip_lens",
    "delete_lens",
    "replace_lens",
    "make_plossl",
    "make_assembly",
    "split_assembly",
    "flip_assembly",
    "save_user_part",
    "insert_user_part",
    "set_field",
    "add_field",
    "remove_fields",
    "select_field",
    "set_image_z",
    "autofocus",
    "optimise_gap",
    "new_design",
    "save_project",
    "load_project",
    "load_zmx",
    "save_zmx",
]
TOOL_NAMES = READ_ONLY_TOOL_NAMES + EDITING_TOOL_NAMES

# Calls the window's link answers that are not offered to agents: the Python API
# (``raycrafter.api.Bench``) uses them to run analyses on a copy of the window's design.
LINK_ONLY_TOOL_NAMES = ["get_state"]


def _number(value: float, digits: int = 4) -> float | None:
    """A float rounded for sending, with NaN and infinity turned into None (not valid JSON)."""
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def _thinned(values: np.ndarray, indices: np.ndarray, digits: int = 5) -> list[float | None]:
    """The chosen entries of an array, rounded and made JSON-safe."""
    return [_number(values[index], digits) for index in indices]


def _thinning_indices(point_count: int) -> np.ndarray:
    """Evenly spaced indices, at most MAX_CURVE_POINTS of them, always including the last."""
    if point_count <= MAX_CURVE_POINTS:
        return np.arange(point_count)
    return np.unique(np.linspace(0, point_count - 1, MAX_CURVE_POINTS).round().astype(int))


class DesignTools:
    """What an agent can ask of one ``OpticalModel``.

    Parameters
    ----------
    model : OpticalModel
        The bench to read and edit.
    """

    def __init__(self, model: OpticalModel) -> None:
        self.model = model

    # ------------------------------------------------------------------ reading

    def get_design(self) -> dict:
        """The whole design: lenses along the axis, fields, image plane, unsaved flag.

        ``project_file`` is the RayCrafter project the design was last saved to or opened
        from (null if none). ``imported_from`` names the Zemax file the design was imported
        from and says whether the bench has been changed since (null if not imported).

        Each lens's ``gap_after_mm`` is the air gap from its back vertex to the next lens's
        front vertex, or to the image plane behind the last lens; ``gap_after_locked`` says
        whether that gap is locked (see ``lock_gap``). ``vendor`` and
        ``mount_type`` are null where unknown. ``surface_range`` holds the numbers of the
        lens's first and last surface, as in the exported ZMX file. ``source`` is the source
        plane, where every field's diameter, offset, angle and vergence are defined (see
        ``set_source_gap``). ``assembly`` is the id of
        the assembly the lens is in (null if none), and ``assembly_name`` its name.
        """
        state = self.model.state
        gaps = self.model.air_gaps()
        lenses = [
            {
                "index": index,
                "part_number": lens.part_number,
                "vendor": lens.vendor,
                "mount_type": lens.mount_type,
                "surface_range": list(state.surface_range(index)),
                "front_z_mm": _number(lens.z),
                "back_z_mm": _number(lens.back_z),
                "gap_after_mm": _number(gaps[index].width_mm),
                "gap_after_locked": gaps[index].locked,
                "semi_aperture_mm": _number(lens.semi_aperture),
                "flipped": lens.flipped,
                "assembly": lens.assembly_id,
                "assembly_name": self.model.assembly_name(index),
                "paraxial_focal_length_mm": lens.paraxial_focal_length_mm,
            }
            for index, lens in enumerate(state.lenses)
        ]
        fields = [
            {
                "index": index,
                "diameter_mm": beam.diameter_mm,
                "angle_deg": beam.angle_deg,
                "vergence_dioptres": beam.vergence_dioptres,
                "offset_mm": beam.offset_mm,
                "wavelength_nm": beam.wavelength_nm,
                "selected": index == state.selected_field_index,
            }
            for index, beam in enumerate(state.fields)
        ]
        source_gap = self.model.source_gap()
        return {
            "lenses": lenses,
            "fields": fields,
            "source": {
                "z_mm": _number(state.source_z),
                "gap_to_first_lens_mm": None
                if source_gap is None
                else _number(source_gap.width_mm),
                "gap_locked": state.source_gap_locked,
            },
            "image_z_mm": _number(state.image_z),
            "has_unsaved_changes": self.model.has_unsaved_changes(),
            "project_file": str(self.model.project_path) if self.model.project_path else None,
            "imported_from": None
            if state.imported_from is None
            else {
                "file_name": state.imported_from.file_name,
                "modified_since_import": state.modified_since_import(),
            },
        }

    def search_lenses(
        self,
        query: str = "",
        min_focal_length_mm: float | None = None,
        max_focal_length_mm: float | None = None,
        min_diameter_mm: float | None = None,
        category: str | None = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> dict:
        """Catalogue lenses matching all the given filters, nearest to the focal range first.

        ``query`` is matched (case-insensitively) against the part number, family and coating.
        Focal lengths are signed: a diverging lens has a negative one.
        """
        matches = search_catalog(
            self.model.catalog.values(),
            query,
            min_focal_length_mm,
            max_focal_length_mm,
            min_diameter_mm,
            category,
        )
        return {
            "total_matches": len(matches),
            "lenses": [self._lens_summary(entry) for entry in matches[: max(limit, 0)]],
        }

    def get_lens(self, part_number: str) -> dict:
        """The catalogue details of one lens."""
        return self._lens_summary(self._catalog_entry(part_number))

    def get_summary(self) -> dict:
        """First-order properties and the spot of every field at the current image plane."""
        state = self.model.state
        fast_trace = self.model.trace_fast()
        f_number = self.model.compute_f_number()

        summary = fast_trace.summary
        spot_results = []
        for index, spot in enumerate(fast_trace.spots):
            airy = analysis.airy_radius(spot)
            spot_results.append(
                {
                    "field_index": index,
                    "rms_radius_um": _number(spot.rms_radius * 1000.0),
                    "airy_radius_um": None if airy is None else _number(airy * 1000.0),
                    "rays_transmitted": len(spot.x),
                }
            )
        return {
            "effective_focal_length_mm": None
            if summary is None
            else _number(summary.effective_focal_length),
            "back_focal_length_mm": None if summary is None else _number(summary.back_focal_length),
            "f_number": None if f_number is None else _number(f_number.f_number),
            "transmitted_beam_diameter_mm": None
            if f_number is None
            else _number(f_number.transmitted_beam_diameter),
            "beam_is_clipped": None if f_number is None else f_number.beam_is_clipped,
            "image_z_mm": _number(state.image_z),
            "spots": spot_results,
        }

    def get_mtf(self) -> dict:
        """MTF of the selected field against spatial frequency, with the diffraction limit."""
        result = self.model.compute_mtf()
        indices = _thinning_indices(len(result.frequency))
        return {
            "field_index": self.model.state.selected_field_index,
            "frequency_cycles_per_mm": _thinned(result.frequency, indices, 2),
            "tangential": _thinned(result.tangential, indices),
            "sagittal": _thinned(result.sagittal, indices),
            "diffraction_limit": _thinned(result.diffraction_limit, indices),
            "note": "Empty lists mean the beam is blocked, so no MTF can be computed.",
        }

    def get_ray_fan(self) -> dict:
        """Transverse ray aberration (um) against pupil position, selected field."""
        result = self.model.compute_ray_fan()
        indices = _thinning_indices(len(result.pupil))
        return {
            "field_index": self.model.state.selected_field_index,
            "pupil_position": _thinned(result.pupil, indices, 3),
            "tangential_error_um": _thinned(result.tangential_error * 1000.0, indices, 3),
            "sagittal_error_um": _thinned(result.sagittal_error * 1000.0, indices, 3),
            "note": "null means the ray was blocked.",
        }

    def get_through_focus(self) -> dict:
        """RMS spot radius (um) of the selected field against image plane offset (mm)."""
        fast_trace = self.model.trace_fast()
        result = self.model.compute_through_focus(fast_trace)
        indices = _thinning_indices(len(result.offsets))
        return {
            "field_index": fast_trace.selected_field_index,
            "offset_from_image_plane_mm": _thinned(result.offsets, indices, 3),
            "rms_radius_um": _thinned(result.rms_current * 1000.0, indices, 3),
            "best_offset_mm": _number(result.best_offset),
            "airy_radius_um": None
            if result.airy_radius is None
            else _number(result.airy_radius * 1000.0),
        }

    def get_field_curvature(self) -> dict:
        """Best-focus shift (mm, from the image plane) of both fans against field angle."""
        result = self.model.compute_field_curvature()
        indices = _thinning_indices(len(result.angles))
        return {
            "angle_deg": _thinned(result.angles, indices, 3),
            "tangential_shift_mm": _thinned(result.tangential_shift, indices, 3),
            "sagittal_shift_mm": _thinned(result.sagittal_shift, indices, 3),
            "note": "null means too few rays got through to find a focus.",
        }

    def get_plot(self, kind: str) -> bytes:
        """A PNG picture of one analysis: the same plot as the GUI and the HTML report.

        ``kind`` is one of ``figures.PLOT_KINDS``. MTF, ray fan and through-focus show the
        selected field.
        """
        return figures.figure_to_png_bytes(figures.draw_named_plot(self.model.state, kind))

    def get_state(self) -> dict:
        """The complete design, every surface included, as ``state_to_dict`` gives it.

        Not offered to agents (``get_design`` is the readable summary); the Python API uses
        it to copy a window's design.
        """
        return state_to_dict(self.model.state)

    # ------------------------------------------------------------------ editing lenses

    def add_lens(self, part_number: str) -> dict:
        """Place a catalogue lens just after the last lens (the image plane moves if needed).

        The result includes ``new_lens_index``.
        """
        self._catalog_entry(part_number)
        self.model.add_lens(part_number)
        return {**self.get_design(), "new_lens_index": len(self.model.state.lenses) - 1}

    def insert_lens(self, part_number: str, front_z_mm: float) -> dict:
        """Place a catalogue lens with its front vertex as near ``front_z_mm`` as fits.

        Lenses never overlap, so the lens goes into the nearest gap it fits in. The result
        includes ``new_lens_index``.
        """
        new_lens = self.model.make_lens(self._catalog_entry(part_number).part_number)
        return self._place(new_lens, front_z_mm, f"{part_number} does not fit anywhere")

    def add_paraxial_lens(self, focal_length_mm: float, front_z_mm: float | None = None) -> dict:
        """Place an ideal, aberration-free thin lens (clear diameter 2 inches).

        With no ``front_z_mm`` it goes just after the last lens, as ``add_lens`` does. The
        result includes ``new_lens_index``.
        """
        new_lens = self.model.make_paraxial_lens(focal_length_mm)
        if front_z_mm is None:
            lenses = self.model.state.lenses
            front_z_mm = lenses[-1].back_z + LENS_GAP_MM if lenses else INITIAL_LENS_Z
        return self._place(new_lens, front_z_mm, "A paraxial lens does not fit anywhere")

    def move_lens(self, lens_index: int, front_z_mm: float) -> dict:
        """Move a lens towards ``front_z_mm``, with the lenses joined to it by locked gaps.

        The lens stops at its neighbours; dragging it past the middle of a neighbour swaps
        their order. The result includes ``moved_lens_index``, the lens's index afterwards.
        """
        self._check_lens_index(lens_index)
        new_order = self.model.move_lens(lens_index, front_z_mm)
        moved_lens_index = lens_index if new_order is None else new_order.index(lens_index)
        return {**self.get_design(), "moved_lens_index": moved_lens_index}

    def set_gap(self, lens_index: int, gap_mm: float) -> dict:
        """Set the air gap behind a lens: back vertex to the next front vertex.

        Behind the last lens it is the distance to the image plane. All later lenses and the
        image plane shift together, so no other gap changes. A locked gap (always so inside an
        assembly) cannot be set, and surfaces may not overlap.
        """
        self._check_lens_index(lens_index)
        self.model.set_gap(lens_index, gap_mm)
        return self.get_design()

    def lock_gap(self, lens_index: int, locked: bool = True) -> dict:
        """Lock (or, with ``locked=False``, unlock) the air gap behind a lens.

        A locked gap keeps its length: lenses joined by locked gaps move together, and
        ``set_gap`` refuses to change it. Behind the last lens it is the gap to the image
        plane, which then moves with the last lens and cannot be autofocused or moved on its
        own. The gaps inside an assembly are always locked.
        """
        self._check_lens_index(lens_index)
        self.model.set_gap_locked(lens_index, bool(locked))
        return self.get_design()

    def flip_lens(self, lens_index: int) -> dict:
        """Turn a lens round. No effect on paraxial lenses or lenses in an assembly."""
        self._check_lens_index(lens_index)
        self.model.flip_lens(lens_index)
        return self.get_design()

    def delete_lens(self, lens_index: int) -> dict:
        """Remove a lens. Its assembly, if any, is dissolved."""
        self._check_lens_index(lens_index)
        self.model.delete_lens(lens_index)
        return self.get_design()

    def replace_lens(
        self,
        lens_index: int,
        part_number: str | None = None,
        paraxial_focal_length_mm: float | None = None,
        plossl_pair: str | None = None,
    ) -> dict:
        """Swap a lens for a catalogue lens (``part_number``) or a paraxial one, in place.

        Give exactly one of ``part_number`` and ``paraxial_focal_length_mm``. The new lens is
        placed so its rear principal plane is where the old lens's was, so a collimated beam
        focuses in about the same place; a locked gap in front keeps the front vertex, and
        one behind keeps the back vertex. It takes over the old lens's assembly, the lock
        behind it, and (catalogue for catalogue) whether it is flipped. It is refused if it
        would overlap a neighbour. For a lens of a Plossl pair, ``plossl_pair`` must be
        "replace_both" (two of the new part, still a Plossl pair) or "split" (split the pair,
        replace this lens only). To find lenses of the same focal length, use search_lenses.
        """
        self._check_lens_index(lens_index)
        if (part_number is None) == (paraxial_focal_length_mm is None):
            raise ValueError("Give exactly one of part_number and paraxial_focal_length_mm")
        if plossl_pair not in (None, "replace_both", "split"):
            raise ValueError('plossl_pair must be "replace_both", "split" or null')
        if plossl_pair == "replace_both":
            if part_number is None:
                raise ValueError("A Plossl pair can only be replaced with a catalogue lens")
            self._catalog_entry(part_number)
            self.model.replace_plossl_pair(lens_index, part_number)
            return self.get_design()
        if paraxial_focal_length_mm is not None:
            new_lens = self.model.make_paraxial_lens(paraxial_focal_length_mm)
        else:
            new_lens = self.model.make_lens(self._catalog_entry(str(part_number)).part_number)
        self.model.replace_lens(lens_index, new_lens, split_plossl_pair=plossl_pair == "split")
        return self.get_design()

    def make_plossl(self, lens_index: int) -> dict:
        """Snap this lens and its identical neighbouring doublet into a Plossl pair."""
        self._check_lens_index(lens_index)
        if self.model.plossl_partner(lens_index) is None:
            raise ValueError(
                "This lens has no Plossl partner: the neighbour must be the same positive "
                "catalogue lens, close by, and not already paired."
            )
        self.model.make_plossl(lens_index)
        return self.get_design()

    def make_assembly(
        self, first_lens_index: int, last_lens_index: int, name: str | None = None
    ) -> dict:
        """Group a run of neighbouring lenses into an assembly that moves and flips as one.

        The gaps between them are locked (their spacing is kept as it is now). Assemblies
        wholly inside the run are absorbed. The result includes ``assembly``, the new id.
        """
        self._check_lens_index(first_lens_index)
        self._check_lens_index(last_lens_index)
        assembly_id = self.model.make_assembly(first_lens_index, last_lens_index, name)
        return {**self.get_design(), "assembly": assembly_id}

    def split_assembly(self, lens_index: int) -> dict:
        """Dissolve the assembly (e.g. a Plossl pair) this lens is in, unlocking its gaps."""
        self._check_lens_index(lens_index)
        self.model.split_assembly(lens_index)
        return self.get_design()

    def flip_assembly(self, lens_index: int) -> dict:
        """Turn this lens's assembly round: its lenses in reverse order, each one flipped.

        It stays in the same stretch of the axis. Lens indices inside it are reversed.
        """
        self._check_lens_index(lens_index)
        if self.model.state.lenses[lens_index].assembly_id is None:
            raise ValueError("This lens is in no assembly; use flip_lens to flip it")
        self.model.flip_assembly(lens_index)
        return self.get_design()

    def save_user_part(self, lens_index: int, name: str, overwrite: bool = False) -> dict:
        """Save this lens's assembly as a user part, which can be inserted in any design.

        The result includes ``file`` (where it was saved). Refused if a part of that name
        exists, unless ``overwrite`` is true.
        """
        self._check_lens_index(lens_index)
        file_path = self.model.save_assembly_as_part(lens_index, name, overwrite)
        return {"file": str(file_path), "user_parts": self.list_user_parts()["user_parts"]}

    def list_user_parts(self) -> dict:
        """The user parts that can be inserted: name, lens part numbers and length (mm)."""
        return {
            "user_parts": [
                {
                    "name": part.name,
                    "lenses": [lens.part_number for lens in part.lenses],
                    "length_mm": _number(part.length_mm),
                }
                for part in self.model.user_parts()
            ]
        }

    def insert_user_part(self, name: str, front_z_mm: float) -> dict:
        """Place a user part (as one assembly) with its front as near ``front_z_mm`` as fits.

        The result includes ``new_lens_indices``.
        """
        part = self.model.user_part(name)
        free_z = self.model.free_front_z(part.length_mm, front_z_mm)
        if free_z is None:
            raise ValueError(f"The user part {name!r} does not fit anywhere on the bench")
        new_indices = self.model.insert_lenses(part.lenses_at(free_z), assembly_name=part.name)
        return {**self.get_design(), "new_lens_indices": new_indices}

    # ------------------------------------------------------------------ editing fields, image

    def set_field(
        self,
        field_index: int,
        diameter_mm: float | None = None,
        angle_deg: float | None = None,
        vergence_dioptres: float | None = None,
        offset_mm: float | None = None,
        wavelength_nm: float | None = None,
        colour: str | None = None,
    ) -> dict:
        """Change any of a field's properties; the ones left out stay as they are.

        A field's diameter, offset, angle and vergence are those at the source plane (see
        ``set_source_gap``), which by default is at the first lens.

        ``colour`` is a hex colour such as ``"#ff8800"``, or ``"wavelength"`` to draw the field
        in the colour of its wavelength (the default).
        """
        self._check_field_index(field_index)
        numbers = {
            "diameter_mm": diameter_mm,
            "angle_deg": angle_deg,
            "vergence_dioptres": vergence_dioptres,
            "offset_mm": offset_mm,
            "wavelength_nm": wavelength_nm,
        }
        for name, value in numbers.items():
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if diameter_mm is not None and diameter_mm <= 0:
            raise ValueError("diameter_mm must be positive")
        if wavelength_nm is not None and wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive")
        if colour is not None and colour != WAVELENGTH_COLOUR and not is_hex_colour(colour):
            raise ValueError(
                f"colour must be a hex colour such as '#ff8800', or {WAVELENGTH_COLOUR!r}"
            )

        beam = self.model.state.fields[field_index]
        if diameter_mm is not None:
            beam.diameter_mm = diameter_mm
        if angle_deg is not None:
            beam.angle_deg = angle_deg
        if vergence_dioptres is not None:
            beam.vergence_dioptres = vergence_dioptres
        if offset_mm is not None:
            beam.offset_mm = offset_mm
        if wavelength_nm is not None:
            beam.wavelength_nm = wavelength_nm
        if colour is not None:
            beam.colour = None if colour == WAVELENGTH_COLOUR else colour
        return self.get_design()

    def add_field(self) -> dict:
        """Add a field (a copy of the selected one) and select it."""
        self.model.add_field()
        return self.get_design()

    def remove_fields(self, field_indices: list[int]) -> dict:
        """Remove fields by index. At least one field always remains."""
        for field_index in field_indices:
            self._check_field_index(field_index)
        self.model.remove_fields(field_indices)
        return self.get_design()

    def select_field(self, field_index: int) -> dict:
        """Choose the field that autofocus, MTF, ray fan and through-focus analyse."""
        self._check_field_index(field_index)
        self.model.select_field(field_index)
        return self.get_design()

    def set_source_gap(self, gap_mm: float) -> dict:
        """Set the distance from the source plane to the first lens's front vertex.

        The source plane is where every field is defined: a light source, a scan mirror or a
        stop. Every lens and the image plane shift together; the source plane stays put.
        Refused while the gap is locked (it is by default, at 0 mm, so the fields are
        defined at the first lens): unlock it with ``lock_source_gap(locked=False)``.
        """
        self.model.set_source_gap(gap_mm)
        return self.get_design()

    def lock_source_gap(self, locked: bool = True) -> dict:
        """Lock (or unlock) the gap from the source plane to the first lens.

        While locked, the source plane moves with the first lens, keeping its distance.
        While unlocked it stays put, so moving the first lens changes where an angled beam
        lands on it, and lenses cannot be moved in front of it.
        """
        self.model.set_source_gap_locked(bool(locked))
        return self.get_design()

    def set_source_z(self, source_z_mm: float) -> dict:
        """Move the source plane to ``source_z_mm``, between the axis start and the first lens.

        Refused while the gap from the source plane is locked.
        """
        self.model.set_source_z(source_z_mm)
        return self.get_design()

    def set_image_z(self, image_z_mm: float) -> dict:
        """Put the image plane at ``image_z_mm`` (it must be behind the last lens).

        Refused if the gap to the image plane is locked.
        """
        if self.model.state.lenses and self.model.state.image_gap_locked:
            raise ValueError("The gap to the image plane is locked: unlock it to move the image")
        minimum_z = self.model.state.last_back_z() + 1.0
        if not math.isfinite(image_z_mm):
            raise ValueError("image_z_mm must be a finite number")
        if image_z_mm < minimum_z:
            raise ValueError(
                f"The image plane must be at z >= {minimum_z:.1f} mm (behind the lenses)"
            )
        self.model.state.image_z = image_z_mm
        return self.get_design()

    def autofocus(self) -> dict:
        """Move the image plane to the smallest RMS spot of the selected field."""
        if not self.model.state.lenses:
            raise ValueError("There are no lenses to focus")
        self.model.autofocus()
        return self.get_design()

    def optimise_gap(
        self, lens_index: int, goal: str, apply: bool = True, lock: bool = False
    ) -> dict:
        """Find (and by default set) the gap behind a lens that best meets a goal.

        ``goal`` is "collimate" (the rays leaving the last lens are parallel; merit is their
        RMS angle in µrad; later lenses and the image plane move together) or "focus" (the
        smallest RMS spot on the image plane, which stays put; merit in µm). Real rays of the
        selected field are used. ``lens_index`` -1 means the gap from the source plane. The
        gap must be unlocked and not the gap to the image plane (use autofocus for that).
        Gaps that clip rays are never chosen. The result's ``optimisation`` gives the gap
        and merit before and after, and ``at_bound`` (true if the best gap is at an end of
        the allowed range, so there may be no real solution). With ``apply`` false the bench
        is not changed; with ``lock`` true the gap is locked afterwards.
        """
        if lens_index != SOURCE_GAP_INDEX:
            self._check_lens_index(lens_index)
        result = self.model.optimise_gap(lens_index, goal)
        if apply:
            self.model.apply_gap_optimisation(result, lock=bool(lock))
        optimisation = {
            name: _number(value) if isinstance(value, float) else value
            for name, value in asdict(result).items()
        }
        return {**self.get_design(), "optimisation": optimisation}

    # ------------------------------------------------------------------ files

    def new_design(self, discard_unsaved: bool = False) -> dict:
        """Start again with an empty bench (no lenses, one default field)."""
        self._check_nothing_to_lose(discard_unsaved)
        self.model.new_design()
        return self.get_design()

    def save_project(self, file_path: str | None = None, forget_import: bool = False) -> dict:
        """Save the whole design as a RayCrafter project (.raycrafter), keeping everything.

        Without ``file_path`` it saves to the project file last saved or opened. This is how
        to save a design: a Zemax file (``save_zmx``) loses fields and assemblies. With
        ``forget_import`` the project no longer keeps the Zemax file the design was imported
        from, nor the design as imported; only do this if the user asks.
        """
        try:
            saved_path = self.model.save_project(file_path, forget_import=forget_import)
        except OSError as error:
            raise ValueError(f"Could not save {file_path}: {error}") from error
        return {"saved_to": str(saved_path)}

    def load_project(self, file_path: str, discard_unsaved: bool = False) -> dict:
        """Replace the design with a saved RayCrafter project (.raycrafter)."""
        self._check_nothing_to_lose(discard_unsaved)
        try:
            self.model.load_project(file_path)
        except ProjectFileError as error:
            raise ValueError(f"Could not open {file_path}: {error}") from error
        return self.get_design()

    def load_zmx(self, file_path: str, discard_unsaved: bool = False) -> dict:
        """Import a Zemax .zmx file in place of the design. Returns the design and any notes.

        The imported design has no project file until ``save_project`` is called.
        """
        self._check_nothing_to_lose(discard_unsaved)
        try:
            notes = self.model.load_zmx(file_path)
        except (ZmxImportError, OSError) as error:
            raise ValueError(f"Could not load {file_path}: {error}") from error
        return {**self.get_design(), "notes": notes}

    def save_zmx(self, file_path: str) -> dict:
        """Export the design to a Zemax .zmx file.

        This does not save the design (``has_unsaved_changes`` stays true): a ZMX file keeps
        only the selected field, and no assemblies. Use ``save_project`` to save it.
        """
        try:
            self.model.save_zmx(file_path)
        except OSError as error:
            raise ValueError(f"Could not save {file_path}: {error}") from error
        return {"saved_to": file_path}

    # ------------------------------------------------------------------ helpers

    def _place(self, new_lens: PlacedLens, front_z_mm: float, no_room_message: str) -> dict:
        """Put a lens not yet on the bench into the free position nearest ``front_z_mm``."""
        free_z = self.model.free_front_z(new_lens.thickness, front_z_mm)
        if free_z is None:
            raise ValueError(f"{no_room_message} on the bench")
        new_lens.z = free_z
        new_lens_index = self.model.insert_lens(new_lens)
        return {**self.get_design(), "new_lens_index": new_lens_index}

    def _catalog_entry(self, part_number: str) -> CatalogEntry:
        """The catalogue entry for a part number, or a ValueError that suggests a search."""
        entry = self.model.catalog.get(part_number)
        if entry is None:
            raise ValueError(f"Unknown part number {part_number!r}; use search_lenses to find one")
        return entry

    @staticmethod
    def _lens_summary(entry: CatalogEntry) -> dict:
        """The catalogue facts about a lens that matter when choosing one."""
        return {
            "part_number": entry.part_number,
            "vendor": entry.vendor,
            "mount_type": entry.mount_type,
            "focal_length_mm": _number(entry.focal_length),
            "clear_diameter_mm": _number(entry.clear_diameter_mm),
            "category": entry.category,
            "family": entry.family,
            "coating": entry.coating,
        }

    def _check_lens_index(self, lens_index: int) -> None:
        lens_count = len(self.model.state.lenses)
        if not 0 <= lens_index < lens_count:
            raise ValueError(f"There is no lens {lens_index}; the bench has {lens_count} lens(es)")

    def _check_field_index(self, field_index: int) -> None:
        field_count = len(self.model.state.fields)
        if not 0 <= field_index < field_count:
            raise ValueError(f"There is no field {field_index}; there are {field_count} field(s)")

    def _check_nothing_to_lose(self, discard_unsaved: bool) -> None:
        """Refuse to throw away unsaved work unless the agent says that is intended."""
        if self.model.has_unsaved_changes() and not discard_unsaved:
            raise ValueError(
                "The design has unsaved changes. Call save_project first, or pass "
                "discard_unsaved=true to throw them away."
            )
