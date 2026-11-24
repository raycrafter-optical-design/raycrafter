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

import numpy as np

from raycrafter.core import analysis
from raycrafter.core.model import OpticalModel
from raycrafter.core.zmx_import import ZmxImportError
from raycrafter.lenscatalog.catalog import CatalogEntry

MAX_CURVE_POINTS = 25  # curves sent to the agent are thinned to about this many points
DEFAULT_SEARCH_LIMIT = 20


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
        """The whole design: lenses along the axis, fields, image plane, unsaved flag."""
        state = self.model.state
        lenses = [
            {
                "index": index,
                "part_number": lens.part_number,
                "front_z_mm": _number(lens.z),
                "back_z_mm": _number(lens.back_z),
                "semi_aperture_mm": _number(lens.semi_aperture),
                "flipped": lens.flipped,
                "plossl_group": lens.group_id,
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
        return {
            "lenses": lenses,
            "fields": fields,
            "image_z_mm": _number(state.image_z),
            "has_unsaved_changes": self.model.has_unsaved_changes(),
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
        needle = query.lower()
        matches: list[CatalogEntry] = []
        for entry in self.model.catalog.values():
            text = f"{entry.part_number} {entry.family} {entry.coating}".lower()
            if needle not in text:
                continue
            if min_focal_length_mm is not None and entry.focal_length < min_focal_length_mm:
                continue
            if max_focal_length_mm is not None and entry.focal_length > max_focal_length_mm:
                continue
            if min_diameter_mm is not None and entry.clear_diameter_mm < min_diameter_mm:
                continue
            if category is not None and entry.category.lower() != category.lower():
                continue
            matches.append(entry)

        matches.sort(key=lambda entry: (abs(entry.focal_length), entry.part_number))
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

    # ------------------------------------------------------------------ editing lenses

    def add_lens(self, part_number: str) -> dict:
        """Place a catalogue lens just after the last lens (the image plane moves if needed)."""
        self._catalog_entry(part_number)
        self.model.add_lens(part_number)
        return self.get_design()

    def insert_lens(self, part_number: str, front_z_mm: float) -> dict:
        """Place a catalogue lens with its front vertex as near ``front_z_mm`` as fits.

        Lenses never overlap, so the lens goes into the nearest gap it fits in.
        """
        new_lens = self.model.make_lens(self._catalog_entry(part_number).part_number)
        free_z = self.model.free_front_z(new_lens.thickness, front_z_mm)
        if free_z is None:
            raise ValueError(f"{part_number} does not fit anywhere on the bench")
        new_lens.z = free_z
        self.model.insert_lens(new_lens)
        return self.get_design()

    def add_paraxial_lens(self, focal_length_mm: float, front_z_mm: float) -> dict:
        """Place an ideal, aberration-free thin lens (clear diameter 2 inches)."""
        new_lens = self.model.make_paraxial_lens(focal_length_mm)
        free_z = self.model.free_front_z(new_lens.thickness, front_z_mm)
        if free_z is None:
            raise ValueError("A paraxial lens does not fit anywhere on the bench")
        new_lens.z = free_z
        self.model.insert_lens(new_lens)
        return self.get_design()

    def move_lens(self, lens_index: int, front_z_mm: float) -> dict:
        """Move a lens (and its Plossl partner) towards ``front_z_mm``.

        The lens stops at its neighbours; dragging it past the middle of a neighbour swaps
        their order.
        """
        self._check_lens_index(lens_index)
        self.model.move_lens(lens_index, front_z_mm)
        return self.get_design()

    def flip_lens(self, lens_index: int) -> dict:
        """Turn a lens round. Has no effect on paraxial lenses or lenses in a Plossl pair."""
        self._check_lens_index(lens_index)
        self.model.flip_lens(lens_index)
        return self.get_design()

    def delete_lens(self, lens_index: int) -> dict:
        """Remove a lens. Its Plossl partner, if any, becomes an ordinary lens."""
        self._check_lens_index(lens_index)
        self.model.delete_lens(lens_index)
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

    def split_plossl(self, lens_index: int) -> dict:
        """Dissolve the Plossl pair this lens belongs to."""
        self._check_lens_index(lens_index)
        self.model.split_plossl(lens_index)
        return self.get_design()

    # ------------------------------------------------------------------ editing fields, image

    def set_field(
        self,
        field_index: int,
        diameter_mm: float | None = None,
        angle_deg: float | None = None,
        vergence_dioptres: float | None = None,
        offset_mm: float | None = None,
        wavelength_nm: float | None = None,
    ) -> dict:
        """Change any of a field's properties; the ones left out stay as they are."""
        self._check_field_index(field_index)
        if diameter_mm is not None and diameter_mm <= 0:
            raise ValueError("diameter_mm must be positive")
        if wavelength_nm is not None and wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive")

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

    def set_image_z(self, image_z_mm: float) -> dict:
        """Put the image plane at ``image_z_mm`` (it must be behind the last lens)."""
        minimum_z = self.model.state.last_back_z() + 1.0
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

    # ------------------------------------------------------------------ files

    def new_design(self, discard_unsaved: bool = False) -> dict:
        """Start again with an empty bench (no lenses, one default field)."""
        self._check_nothing_to_lose(discard_unsaved)
        self.model.new_design()
        return self.get_design()

    def load_zmx(self, file_path: str, discard_unsaved: bool = False) -> dict:
        """Replace the design with a Zemax .zmx file. Returns the design and any notes."""
        self._check_nothing_to_lose(discard_unsaved)
        try:
            notes = self.model.load_zmx(file_path)
        except (ZmxImportError, OSError) as error:
            raise ValueError(f"Could not load {file_path}: {error}") from error
        return {**self.get_design(), "notes": notes}

    def save_zmx(self, file_path: str) -> dict:
        """Write the design to a Zemax .zmx file."""
        try:
            self.model.save_zmx(file_path)
        except OSError as error:
            raise ValueError(f"Could not save {file_path}: {error}") from error
        return {"saved_to": file_path}

    # ------------------------------------------------------------------ helpers

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
                "The design has unsaved changes. Call save_zmx first, or pass "
                "discard_unsaved=true to throw them away."
            )
