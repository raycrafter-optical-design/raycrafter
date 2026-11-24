"""Load a Zemax .zmx file as a RayCrafter bench: lenses, fields and image plane.

RayCrafter works with whole lenses, while a Zemax file is a list of surfaces. The file is
read by Optiland, then its surfaces are grouped into lenses: a lens is a run of surfaces
with glass between them, ending at the first surface followed by air. A paraxial surface
becomes a paraxial lens. Each glass lens is looked up in the catalogue, so a lens saved by
RayCrafter (or a vendor lens from any Zemax file) comes back with its part number,
orientation and catalogue data; a lens that is not found keeps its prescription and is
named "Imported lens N".

Some things a Zemax file can describe have no place in RayCrafter yet. Aspheres, mirrors,
tilts and decentres make the import fail with ``ZmxImportError``. Others are simplified,
and each simplification is reported in ``ZmxImport.notes`` so the user can be told.

Fields: Zemax gives all fields one entrance pupil and object, and pairs every field angle
with every wavelength. Each field angle becomes a RayCrafter field at the primary
wavelength, and each other wavelength an on-axis field. The beam is defined at the first
lens, as RayCrafter always does, whatever the file's stop surface.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from optiland.fileio import load_zemax_file

from raycrafter.core.optic_builder import build_optic
from raycrafter.core.prescription import (
    LensSurface,
    flipped_prescription,
    prescription_from_zmx_text,
)
from raycrafter.core.state import BeamField, OpticalState, PlacedLens
from raycrafter.lenscatalog.catalog import CatalogEntry

FIRST_LENS_Z = 100.0  # mm; a Zemax file has no axis start, so the first lens is put here
DEFAULT_SEMI_APERTURE_MM = 12.7  # used for a lens whose file gives no clear aperture
DEFAULT_BEAM_DIAMETER_MM = 10.0  # used when the file's aperture is not an entrance pupil
RADIUS_TOLERANCE = 1e-6  # relative; radii match the catalogue when this close
THICKNESS_TOLERANCE_MM = 1e-3
SEMI_APERTURE_TOLERANCE_MM = 0.01


class ZmxImportError(ValueError):
    """The file cannot be read, or describes something RayCrafter cannot represent."""


@dataclass
class ZmxImport:
    """The result of loading a .zmx file.

    Attributes
    ----------
    state : OpticalState
        The bench described by the file.
    notes : list of str
        Plain-language notes on anything that was simplified or left out.
    """

    state: OpticalState
    notes: list[str] = field(default_factory=list)


def load_zmx(file_path: str, catalog: dict[str, CatalogEntry]) -> ZmxImport:
    """Read a .zmx file and turn it into a bench.

    Parameters
    ----------
    file_path : str
        The Zemax file.
    catalog : dict
        Part number -> CatalogEntry; lenses found here keep their part number.

    Returns
    -------
    ZmxImport
        The bench and any notes about what was simplified.

    Raises
    ------
    ZmxImportError
        If the file cannot be read or uses surfaces RayCrafter does not support.
    """
    # Optiland warns about catalogue and glass details that do not matter here, also later
    # on, when it first looks up a glass's refractive index.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            optic = load_zemax_file(file_path)
        except Exception as error:  # Optiland raises many kinds of error for a bad file
            raise ZmxImportError(f"Optiland could not read the file: {error}") from error

        notes: list[str] = []
        surfaces = optic.surfaces.surfaces
        lenses, image_z = _lenses_from_surfaces(surfaces, catalog, notes)
        # Air surfaces (a stop, say) may come before the first lens; the beam is defined at
        # the lens, so its vergence must count that air too.
        air_before_first_lens_mm = lenses[0].z - FIRST_LENS_Z if lenses else 0.0
        fields = _fields_from_optic(optic, air_before_first_lens_mm, notes)
        state = OpticalState(lenses=lenses, image_z=image_z, fields=fields)
        for lens in state.lenses:
            if not lens.is_paraxial and lens.part_number.startswith("Imported"):
                lens.is_negative = _is_diverging(lens)
    return ZmxImport(state, notes)


# --------------------------------------------------------------------------- lenses


def _lenses_from_surfaces(
    surfaces: list, catalog: dict[str, CatalogEntry], notes: list[str]
) -> tuple[list[PlacedLens], float]:
    """Group the file's surfaces (object first, image last) into lenses.

    Returns the lenses, placed from FIRST_LENS_Z onwards, and the z of the image plane.
    """
    lenses: list[PlacedLens] = []
    current_surfaces: list[LensSurface] = []  # the lens being collected
    current_apertures: list[float | None] = []  # None where the file gives no aperture
    lens_front_z = 0.0
    z = FIRST_LENS_Z  # the vertex z of the surface being read
    unmatched_count = 0

    for surface_number, surface in enumerate(surfaces[1:-1], start=1):
        _check_supported(surface, surface_number)
        radius = float(surface.geometry.radius) if hasattr(surface.geometry, "radius") else np.inf
        thickness = float(surface.thickness)
        semi_aperture = _semi_aperture(surface)
        material = _material_name(surface, surface_number)

        if getattr(surface, "surface_type", "") == "paraxial":
            if current_surfaces:
                raise ZmxImportError(
                    f"Surface {surface_number} is a paraxial surface inside glass."
                )
            focal_length_mm = float(surface.interaction_model.f)
            paraxial_lens = PlacedLens(
                part_number=f"Paraxial f={focal_length_mm:+g} mm",
                surfaces=[],
                semi_aperture=semi_aperture or DEFAULT_SEMI_APERTURE_MM,
                z=z,
                is_negative=focal_length_mm < 0,
                paraxial_focal_length_mm=focal_length_mm,
            )
            lenses.append(paraxial_lens)
        elif material != "air":
            if not current_surfaces:
                lens_front_z = z
            current_surfaces.append(LensSurface(radius, thickness, material))
            current_apertures.append(semi_aperture)
        elif current_surfaces:
            # Air after glass: this surface is the back of the lens being collected.
            current_surfaces.append(LensSurface(radius, 0.0, "air"))
            current_apertures.append(semi_aperture)
            matched = _match_catalogue(current_surfaces, current_apertures, lens_front_z, catalog)
            if matched is None:
                unmatched_count += 1
                matched = _imported_lens(
                    current_surfaces, current_apertures, lens_front_z, unmatched_count, notes
                )
            lenses.append(matched)
            current_surfaces, current_apertures = [], []
        elif not np.isinf(radius):
            notes.append(f"Surface {surface_number} is curved but has air on both sides; ignored.")
        z += thickness

    if current_surfaces:
        raise ZmxImportError("The last lens has no back surface before the image.")
    if unmatched_count:
        notes.append(
            f'{unmatched_count} lens(es) are not in the catalogue and were named "Imported lens N".'
        )
    return lenses, z


def _check_supported(surface, surface_number: int) -> None:
    """Raise ZmxImportError for a surface RayCrafter cannot represent."""
    surface_type = getattr(surface, "surface_type", "standard")
    if surface_type not in ("standard", "paraxial"):
        raise ZmxImportError(
            f"Surface {surface_number} is of type '{surface_type}', which RayCrafter does not "
            "support yet (aspheric lenses are issue #2)."
        )
    if float(getattr(surface.geometry, "k", 0.0) or 0.0) != 0.0:
        raise ZmxImportError(
            f"Surface {surface_number} is a conic, which RayCrafter does not support yet."
        )
    if getattr(surface.interaction_model, "is_reflective", False):
        raise ZmxImportError(f"Surface {surface_number} is a mirror; RayCrafter has no mirrors.")
    cs = surface.geometry.cs
    if any(abs(float(getattr(cs, name, 0.0))) > 1e-9 for name in ("x", "y", "rx", "ry", "rz")):
        raise ZmxImportError(
            f"Surface {surface_number} is tilted or decentred; RayCrafter has one straight axis."
        )


def _material_name(surface, surface_number: int) -> str:
    """The glass after a surface, or "air".

    Air is any material with a refractive index of 1. A glass must have a catalogue name,
    because the bench refers to glasses by name; a Zemax "model glass" (index and Abbe
    number only) has none.
    """
    material = surface.material_post
    index = float(np.asarray(material.n(0.555)).ravel()[0])
    if abs(index - 1.0) < 1e-6:
        return "air"
    name = getattr(material, "name", None)
    if not isinstance(name, str) or not name:
        raise ZmxImportError(
            f"The glass after surface {surface_number} has no catalogue name (a model glass?)."
        )
    return name


def _semi_aperture(surface) -> float | None:
    """The surface's clear semi-aperture (mm) from the file, or None if it gives none."""
    aperture = surface.aperture
    r_max = getattr(aperture, "r_max", None) if aperture is not None else None
    return float(r_max) if r_max is not None else None


def _imported_lens(
    surfaces: list[LensSurface],
    apertures: list[float | None],
    front_z: float,
    lens_number: int,
    notes: list[str],
) -> PlacedLens:
    """A lens that is not in the catalogue, with the file's prescription and aperture."""
    known_apertures = [aperture for aperture in apertures if aperture is not None]
    if known_apertures:
        semi_aperture = min(known_apertures)
    else:
        semi_aperture = DEFAULT_SEMI_APERTURE_MM
        notes.append(
            f"Imported lens {lens_number} has no clear aperture in the file; "
            f"{2 * DEFAULT_SEMI_APERTURE_MM:g} mm was assumed."
        )
    return PlacedLens(
        part_number=f"Imported lens {lens_number}",
        surfaces=surfaces,
        semi_aperture=semi_aperture,
        z=front_z,
    )


def _is_diverging(lens: PlacedLens) -> bool:
    """True if the lens on its own has a negative focal length."""
    alone = OpticalState(lenses=[lens], image_z=lens.back_z + 10.0)
    return float(build_optic(alone).paraxial.f2()) < 0


# ------------------------------------------------------------------ catalogue matching


def _match_catalogue(
    surfaces: list[LensSurface],
    apertures: list[float | None],
    front_z: float,
    catalog: dict[str, CatalogEntry],
) -> PlacedLens | None:
    """The catalogue lens with this prescription, in either orientation, or None.

    Only entries with the same curvatures (read cheaply from their ZMX text) are parsed and
    compared in full, since parsing every entry with Optiland would take seconds. The clear
    aperture must match too when the file gives one.
    """
    curvature_key = _curvature_key([surface.radius for surface in surfaces])
    known_apertures = [aperture for aperture in apertures if aperture is not None]
    for part_number, entry in catalog.items():
        if _catalogue_curvature_key(entry.zmx_text) != curvature_key:
            continue
        if known_apertures and any(
            abs(aperture - entry.semi_aperture) > SEMI_APERTURE_TOLERANCE_MM
            for aperture in known_apertures
        ):
            continue
        catalogue_surfaces = prescription_from_zmx_text(entry.zmx_text)
        for flipped in (False, True):
            oriented = flipped_prescription(catalogue_surfaces) if flipped else catalogue_surfaces
            if _same_prescription(oriented, surfaces):
                return PlacedLens(
                    part_number=part_number,
                    surfaces=catalogue_surfaces,
                    semi_aperture=entry.semi_aperture,
                    z=front_z,
                    flipped=flipped,
                    is_negative=entry.is_negative,
                )
    return None


def _curvature_key(radii: list[float]) -> tuple[float, ...]:
    """Sorted magnitudes of the non-zero curvatures (1/mm), rounded so tiny errors match."""
    curvatures = [abs(1.0 / radius) for radius in radii if np.isfinite(radius) and radius != 0]
    return tuple(sorted(round(curvature, 7) for curvature in curvatures))


_CURV_PATTERN = re.compile(r"^\s*CURV\s+(\S+)", re.MULTILINE)


@lru_cache(maxsize=None)
def _catalogue_curvature_key(zmx_text: str) -> tuple[float, ...]:
    """The curvature key of a catalogue ZMX text, from its CURV lines (cached per text)."""
    curvatures = [float(value) for value in _CURV_PATTERN.findall(zmx_text)]
    return _curvature_key([1.0 / curvature for curvature in curvatures if curvature != 0.0])


def _same_prescription(expected: list[LensSurface], found: list[LensSurface]) -> bool:
    """True if two prescriptions have the same radii, glass thicknesses and glasses."""
    if len(expected) != len(found):
        return False
    for index, (one, other) in enumerate(zip(expected, found)):
        if np.isinf(one.radius) != np.isinf(other.radius):
            return False
        if np.isfinite(one.radius) and not np.isclose(
            one.radius, other.radius, rtol=RADIUS_TOLERANCE
        ):
            return False
        if one.material_after.upper() != other.material_after.upper():
            return False
        is_last = index == len(expected) - 1  # its thickness is the gap after the lens
        if (
            not is_last
            and abs(one.thickness_after - other.thickness_after) > THICKNESS_TOLERANCE_MM
        ):
            return False
    return True


# --------------------------------------------------------------------------- fields


def _fields_from_optic(optic, air_before_first_lens_mm: float, notes: list[str]) -> list[BeamField]:
    """RayCrafter fields from the file's aperture, object, field angles and wavelengths.

    The vergence is that of the beam at the first lens, which is the object distance plus
    ``air_before_first_lens_mm`` away from the object.
    """
    aperture = optic.aperture
    if aperture is not None and aperture.ap_type == "EPD":
        diameter_mm = float(aperture.value)
    else:
        diameter_mm = DEFAULT_BEAM_DIAMETER_MM
        notes.append(
            f"The file's aperture is not an entrance pupil diameter; the beam was set to "
            f"{DEFAULT_BEAM_DIAMETER_MM:g} mm."
        )

    object_distance_mm = float(optic.surfaces.surfaces[0].thickness) + air_before_first_lens_mm
    vergence = 0.0 if np.isinf(object_distance_mm) else -1000.0 / object_distance_mm

    field_type = getattr(optic.fields.field_definition, "name", str(optic.fields.field_definition))
    angles = [float(field_point.y) for field_point in optic.fields.fields] or [0.0]
    if "angle" not in field_type.lower():
        notes.append("The file's fields are not angles; one on-axis field was used instead.")
        angles = [0.0]
    if any(float(field_point.x) != 0.0 for field_point in optic.fields.fields):
        notes.append("Field x coordinates were dropped: RayCrafter beams tilt in y only.")

    wavelengths_um = [float(wavelength.value) for wavelength in optic.wavelengths.wavelengths]
    primary_um = float(optic.wavelengths.primary_wavelength.value)

    def beam(angle_deg: float, wavelength_um: float) -> BeamField:
        return BeamField(
            diameter_mm=diameter_mm,
            angle_deg=angle_deg,
            vergence_dioptres=vergence,
            wavelength_nm=round(wavelength_um * 1000.0, 3),
        )

    fields = [beam(angle, primary_um) for angle in dict.fromkeys(angles)]
    fields += [beam(0.0, wavelength) for wavelength in wavelengths_um if wavelength != primary_um]
    if len(angles) > 1 and len(wavelengths_um) > 1:
        notes.append(
            "Each field angle was loaded at the primary wavelength, and each other wavelength "
            "as an on-axis field."
        )
    return fields
