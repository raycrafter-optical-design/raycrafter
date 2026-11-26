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
wavelength, and each other wavelength an on-axis field. When there are several angles,
each is drawn in its own colour, as Zemax does. When the stop is in air in front of the
first lens (a scan mirror, say), each beam's chief ray passes through the centre of the
stop, so it reaches the first lens off axis; otherwise the beams are centred on the first
lens, as RayCrafter's own fields are.

Clear apertures: Zemax blocks rays only at surfaces with an aperture (CLAP, FLAP, SQAP);
a semi-diameter (DIAM) only sets the drawn size. RayCrafter blocks rays at every lens rim.
A lens with no aperture in the file keeps its catalogue or drawn size, which is usually the
real one, and the user is told if that blocks rays Zemax lets through. Apertures on surfaces
in air cannot be modelled and are reported.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from optiland.fileio import load_zemax_file

from raycrafter import palette
from raycrafter.core.optic_builder import build_optic
from raycrafter.core.prescription import (
    LensSurface,
    flipped_prescription,
    prescription_from_zmx_text,
)
from raycrafter.core.state import BeamField, OpticalState, PlacedLens
from raycrafter.lenscatalog.catalog import CatalogEntry

FIRST_SURFACE_Z = 100.0  # mm; a Zemax file has no axis start, so its surface 1 is put here
DEFAULT_SEMI_APERTURE_MM = 12.7  # used for a lens whose file gives no size at all
DEFAULT_BEAM_DIAMETER_MM = 10.0  # used when the file's aperture is not an entrance pupil
RADIUS_TOLERANCE = 1e-6  # relative; radii match the catalogue when this close
THICKNESS_TOLERANCE_MM = 1e-3
SEMI_APERTURE_TOLERANCE_MM = 0.01
RAY_CHECK_RINGS = 6  # hexapolar rings per field and wavelength when checking apertures


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

        surfaces = optic.surfaces.surfaces
        surface_extras = _read_surface_extras(file_path, len(surfaces))
        notes: list[str] = []
        grouping = _lenses_from_surfaces(surfaces, surface_extras, catalog, notes)

        # The beam is defined at the first lens; with none, the fields are just on axis.
        first_lens_z = grouping.lenses[0].z if grouping.lenses else grouping.image_z
        first_lens_surface = grouping.lens_surface_numbers[0][0] if grouping.lenses else None
        fields = _fields_from_optic(optic, first_lens_z, first_lens_surface, notes)

        ray_extent = _traced_ray_extent(optic)
        _note_lenses_that_block_more_than_zemax(grouping, ray_extent, notes)
        _note_apertures_in_air(grouping.air_apertures, ray_extent, notes)

        state = OpticalState(lenses=grouping.lenses, image_z=grouping.image_z, fields=fields)
        for lens in state.lenses:
            if not lens.is_paraxial and lens.part_number.startswith("Imported"):
                lens.is_negative = _is_diverging(lens)
    return ZmxImport(state, notes)


# ------------------------------------------------------- what Optiland does not read
#
# Optiland workaround (Optiland 0.6.3, upstream issue not yet filed): its Zemax reader
# ignores square apertures (SQAP) and surface comments (COMM), and keeps the semi-diameter
# (DIAM) only for the stop. RayCrafter needs all three, so they are read from the file text
# here. Once Optiland reads them, this section and its callers should use Optiland instead.


@dataclass
class SurfaceExtras:
    """What the file says about one surface beyond what Optiland reads (all optional).

    Attributes
    ----------
    comment : str
        The surface's comment (COMM), e.g. "fold mirror".
    semi_diameter_mm : float or None
        The drawn semi-diameter (DIAM). Zemax does not block rays outside it.
    square_half_widths_mm : tuple of float or None
        Half-widths in x and y of a rectangular aperture (SQAP), which does block rays.
    """

    comment: str = ""
    semi_diameter_mm: float | None = None
    square_half_widths_mm: tuple[float, float] | None = None


_SURFACE_START = re.compile(r"^SURF\s+\d+", re.MULTILINE)


def _read_surface_extras(file_path: str, surface_count: int) -> list[SurfaceExtras]:
    """Comment, semi-diameter and square aperture of every surface, from the file text.

    Returns ``surface_count`` empty entries if the text cannot be matched to Optiland's
    surfaces (a different count), so that the import goes on without these details.
    """
    with open(file_path, "rb") as file:
        raw = file.read()
    # Zemax writes UTF-16 files with a byte-order mark; older files and RayCrafter's are UTF-8.
    is_utf16 = raw[:2] in (b"\xff\xfe", b"\xfe\xff")
    text = raw.decode("utf-16" if is_utf16 else "utf-8", errors="replace")

    starts = [match.start() for match in _SURFACE_START.finditer(text)]
    blocks = [text[start:end] for start, end in zip(starts, starts[1:] + [len(text)])]
    if len(blocks) != surface_count:
        return [SurfaceExtras() for _ in range(surface_count)]

    extras = []
    for block in blocks:
        surface = SurfaceExtras()
        for line in block.splitlines()[1:]:
            words = line.split()
            if not words:
                continue
            if words[0] == "COMM":
                surface.comment = line.strip()[len("COMM") :].strip()
            elif words[0] == "DIAM" and len(words) > 1:
                surface.semi_diameter_mm = float(words[1])
            elif words[0] == "SQAP" and len(words) > 2:
                surface.square_half_widths_mm = (float(words[1]), float(words[2]))
        extras.append(surface)
    return extras


# --------------------------------------------------------------------------- lenses


@dataclass
class ClearAperture:
    """A surface's aperture from the file: round, or rectangular (SQAP).

    Attributes
    ----------
    semi_aperture_mm : float
        Radius of the round aperture, or of the largest circle inside the rectangle.
    square_half_widths_mm : tuple of float or None
        For a rectangular aperture, its half-widths in x and y.
    """

    semi_aperture_mm: float
    square_half_widths_mm: tuple[float, float] | None = None

    def describe(self) -> str:
        """E.g. "a 38 mm round aperture" or "a 35 x 48 mm rectangular aperture"."""
        if self.square_half_widths_mm is None:
            return f"a {2 * self.semi_aperture_mm:g} mm round aperture"
        half_x, half_y = self.square_half_widths_mm
        return f"a {2 * half_x:g} x {2 * half_y:g} mm rectangular aperture"


@dataclass
class AirAperture:
    """An aperture on a surface with air on both sides, which RayCrafter cannot model."""

    surface_number: int
    comment: str
    aperture: ClearAperture


@dataclass
class SurfaceGrouping:
    """The file's surfaces turned into lenses, plus what could not be turned into anything.

    Attributes
    ----------
    lenses : list of PlacedLens
        Placed from FIRST_SURFACE_Z onwards.
    lens_surface_numbers : list of list of int
        For each lens, the numbers of the file surfaces it is made of.
    lens_has_file_aperture : list of bool
        For each lens, True if the file gives one of its surfaces an aperture. Otherwise
        Zemax blocks no rays at the lens, whatever size it is drawn.
    air_apertures : list of AirAperture
        Apertures on surfaces in air, which were left out.
    image_z : float
        z of the image plane.
    """

    lenses: list[PlacedLens] = field(default_factory=list)
    lens_surface_numbers: list[list[int]] = field(default_factory=list)
    lens_has_file_aperture: list[bool] = field(default_factory=list)
    air_apertures: list[AirAperture] = field(default_factory=list)
    image_z: float = 0.0

    def add(
        self,
        lens: PlacedLens,
        surface_numbers: list[int],
        apertures: list[ClearAperture | None],
    ) -> None:
        """Append one lens, the file surfaces it came from and their apertures."""
        self.lenses.append(lens)
        self.lens_surface_numbers.append(surface_numbers)
        self.lens_has_file_aperture.append(any(aperture is not None for aperture in apertures))


def _lenses_from_surfaces(
    surfaces: list,
    surface_extras: list[SurfaceExtras],
    catalog: dict[str, CatalogEntry],
    notes: list[str],
) -> SurfaceGrouping:
    """Group the file's surfaces (object first, image last) into lenses."""
    grouping = SurfaceGrouping()
    current_surfaces: list[LensSurface] = []  # the lens being collected
    current_numbers: list[int] = []  # its file surface numbers
    current_apertures: list[ClearAperture | None] = []  # None where the file gives none
    lens_front_z = 0.0
    z = FIRST_SURFACE_Z  # the vertex z of the surface being read
    unmatched_count = 0

    for surface_number, surface in enumerate(surfaces[1:-1], start=1):
        _check_supported(surface, surface_number)
        radius = float(surface.geometry.radius) if hasattr(surface.geometry, "radius") else np.inf
        thickness = float(surface.thickness)
        extras = surface_extras[surface_number]
        aperture = _clear_aperture(surface, extras)
        material = _material_name(surface, surface_number)

        if getattr(surface, "surface_type", "") == "paraxial":
            if current_surfaces:
                raise ZmxImportError(
                    f"Surface {surface_number} is a paraxial surface inside glass."
                )
            paraxial_lens = _paraxial_lens(surface, z, aperture, [extras], notes)
            grouping.add(paraxial_lens, [surface_number], [aperture])
        elif material != "air":
            if not current_surfaces:
                lens_front_z = z
            current_surfaces.append(LensSurface(radius, thickness, material))
            current_numbers.append(surface_number)
            current_apertures.append(aperture)
        elif current_surfaces:
            # Air after glass: this surface is the back of the lens being collected.
            current_surfaces.append(LensSurface(radius, 0.0, "air"))
            current_numbers.append(surface_number)
            current_apertures.append(aperture)
            matched = _match_catalogue(current_surfaces, current_apertures, lens_front_z, catalog)
            if matched is None:
                unmatched_count += 1
                lens_extras = [surface_extras[number] for number in current_numbers]
                matched = _imported_lens(
                    current_surfaces,
                    current_apertures,
                    lens_extras,
                    lens_front_z,
                    unmatched_count,
                    notes,
                )
            grouping.add(matched, current_numbers, current_apertures)
            current_surfaces, current_numbers, current_apertures = [], [], []
        else:
            # Air on both sides: a dummy surface, a stop or a baffle. Only its thickness and
            # any aperture matter.
            if not np.isinf(radius):
                notes.append(
                    f"Surface {surface_number} is curved but has air on both sides; ignored."
                )
            if aperture is not None:
                grouping.air_apertures.append(AirAperture(surface_number, extras.comment, aperture))
        z += thickness

    if current_surfaces:
        raise ZmxImportError("The last lens has no back surface before the image.")
    if unmatched_count:
        notes.append(
            f'{unmatched_count} lens(es) are not in the catalogue and were named "Imported lens N".'
        )
    grouping.image_z = z
    return grouping


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


def _clear_aperture(surface, extras: SurfaceExtras) -> ClearAperture | None:
    """The surface's aperture in the file (round from Optiland, or SQAP), or None."""
    aperture = surface.aperture
    r_max = getattr(aperture, "r_max", None) if aperture is not None else None
    if r_max is not None:
        return ClearAperture(float(r_max))
    if extras.square_half_widths_mm is not None:
        return ClearAperture(min(extras.square_half_widths_mm), extras.square_half_widths_mm)
    return None


def _drawn_semi_aperture(extras: list[SurfaceExtras]) -> float:
    """The largest semi-diameter (DIAM) of a lens's surfaces, or the default if none is given."""
    semi_diameters = [
        surface.semi_diameter_mm for surface in extras if surface.semi_diameter_mm is not None
    ]
    return max(semi_diameters, default=0.0) or DEFAULT_SEMI_APERTURE_MM


def _note_rectangular(
    lens_name: str, apertures: list[ClearAperture | None], notes: list[str]
) -> None:
    """Tell the user when a rectangular aperture was replaced by the circle inside it."""
    rectangular = [
        aperture for aperture in apertures if aperture and aperture.square_half_widths_mm
    ]
    if rectangular:
        notes.append(
            f"{lens_name} has {rectangular[0].describe()}; RayCrafter apertures are round, "
            f"so the {2 * rectangular[0].semi_aperture_mm:g} mm circle inside it was used."
        )


def _paraxial_lens(
    surface,
    z: float,
    aperture: ClearAperture | None,
    extras: list[SurfaceExtras],
    notes: list[str],
) -> PlacedLens:
    """A paraxial lens at ``z``, sized by its aperture or else by its drawn semi-diameter."""
    focal_length_mm = float(surface.interaction_model.f)
    name = f"Paraxial f={focal_length_mm:+g} mm"
    _note_rectangular(name, [aperture], notes)
    return PlacedLens(
        part_number=name,
        surfaces=[],
        semi_aperture=aperture.semi_aperture_mm if aperture else _drawn_semi_aperture(extras),
        z=z,
        is_negative=focal_length_mm < 0,
        paraxial_focal_length_mm=focal_length_mm,
    )


def _imported_lens(
    surfaces: list[LensSurface],
    apertures: list[ClearAperture | None],
    extras: list[SurfaceExtras],
    front_z: float,
    lens_number: int,
    notes: list[str],
) -> PlacedLens:
    """A lens that is not in the catalogue, with the file's prescription and aperture."""
    name = f"Imported lens {lens_number}"
    known_apertures = [aperture.semi_aperture_mm for aperture in apertures if aperture]
    _note_rectangular(name, apertures, notes)
    return PlacedLens(
        part_number=name,
        surfaces=surfaces,
        semi_aperture=min(known_apertures) if known_apertures else _drawn_semi_aperture(extras),
        z=front_z,
    )


def _is_diverging(lens: PlacedLens) -> bool:
    """True if the lens on its own has a negative focal length.

    A block with no power (a flat plate or a beam-splitter cube) is not diverging, although
    Optiland gives its focal length as minus infinity.
    """
    alone = OpticalState(lenses=[lens], image_z=lens.back_z + 10.0)
    focal_length_mm = float(build_optic(alone).paraxial.f2())
    return bool(np.isfinite(focal_length_mm) and focal_length_mm < 0)


# ------------------------------------------------------------------ catalogue matching


def _match_catalogue(
    surfaces: list[LensSurface],
    apertures: list[ClearAperture | None],
    front_z: float,
    catalog: dict[str, CatalogEntry],
) -> PlacedLens | None:
    """The catalogue lens with this prescription, in either orientation, or None.

    Only entries with the same curvatures (read cheaply from their ZMX text) are parsed and
    compared in full, since parsing every entry with Optiland would take seconds. The clear
    aperture must match too when the file gives one.
    """
    curvature_key = _curvature_key([surface.radius for surface in surfaces])
    known_apertures = [aperture.semi_aperture_mm for aperture in apertures if aperture]
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
                    vendor=entry.vendor,
                    mount_type=entry.mount_type,
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


# ------------------------------------------------------------- apertures and rays


@dataclass
class RayExtent:
    """How far from the axis the file's own rays reach at each surface (mm).

    Only rays that arrive at a surface count there; a ray blocked further up is left out.
    Each array has one entry per surface of the file.
    """

    radius: np.ndarray
    abs_x: np.ndarray
    abs_y: np.ndarray


def _traced_ray_extent(optic) -> RayExtent | None:
    """Trace every field and wavelength of the file in Optiland, and record the ray extent.

    This is the file's own system, so the rays go where Zemax sends them (except that
    Optiland does not block rays at rectangular apertures). Returns None if Optiland cannot
    trace the system, in which case apertures are not checked.
    """
    surface_count = len(optic.surfaces.surfaces)
    extent = RayExtent(*(np.zeros(surface_count) for _ in range(3)))
    max_field = float(optic.fields.max_field) or 1.0  # Optiland's fields are relative to it
    try:
        for field_point in optic.fields.fields:
            for wavelength in optic.wavelengths.wavelengths:
                optic.trace(
                    float(field_point.x) / max_field,
                    float(field_point.y) / max_field,
                    float(wavelength.value),
                    RAY_CHECK_RINGS,
                    "hexapolar",
                )
                x = np.abs(np.asarray(optic.surfaces.x, dtype=float))
                y = np.abs(np.asarray(optic.surfaces.y, dtype=float))
                intensity = np.asarray(optic.surfaces.intensity, dtype=float)
                # A ray arrives at a surface if it got through the one before.
                arrived = np.ones_like(intensity, dtype=bool)
                arrived[1:] = intensity[:-1] > 0
                extent.radius = np.maximum(
                    extent.radius, _largest_arriving(np.hypot(x, y), arrived)
                )
                extent.abs_x = np.maximum(extent.abs_x, _largest_arriving(x, arrived))
                extent.abs_y = np.maximum(extent.abs_y, _largest_arriving(y, arrived))
    except Exception:  # Optiland cannot trace every system it can read
        return None
    return extent


def _largest_arriving(values: np.ndarray, arrived: np.ndarray) -> np.ndarray:
    """Per surface (row), the largest value among the rays that arrived there."""
    return np.nanmax(np.where(arrived, values, 0.0), axis=1)


def _note_lenses_that_block_more_than_zemax(
    grouping: SurfaceGrouping, ray_extent: RayExtent | None, notes: list[str]
) -> None:
    """Tell the user where RayCrafter blocks rays that Zemax lets through.

    Zemax blocks rays only at surfaces with an aperture; a lens with none in the file is
    just drawn at its semi-diameter. RayCrafter blocks rays at every lens rim, and keeps the
    lens's size as it is: its catalogue size, or the size the file draws it. That size is
    often the real one (e.g. an objective's back aperture), so it is not enlarged to match
    Zemax; the difference is reported instead.
    """
    for lens, numbers, file_has_aperture in zip(
        grouping.lenses, grouping.lens_surface_numbers, grouping.lens_has_file_aperture
    ):
        if file_has_aperture:
            continue  # Zemax blocks rays at the file's aperture too
        if ray_extent is not None:
            widest_ray_mm = float(max(ray_extent.radius[number] for number in numbers))
            if widest_ray_mm <= lens.semi_aperture:
                continue  # no difference from Zemax
            effect = (
                f"some of the file's rays reach {widest_ray_mm:.1f} mm from the axis there, "
                "so RayCrafter blocks rays that Zemax lets through"
            )
        else:
            effect = "this may block rays that Zemax lets through"
        is_catalogue_lens = lens.vendor is not None
        size = "its catalogue size" if is_catalogue_lens else "the size the file draws it"
        notes.append(
            f"{lens.part_number} has no aperture in the file, so Zemax blocks no rays there. "
            f"RayCrafter blocks rays outside {size} ({2 * lens.semi_aperture:g} mm across); "
            f"{effect}."
        )


def _note_apertures_in_air(
    air_apertures: list[AirAperture], ray_extent: RayExtent | None, notes: list[str]
) -> None:
    """Tell the user about each aperture in air that was left out, and whether it matters."""
    for air_aperture in air_apertures:
        number, aperture = air_aperture.surface_number, air_aperture.aperture
        if ray_extent is None:
            effect = "whether it blocks any of the file's rays was not checked"
        elif _blocks_rays(aperture, ray_extent, number):
            effect = "it blocks some of the file's rays, which RayCrafter lets through"
        else:
            effect = "none of the file's rays reach its edge"
        label = f' ("{air_aperture.comment}")' if air_aperture.comment else ""
        notes.append(
            f"Surface {number}{label} has {aperture.describe()} in air. RayCrafter cannot "
            f"model that, so it was left out; {effect}."
        )


def _blocks_rays(aperture: ClearAperture, ray_extent: RayExtent, surface_number: int) -> bool:
    """True if some traced ray reaches the surface outside the aperture."""
    if aperture.square_half_widths_mm is None:
        return bool(ray_extent.radius[surface_number] > aperture.semi_aperture_mm)
    half_x, half_y = aperture.square_half_widths_mm
    return bool(
        ray_extent.abs_x[surface_number] > half_x or ray_extent.abs_y[surface_number] > half_y
    )


# --------------------------------------------------------------------------- fields


def _fields_from_optic(
    optic, first_lens_z: float, first_lens_surface: int | None, notes: list[str]
) -> list[BeamField]:
    """RayCrafter fields from the file's aperture, object, stop, field angles and wavelengths.

    Each beam is described where it meets the first lens (see ``BeamField``):

    - its vergence is that of light from the object, ``first_lens_z - object_z`` away;
    - if the stop is in air in front of the first lens, the chief ray of a beam at angle a
      crosses the axis at the stop, so it meets the first lens ``d tan(a)`` above the axis,
      d being the distance from stop to lens. The entrance pupil diameter is the beam's
      width at the stop, so a beam from a near object is wider by the time it reaches the
      lens: by (s + d) / s, s being the distance from object to stop.

    Parameters
    ----------
    optic : optiland.optic.Optic
        The system read from the file.
    first_lens_z : float
        z (mm) of the first lens's front vertex on the bench.
    first_lens_surface : int or None
        The file surface number of the first lens's front surface (None with no lenses).
    notes : list of str
        Notes for the user are appended here.
    """
    surfaces = optic.surfaces.surfaces
    object_distance_mm = float(surfaces[0].thickness)
    object_z = FIRST_SURFACE_Z - object_distance_mm  # -inf for an object at infinity
    stop_number = int(optic.surfaces.stop_index)
    stop_z = FIRST_SURFACE_Z + sum(float(surface.thickness) for surface in surfaces[1:stop_number])
    stop_is_in_front = first_lens_surface is not None and stop_number < first_lens_surface
    stop_to_lens_mm = first_lens_z - stop_z if stop_is_in_front else 0.0

    aperture = optic.aperture
    if aperture is not None and aperture.ap_type == "EPD":
        diameter_mm = float(aperture.value)
        if stop_is_in_front and np.isfinite(object_z) and stop_z != object_z:
            object_to_stop_mm = stop_z - object_z
            diameter_mm *= abs((object_to_stop_mm + stop_to_lens_mm) / object_to_stop_mm)
    else:
        diameter_mm = DEFAULT_BEAM_DIAMETER_MM
        notes.append(
            f"The file's aperture is not an entrance pupil diameter; the beam was set to "
            f"{DEFAULT_BEAM_DIAMETER_MM:g} mm."
        )

    object_to_lens_mm = first_lens_z - object_z
    vergence = 0.0 if np.isinf(object_to_lens_mm) else -1000.0 / object_to_lens_mm

    field_type = getattr(optic.fields.field_definition, "name", str(optic.fields.field_definition))
    angles = [float(field_point.y) for field_point in optic.fields.fields] or [0.0]
    if "angle" not in field_type.lower():
        notes.append("The file's fields are not angles; one on-axis field was used instead.")
        angles = [0.0]
    if any(float(field_point.x) != 0.0 for field_point in optic.fields.fields):
        notes.append("Field x coordinates were dropped: RayCrafter beams tilt in y only.")
    angles = list(dict.fromkeys(angles))  # each angle once, in the file's order
    if first_lens_surface is not None and stop_number > first_lens_surface and any(angles):
        notes.append(
            f"The stop (surface {stop_number}) is not in front of the first lens, so each "
            "beam was centred on the first lens rather than aimed through the stop."
        )

    # Several angles are told apart by colour, as in Zemax; a single one keeps the colour
    # of its wavelength.
    angle_colours = {
        angle: palette.field_colour(number) if len(angles) > 1 else None
        for number, angle in enumerate(angles)
    }

    def beam(angle_deg: float, wavelength_um: float) -> BeamField:
        return BeamField(
            diameter_mm=diameter_mm,
            angle_deg=angle_deg,
            vergence_dioptres=vergence,
            offset_mm=stop_to_lens_mm * float(np.tan(np.radians(angle_deg))),
            wavelength_nm=round(wavelength_um * 1000.0, 3),
            colour=angle_colours.get(angle_deg),
        )

    wavelengths_um = [float(wavelength.value) for wavelength in optic.wavelengths.wavelengths]
    primary_um = float(optic.wavelengths.primary_wavelength.value)
    fields = [beam(angle, primary_um) for angle in angles]
    fields += [beam(0.0, wavelength) for wavelength in wavelengths_um if wavelength != primary_um]
    if len(angles) > 1 and len(wavelengths_um) > 1:
        notes.append(
            "Each field angle was loaded at the primary wavelength, and each other wavelength "
            "as an on-axis field."
        )
    return fields
