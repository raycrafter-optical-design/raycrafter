"""The state of the lens bench: the lenses on the axis, the beams and the image plane.

All z positions are absolute positions along the optical axis, in mm. The beams come from
the left (negative z). These classes hold data only; editing is done by ``OpticalModel``
(``raycrafter.core.model``) and optics by ``raycrafter.core.optic_builder`` and
``raycrafter.core.analysis``. ``state_to_dict`` and ``state_from_dict`` copy a state between
processes (a script and the RayCrafter window).
"""

from __future__ import annotations

import base64
import copy
import hashlib
from dataclasses import asdict, dataclass, field, replace

import raycrafter
from raycrafter.core.prescription import LensSurface, flipped_prescription, lens_thickness
from raycrafter.palette import wavelength_colour

AXIS_START_Z = 0.0  # mm, where the beams start
DEFAULT_WAVELENGTH_NM = 555.0  # green, where the eye is most sensitive


@dataclass
class PlacedLens:
    """One lens on the optical axis: its prescription, position, orientation and size.

    A lens is also the "part" that is bought and mounted: a contiguous run of surfaces (see
    ``OpticalState.surface_range``) with a vendor, part number, mount type and clear
    aperture.

    Attributes
    ----------
    part_number : str
        Catalogue part number, e.g. ``"ACT508-200-A"``.
    surfaces : list of LensSurface
        The prescription in the catalogue orientation (see ``surfaces_in_use`` for the
        current one).
    semi_aperture : float
        Half the clear diameter, in mm.
    z : float
        z of the front vertex, in mm.
    flipped : bool
        True if the lens is turned around relative to the catalogue.
    assembly_id : int or None
        Lenses that share an id form an assembly (``OpticalState.assembly_names``): a run of
        neighbouring lenses that moves and flips as one, such as a Plossl pair. The gaps
        inside an assembly are always locked.
    locked_to_next : bool
        True if the air gap behind this lens, to the next lens, is locked: the two keep
        their spacing and move as one (see ``raycrafter.core.gaps``). Always False for the
        last lens; the gap behind it, to the image plane, is ``OpticalState.image_gap_locked``.
    is_negative : bool
        True for a diverging lens (negative focal length). These cannot form a Plossl pair.
    paraxial_focal_length_mm : float or None
        Set only for a paraxial lens: an ideal thin lens of this focal length, with no
        surfaces (``surfaces`` is empty) and zero thickness. It cannot be flipped or paired.
    vendor : str or None
        Who makes the lens, e.g. ``"Thorlabs"``; None if unknown (an imported or paraxial
        lens).
    mount_type : str or None
        How the lens is mounted, e.g. ``"Unmounted"``; None if unknown.
    """

    part_number: str
    surfaces: list[LensSurface]
    semi_aperture: float
    z: float
    flipped: bool = False
    assembly_id: int | None = None
    locked_to_next: bool = False
    is_negative: bool = False
    paraxial_focal_length_mm: float | None = None
    vendor: str | None = None
    mount_type: str | None = None

    @property
    def is_paraxial(self) -> bool:
        """True for an ideal thin (paraxial) lens rather than a catalogue lens."""
        return self.paraxial_focal_length_mm is not None

    @property
    def surface_count(self) -> int:
        """How many surfaces the lens adds to the optical system (1 for a paraxial lens)."""
        return 1 if self.is_paraxial else len(self.surfaces)

    def surfaces_in_use(self) -> list[LensSurface]:
        """The surfaces in the current orientation."""
        return flipped_prescription(self.surfaces) if self.flipped else self.surfaces

    def positive_element_is_first(self) -> bool:
        """True if, in the catalogue orientation, the front glass element is the stronger one.

        Each element's power is approximated by 1/R_front - 1/R_back (the glass index factor
        is positive for both, so it does not change which is larger). For a positive doublet
        the stronger element is the crown.
        """
        glass_surface_indices = [
            index
            for index, surface in enumerate(self.surfaces[:-1])
            if surface.material_after != "air"
        ]
        powers = []
        for index in (glass_surface_indices[0], glass_surface_indices[-1]):
            front, back = self.surfaces[index], self.surfaces[index + 1]
            powers.append(1.0 / front.radius - 1.0 / back.radius)
        return powers[0] > powers[1]

    @property
    def thickness(self) -> float:
        """Front-vertex to back-vertex distance (mm); zero for a paraxial lens."""
        return lens_thickness(self.surfaces)

    @property
    def back_z(self) -> float:
        """z of the back vertex (mm)."""
        return self.z + self.thickness


@dataclass
class BeamField:
    """One beam of light entering the bench from the left (a "field").

    The beam is defined where it crosses the source plane (``OpticalState.source_z``): there
    it is ``diameter_mm`` wide, its centre is ``offset_mm`` above the axis, and it travels
    at ``angle_deg`` with ``vergence_dioptres``. By default the source plane is at the first
    lens's front vertex and moves with it.

    Attributes
    ----------
    diameter_mm : float
        Beam diameter at the source plane.
    angle_deg : float
        Angle of the beam's central ray to the axis, in degrees (positive rises).
    vergence_dioptres : float
        Dioptres (1/m): positive is converging, negative diverging, zero collimated.
    offset_mm : float
        Height of the beam centre above the axis at the source plane.
    wavelength_nm : float
        Wavelength the beam is traced at.
    colour : str or None
        Hex colour the beam is drawn in; None for the colour of its wavelength.
    """

    diameter_mm: float = 10.0
    angle_deg: float = 0.0
    vergence_dioptres: float = 0.0
    offset_mm: float = 0.0
    wavelength_nm: float = DEFAULT_WAVELENGTH_NM
    colour: str | None = None

    @property
    def wavelength_um(self) -> float:
        """Wavelength in micrometres, as Optiland wants it."""
        return self.wavelength_nm / 1000.0

    def display_colour(self) -> str:
        """The colour to draw the beam in: its own, or else that of its wavelength."""
        return self.colour if self.colour is not None else wavelength_colour(self.wavelength_nm)


def raycrafter_version() -> str:
    """This program and its version, e.g. ``"RayCrafter 0.0.3"``, as recorded in files."""
    return f"RayCrafter {getattr(raycrafter, '__version__', 'unknown')}"


@dataclass(frozen=True)
class ImportedSource:
    """The file a design was imported from, kept as a record of where it came from.

    Importing is lossy and the importer keeps improving, so the original file is kept with
    the design (and in its project file): the design can then be imported again later, and
    import problems can be reproduced. It is never loaded in place of the design.

    Once the bench has been edited, the file no longer describes it. So the bench exactly as
    the importer made it is kept too (``imported_design``). Comparing the current bench with
    it tells whether the user has changed anything (``OpticalState.modified_since_import``),
    and comparing a fresh import of the file with it shows what a newer importer does
    differently.

    Attributes
    ----------
    file_format : str
        The kind of file, e.g. ``"zmx"``.
    file_name : str
        The file's name when it was imported, without its folder.
    content_base64 : str
        The file's exact bytes, base64-encoded (ZMX files may be UTF-16), so that the state
        stays plain JSON-ready data.
    sha256 : str
        Hex SHA-256 of the bytes, to check the copy is intact.
    imported_by : str
        The RayCrafter version that imported it, e.g. ``"RayCrafter 0.0.3"``.
    imported_design : OpticalState
        The bench as the importer made it, before any edits (its own ``imported_from`` is
        None).
    notes : tuple of str
        The notes on what the import simplified, as the user was shown them.
    """

    file_format: str
    file_name: str
    content_base64: str
    sha256: str
    imported_by: str
    imported_design: OpticalState
    notes: tuple[str, ...] = ()

    @classmethod
    def from_bytes(
        cls,
        file_format: str,
        file_name: str,
        content: bytes,
        imported_by: str,
        imported_design: OpticalState,
        notes: list[str] | tuple[str, ...] = (),
    ) -> ImportedSource:
        """Make the record from the file's bytes, encoding them and computing the hash.

        ``imported_design`` is copied, so later edits to the bench do not change it.
        """
        return cls(
            file_format=file_format,
            file_name=file_name,
            content_base64=base64.b64encode(content).decode("ascii"),
            sha256=hashlib.sha256(content).hexdigest(),
            imported_by=imported_by,
            imported_design=copy.deepcopy(replace(imported_design, imported_from=None)),
            notes=tuple(notes),
        )

    def content(self) -> bytes:
        """The file's original bytes."""
        return base64.b64decode(self.content_base64)


@dataclass
class OpticalState:
    """Everything the user can change: the lenses, the beams and the image plane.

    Attributes
    ----------
    lenses : list of PlacedLens
        Ordered left to right along the axis.
    next_assembly_id : int
        The id the next assembly will get.
    assembly_names : dict of int to str
        The name of each assembly, by id, e.g. ``{1: "Plossl pair"}``.
    fields : list of BeamField
        The beams entering the bench. There is always at least one.
    selected_field_index : int
        The field that autofocus and the single-field analyses (through-focus, MTF, ray
        fan, field curvature, f-number) use. The spot diagram and side view show every field.
    image_z : float
        z of the image plane, in mm.
    image_gap_locked : bool
        True if the gap from the last lens to the image plane is locked: the image plane then
        moves with the last lens, and cannot be moved on its own (e.g. by autofocus).
    source_z : float
        z of the source plane, where every field is defined (see ``BeamField``), in mm. It
        could be a light source, a scan mirror or a stop. It is never behind the first lens
        or in front of the start of the axis.
    source_gap_locked : bool
        True (the default) if the gap from the source plane to the first lens is locked:
        the source plane then stays that far in front of whichever lens is first, and moves
        with it. With the gap at 0 mm, the beams are defined at the first lens.
    imported_from : ImportedSource or None
        The file the design was imported from, if it was; editing leaves it unchanged.
    """

    lenses: list[PlacedLens] = field(default_factory=list)
    next_assembly_id: int = 1
    assembly_names: dict[int, str] = field(default_factory=dict)
    fields: list[BeamField] = field(default_factory=lambda: [BeamField()])
    selected_field_index: int = 0
    image_z: float = 310.0
    image_gap_locked: bool = False
    source_z: float = AXIS_START_Z
    source_gap_locked: bool = True
    imported_from: ImportedSource | None = None

    @property
    def selected_field(self) -> BeamField:
        """The field the single-field analyses use."""
        return self.fields[self.selected_field_index]

    def last_back_z(self) -> float:
        """z of the back vertex of the rightmost lens (0 if there are no lenses)."""
        return self.lenses[-1].back_z if self.lenses else AXIS_START_Z

    def surface_range(self, lens_index: int) -> tuple[int, int]:
        """The numbers of the first and last surface of one lens.

        Surfaces are numbered as in the optical system and the exported ZMX file: the object
        is surface 0 and the lens surfaces follow from 1, left to right.
        """
        first = 1 + sum(lens.surface_count for lens in self.lenses[:lens_index])
        return first, first + self.lenses[lens_index].surface_count - 1

    def modified_since_import(self) -> bool | None:
        """True if the bench differs from the one imported; None if it was not imported.

        Which field is selected does not count, as it does not for unsaved changes.
        """
        if self.imported_from is None:
            return None
        current = replace(self, imported_from=None, selected_field_index=0)
        return current != replace(self.imported_from.imported_design, selected_field_index=0)

    def beam_reference_z(self) -> float:
        """z of the plane where each field is defined (mm): the source plane."""
        return self.source_z

    def first_surface_z(self) -> float:
        """z of the optical system's first surface (mm), where Optiland's z is anchored.

        This is the front vertex of the first lens, or the start of the axis with no lenses
        (see ``optic_builder``).
        """
        return self.lenses[0].z if self.lenses else AXIS_START_Z


def state_to_dict(state: OpticalState) -> dict:
    """The whole state as nested dictionaries and lists, ready to send as JSON.

    A flat surface has an infinite radius. Python's ``json`` module writes and reads that as
    ``Infinity``, which is enough for RayCrafter talking to itself (``state_from_dict``).
    """
    return asdict(state)


def state_from_dict(data: dict) -> OpticalState:
    """Rebuild an ``OpticalState`` from ``state_to_dict`` output."""
    lenses = [
        PlacedLens(
            **{
                **lens_data,
                "surfaces": [LensSurface(**surface) for surface in lens_data["surfaces"]],
            }
        )
        for lens_data in data["lenses"]
    ]
    fields = [BeamField(**field_data) for field_data in data["fields"]]
    # JSON turns the assembly ids, which are dictionary keys, into text.
    assembly_names = {int(key): name for key, name in data["assembly_names"].items()}
    source_data = data.get("imported_from")
    imported_from = None
    if source_data is not None:
        imported_from = ImportedSource(
            **{
                **source_data,
                "imported_design": state_from_dict(source_data["imported_design"]),
                "notes": tuple(source_data["notes"]),
            }
        )
    return OpticalState(
        **{
            **data,
            "lenses": lenses,
            "fields": fields,
            "assembly_names": assembly_names,
            "imported_from": imported_from,
        },
    )
