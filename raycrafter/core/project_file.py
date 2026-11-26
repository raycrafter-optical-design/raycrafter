"""Save and load a RayCrafter project file (``.raycrafter``): the whole design, as JSON.

A project file holds everything a ZMX file cannot: every field, Plossl pairs, the part
metadata of each lens, and (for a design imported from a ZMX file) the original file. The
format is described, with an example, in issue #40.

Every key is named here, separately from the attribute names in ``raycrafter.core.state``,
so that renaming something in the code does not change the file format. Loading is strict:
a missing or unknown key is refused and named, so that saving again cannot silently drop
data. Each lens's full prescription is stored, so a later catalogue update cannot change a
saved design.

``FORMAT_VERSION`` is raised whenever the file gains data or changes meaning. An older file
is upgraded one version at a time (``_UPGRADES``) before it is read; a newer one is refused.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from raycrafter.core.optic_builder import is_diverging
from raycrafter.core.prescription import LensSurface
from raycrafter.core.state import (
    BeamField,
    ImportedSource,
    OpticalState,
    PlacedLens,
    raycrafter_version,
)

FILE_EXTENSION = ".raycrafter"
FORMAT_NAME = "raycrafter-project"
FORMAT_VERSION = 1

# Format version -> function that turns a file of that version into the next version's
# layout. Empty until version 2 exists.
_UPGRADES: dict[int, Callable[[dict], dict]] = {}

TOP_LEVEL_KEYS = [
    "format",
    "format_version",
    "written_by",
    "lenses",
    "next_group",
    "fields",
    "selected_field",
    "image_z_mm",
    "imported_from",
]
PART_KEYS = ["vendor", "part_number", "mount_type", "clear_diameter_mm"]
GLASS_LENS_KEYS = ["type", "part", "front_z_mm", "flipped", "group", "surfaces"]
PARAXIAL_LENS_KEYS = ["type", "part", "front_z_mm", "focal_length_mm"]
SURFACE_KEYS = ["radius_mm", "thickness_after_mm", "material_after"]
FIELD_KEYS = [
    "diameter_mm",
    "angle_deg",
    "vergence_dioptres",
    "offset_mm",
    "wavelength_nm",
    "colour",
]
SOURCE_KEYS = ["format", "file_name", "sha256", "imported_by", "notes", "content_base64"]


class ProjectFileError(ValueError):
    """The file is not a RayCrafter project RayCrafter can read; the message says why."""


# ------------------------------------------------------------------------------ saving


def save_project(state: OpticalState, file_path: str | Path) -> None:
    """Write the design to a project file.

    The file is written to a temporary file in the same folder and then moved into place,
    so a failure part-way through leaves any earlier file intact.

    Parameters
    ----------
    state : OpticalState
        The design to save.
    file_path : str or Path
        Where to write it; conventionally ending in ``FILE_EXTENSION``.
    """
    # allow_nan=False makes json refuse inf and NaN; flat surfaces are written as null.
    text = json.dumps(project_to_dict(state), indent=2, ensure_ascii=False, allow_nan=False)
    target = Path(file_path)
    file_descriptor, temp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temp_file:
            temp_file.write(text + "\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def project_to_dict(state: OpticalState) -> dict:
    """The design as the dictionary that ``save_project`` writes as JSON."""
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "written_by": raycrafter_version(),
        "lenses": [_lens_to_dict(lens) for lens in state.lenses],
        "next_group": state.next_group_id,
        "fields": [_field_to_dict(beam) for beam in state.fields],
        "selected_field": state.selected_field_index,
        "image_z_mm": state.image_z,
        "imported_from": _source_to_dict(state.imported_from),
    }


def _lens_to_dict(lens: PlacedLens) -> dict:
    part = {
        "vendor": lens.vendor,
        "part_number": lens.part_number,
        "mount_type": lens.mount_type,
        "clear_diameter_mm": 2.0 * lens.semi_aperture,
    }
    if lens.paraxial_focal_length_mm is not None:
        return {
            "type": "paraxial",
            "part": part,
            "front_z_mm": lens.z,
            "focal_length_mm": lens.paraxial_focal_length_mm,
        }
    return {
        "type": "lens",
        "part": part,
        "front_z_mm": lens.z,
        "flipped": lens.flipped,
        "group": lens.group_id,
        # The catalogue orientation; ``flipped`` is applied when the lens is used.
        "surfaces": [
            {
                "radius_mm": None if math.isinf(surface.radius) else surface.radius,
                "thickness_after_mm": surface.thickness_after,
                "material_after": surface.material_after,
            }
            for surface in lens.surfaces
        ],
    }


def _field_to_dict(beam: BeamField) -> dict:
    return {
        "diameter_mm": beam.diameter_mm,
        "angle_deg": beam.angle_deg,
        "vergence_dioptres": beam.vergence_dioptres,
        "offset_mm": beam.offset_mm,
        "wavelength_nm": beam.wavelength_nm,
        "colour": beam.colour,
    }


def _source_to_dict(source: ImportedSource | None) -> dict | None:
    if source is None:
        return None
    return {
        "format": source.file_format,
        "file_name": source.file_name,
        "sha256": source.sha256,
        "imported_by": source.imported_by,
        "notes": list(source.notes),
        "content_base64": source.content_base64,
    }


# ----------------------------------------------------------------------------- loading


def load_project(file_path: str | Path) -> OpticalState:
    """Read a project file.

    Parameters
    ----------
    file_path : str or Path
        The project file.

    Returns
    -------
    OpticalState
        The saved design.

    Raises
    ------
    ProjectFileError
        If the file cannot be read, is not a RayCrafter project, was written by a newer
        RayCrafter, or holds missing, unknown or invalid data.
    """
    try:
        text = Path(file_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ProjectFileError(f"The file could not be read: {error}") from error
    try:
        # parse_constant refuses Infinity and NaN, which RayCrafter never writes.
        data = json.loads(text, parse_constant=_refuse_constant)
    except json.JSONDecodeError as error:
        raise ProjectFileError(f"The file is not valid JSON: {error}") from error
    return project_from_dict(data)


def project_from_dict(data: object) -> OpticalState:
    """Rebuild the design from the dictionary that ``project_to_dict`` makes.

    Raises
    ------
    ProjectFileError
        As for ``load_project``.
    """
    if not isinstance(data, dict) or data.get("format") != FORMAT_NAME:
        raise ProjectFileError("This is not a RayCrafter project file.")
    data = _upgraded(data)
    _check_keys(data, TOP_LEVEL_KEYS, "the project")

    lenses = [
        _lens_from_dict(lens_data, f"lens {index}")
        for index, lens_data in enumerate(_list(data["lenses"], "lenses"))
    ]
    front_positions = [lens.z for lens in lenses]
    if front_positions != sorted(front_positions):
        raise ProjectFileError("The lenses must be listed in order along the axis.")

    fields = [
        _field_from_dict(field_data, f"field {index}")
        for index, field_data in enumerate(_list(data["fields"], "fields"))
    ]
    if not fields:
        raise ProjectFileError("The project has no fields; there must be at least one.")
    selected_field = _integer(data["selected_field"], "selected_field")
    if not 0 <= selected_field < len(fields):
        raise ProjectFileError(
            f"selected_field is {selected_field}, but there are {len(fields)} field(s)."
        )

    return OpticalState(
        lenses=lenses,
        next_group_id=_integer(data["next_group"], "next_group"),
        fields=fields,
        selected_field_index=selected_field,
        image_z=_number(data["image_z_mm"], "image_z_mm"),
        imported_from=_source_from_dict(data["imported_from"]),
    )


def _upgraded(data: dict) -> dict:
    """The file's data in the current version's layout, upgrading it one step at a time."""
    version = data.get("format_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ProjectFileError(f"The project's format_version ({version!r}) is not valid.")
    if version > FORMAT_VERSION:
        raise ProjectFileError(
            f"This project was written by a newer RayCrafter ({data.get('written_by')}, "
            f"format version {version}). Update RayCrafter to open it."
        )
    while version < FORMAT_VERSION:
        data = _UPGRADES[version](data)
        version += 1
        data["format_version"] = version
    return data


def _lens_from_dict(data: object, where: str) -> PlacedLens:
    if not isinstance(data, dict):
        raise ProjectFileError(f"{where} is not a JSON object.")
    lens_type = data.get("type")
    if lens_type == "paraxial":
        _check_keys(data, PARAXIAL_LENS_KEYS, where)
    elif lens_type == "lens":
        _check_keys(data, GLASS_LENS_KEYS, where)
    else:
        raise ProjectFileError(f'{where} has type {lens_type!r}; expected "lens" or "paraxial".')

    part = data["part"]
    if not isinstance(part, dict):
        raise ProjectFileError(f"The part of {where} is not a JSON object.")
    _check_keys(part, PART_KEYS, f"the part of {where}")
    clear_diameter_mm = _number(part["clear_diameter_mm"], f"{where} clear_diameter_mm")
    if clear_diameter_mm <= 0:
        raise ProjectFileError(f"{where} has a clear diameter of {clear_diameter_mm} mm.")

    lens = PlacedLens(
        part_number=_text(part["part_number"], f"{where} part_number"),
        surfaces=[],
        semi_aperture=clear_diameter_mm / 2.0,
        z=_number(data["front_z_mm"], f"{where} front_z_mm"),
        vendor=_optional_text(part["vendor"], f"{where} vendor"),
        mount_type=_optional_text(part["mount_type"], f"{where} mount_type"),
    )
    if lens_type == "paraxial":
        focal_length_mm = _number(data["focal_length_mm"], f"{where} focal_length_mm")
        if focal_length_mm == 0:
            raise ProjectFileError(f"{where} is a paraxial lens with zero focal length.")
        lens.paraxial_focal_length_mm = focal_length_mm
    else:
        surfaces = _list(data["surfaces"], f"{where} surfaces")
        if len(surfaces) < 2:
            raise ProjectFileError(f"{where} has {len(surfaces)} surface(s); a lens needs 2.")
        lens.surfaces = [
            _surface_from_dict(surface, f"{where} surface {number}")
            for number, surface in enumerate(surfaces)
        ]
        lens.flipped = _boolean(data["flipped"], f"{where} flipped")
        group = data["group"]
        lens.group_id = None if group is None else _integer(group, f"{where} group")
    # Not stored, as it follows from the prescription.
    lens.is_negative = is_diverging(lens)
    return lens


def _surface_from_dict(data: object, where: str) -> LensSurface:
    if not isinstance(data, dict):
        raise ProjectFileError(f"{where} is not a JSON object.")
    _check_keys(data, SURFACE_KEYS, where)
    radius = data["radius_mm"]
    return LensSurface(
        radius=math.inf if radius is None else _number(radius, f"{where} radius_mm"),
        thickness_after=_number(data["thickness_after_mm"], f"{where} thickness_after_mm"),
        material_after=_text(data["material_after"], f"{where} material_after"),
    )


def _field_from_dict(data: object, where: str) -> BeamField:
    if not isinstance(data, dict):
        raise ProjectFileError(f"{where} is not a JSON object.")
    _check_keys(data, FIELD_KEYS, where)
    diameter_mm = _number(data["diameter_mm"], f"{where} diameter_mm")
    wavelength_nm = _number(data["wavelength_nm"], f"{where} wavelength_nm")
    if diameter_mm <= 0 or wavelength_nm <= 0:
        raise ProjectFileError(f"{where} needs a positive diameter and wavelength.")
    return BeamField(
        diameter_mm=diameter_mm,
        angle_deg=_number(data["angle_deg"], f"{where} angle_deg"),
        vergence_dioptres=_number(data["vergence_dioptres"], f"{where} vergence_dioptres"),
        offset_mm=_number(data["offset_mm"], f"{where} offset_mm"),
        wavelength_nm=wavelength_nm,
        colour=_optional_text(data["colour"], f"{where} colour"),
    )


def _source_from_dict(data: object) -> ImportedSource | None:
    if data is None:
        return None
    where = "imported_from"
    if not isinstance(data, dict):
        raise ProjectFileError(f"{where} is not a JSON object.")
    _check_keys(data, SOURCE_KEYS, where)
    source = ImportedSource(
        file_format=_text(data["format"], f"{where} format"),
        file_name=_text(data["file_name"], f"{where} file_name"),
        content_base64=_text(data["content_base64"], f"{where} content_base64"),
        sha256=_text(data["sha256"], f"{where} sha256"),
        imported_by=_text(data["imported_by"], f"{where} imported_by"),
        notes=tuple(
            _text(note, f"{where} note") for note in _list(data["notes"], f"{where} notes")
        ),
    )
    # A damaged copy is refused rather than kept, as it could not be imported again.
    try:
        content = source.content()
    except ValueError as error:
        raise ProjectFileError(f"The imported file's content is damaged: {error}") from error
    if hashlib.sha256(content).hexdigest() != source.sha256:
        raise ProjectFileError(
            f"The copy of {source.file_name} in the project does not match its checksum."
        )
    return source


# -------------------------------------------------------------- checking each value


def _check_keys(data: dict, expected: list[str], where: str) -> None:
    """Refuse a missing or unknown key, naming it."""
    missing = [key for key in expected if key not in data]
    unknown = [key for key in data if key not in expected]
    problems = []
    if missing:
        problems.append("is missing " + ", ".join(missing))
    if unknown:
        problems.append("has unknown " + ", ".join(unknown))
    if problems:
        raise ProjectFileError(f"{where[0].upper()}{where[1:]} {' and '.join(problems)}.")


def _number(value: object, where: str) -> float:
    # bool is a subclass of int in Python, but true and false are not numbers in a project.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProjectFileError(f"{where} should be a number, not {value!r}.")
    return float(value)


def _integer(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProjectFileError(f"{where} should be a whole number, not {value!r}.")
    return value


def _boolean(value: object, where: str) -> bool:
    if not isinstance(value, bool):
        raise ProjectFileError(f"{where} should be true or false, not {value!r}.")
    return value


def _text(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise ProjectFileError(f"{where} should be text, not {value!r}.")
    return value


def _optional_text(value: object, where: str) -> str | None:
    return None if value is None else _text(value, where)


def _list(value: object, where: str) -> list:
    if not isinstance(value, list):
        raise ProjectFileError(f"{where} should be a list, not {value!r}.")
    return value


def _refuse_constant(name: str) -> float:
    raise ProjectFileError(f"The file contains {name}, which is not allowed in a project.")
