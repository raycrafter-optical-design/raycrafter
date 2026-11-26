"""User parts: assemblies saved by the user, to be inserted again like a catalogue lens.

A user part is a named run of lenses with their spacings and orientations, saved from an
assembly on the bench (``raycrafter.core.assemblies``). Each part is a small JSON file
(``PART_EXTENSION``) in the user's parts folder (``default_parts_folder``), so it is
available in every design. Its lenses are stored as in a project file
(``raycrafter.core.project_file.lens_to_dict``), with the first lens's front vertex at
z = 0. It has no Qt code.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from raycrafter.core.project_file import ProjectFileError, lens_from_dict, lens_to_dict
from raycrafter.core.state import PlacedLens, raycrafter_version

PART_EXTENSION = ".raycrafter-part"
FORMAT_NAME = "raycrafter-part"
FORMAT_VERSION = 1
PART_FILE_KEYS = ["format", "format_version", "written_by", "name", "lenses"]


class UserPartError(ValueError):
    """A user part cannot be saved or read; the message says why."""


@dataclass
class UserPart:
    """A saved assembly: its name and its lenses.

    Attributes
    ----------
    name : str
        What the user called it, e.g. ``"4f relay"``.
    lenses : list of PlacedLens
        In order along the axis, the first front vertex at z = 0. They are in no assembly
        and have no locks; inserting the part makes them an assembly.
    file_path : Path or None
        The file the part was read from or saved to.
    """

    name: str
    lenses: list[PlacedLens] = field(default_factory=list)
    file_path: Path | None = None

    @property
    def length_mm(self) -> float:
        """From the first front vertex to the last back vertex (mm)."""
        return self.lenses[-1].back_z - self.lenses[0].z

    def lenses_at(self, front_z: float) -> list[PlacedLens]:
        """Fresh copies of the lenses, with the first front vertex at ``front_z`` (mm)."""
        placed = copy.deepcopy(self.lenses)
        for lens in placed:
            lens.z += front_z
        return placed


def default_parts_folder() -> Path:
    """The folder user parts are kept in, in the usual place for application data.

    ``~/Library/Application Support/RayCrafter/parts`` on macOS, ``%APPDATA%/RayCrafter/parts``
    on Windows and ``$XDG_DATA_HOME/raycrafter/parts`` (by default
    ``~/.local/share/raycrafter/parts``) elsewhere. It may not exist yet.
    """
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "RayCrafter" / "parts"
    if sys.platform == "win32":
        app_data = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(app_data) / "RayCrafter" / "parts"
    data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data_home) / "raycrafter" / "parts"


def part_from_lenses(name: str, lenses: list[PlacedLens]) -> UserPart:
    """A part made from copies of these lenses, moved so the first front vertex is at 0.

    Raises
    ------
    UserPartError
        If the name is blank or there are no lenses.
    """
    name = name.strip()
    if not name:
        raise UserPartError("A user part needs a name")
    if not lenses:
        raise UserPartError("A user part needs at least one lens")
    copies = copy.deepcopy(lenses)
    start_z = copies[0].z
    for lens in copies:
        lens.z -= start_z
        lens.assembly_id = None
        lens.locked_to_next = False
    return UserPart(name=name, lenses=copies)


def part_file_path(name: str, folder: Path) -> Path:
    """The file a part of this name is saved in: the name with awkward characters replaced."""
    stem = re.sub(r"[^\w .()+-]", "_", name.strip()).strip(" .") or "part"
    return folder / f"{stem}{PART_EXTENSION}"


def save_part(part: UserPart, folder: Path, overwrite: bool = False) -> Path:
    """Write a part to its file in ``folder`` (made if needed) and return the file's path.

    Raises
    ------
    UserPartError
        If a part of that name already exists and ``overwrite`` is False.
    """
    file_path = part_file_path(part.name, folder)
    if file_path.exists() and not overwrite:
        raise UserPartError(f"There is already a user part called {part.name!r}")
    data = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "written_by": raycrafter_version(),
        "name": part.name,
        "lenses": [lens_to_dict(lens, on_bench=False) for lens in part.lenses],
    }
    folder.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    part.file_path = file_path
    return file_path


def load_part(file_path: Path) -> UserPart:
    """Read one part file.

    Raises
    ------
    UserPartError
        If the file cannot be read or is not a valid part.
    """
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UserPartError(f"{file_path.name} could not be read: {error}") from error
    if not isinstance(data, dict) or data.get("format") != FORMAT_NAME:
        raise UserPartError(f"{file_path.name} is not a RayCrafter user part")
    if data.get("format_version") != FORMAT_VERSION:
        raise UserPartError(
            f"{file_path.name} has format version {data.get('format_version')!r}; this "
            f"RayCrafter reads version {FORMAT_VERSION}"
        )
    if sorted(data) != sorted(PART_FILE_KEYS):
        raise UserPartError(f"{file_path.name} should have exactly the keys {PART_FILE_KEYS}")
    lens_list = data["lenses"]
    if not isinstance(data["name"], str) or not isinstance(lens_list, list) or not lens_list:
        raise UserPartError(f"{file_path.name} needs a name and at least one lens")
    try:
        lenses = [
            lens_from_dict(lens_data, f"lens {index}", on_bench=False)
            for index, lens_data in enumerate(lens_list)
        ]
    except ProjectFileError as error:
        raise UserPartError(f"{file_path.name}: {error}") from error
    return UserPart(name=data["name"], lenses=lenses, file_path=file_path)


def load_parts(folder: Path) -> tuple[list[UserPart], list[str]]:
    """Every part in ``folder``, sorted by name, and a note on each file that could not be read.

    A folder that does not exist yet holds no parts.
    """
    parts, problems = [], []
    if folder.is_dir():
        for file_path in sorted(folder.glob(f"*{PART_EXTENSION}")):
            try:
                parts.append(load_part(file_path))
            except UserPartError as error:
                problems.append(str(error))
    parts.sort(key=lambda part: part.name.lower())
    return parts, problems
