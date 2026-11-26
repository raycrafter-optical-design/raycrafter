"""The ``Bench`` methods that save, open, import and export designs.

``Bench`` (``raycrafter.api``) inherits these, so a script calls them on a bench:

>>> from raycrafter import Bench
>>> bench = Bench()
>>> bench.save_project("my_design.raycrafter")  # doctest: +SKIP
>>> bench.save_zmx("my_design.zmx")  # doctest: +SKIP

A design is saved as a RayCrafter project (``.raycrafter``), which keeps everything: all the
fields, assemblies, locked gaps, the part details of each lens and, for a design imported from a Zemax
file, that file and the design as it was imported. A Zemax ``.zmx`` file is for exchanging
designs with other programs: it keeps only the selected field and no assemblies, so
exporting one does not count as saving the design.

They are kept apart from ``raycrafter.api`` because they concern files rather than optics.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING


class DesignFileMethods:
    """Saving, opening, importing and exporting, for ``Bench`` to inherit.

    Every method goes through ``Bench._call``, so it works the same on a headless bench and
    on one connected to a RayCrafter window.
    """

    if TYPE_CHECKING:  # provided by Bench

        def _call(self, tool_name: str, **arguments) -> dict: ...

    @property
    def has_unsaved_changes(self) -> bool:
        """True if the design differs from the one last saved, opened, imported or started."""
        return self._call("get_design")["has_unsaved_changes"]

    @property
    def project_path(self) -> Path | None:
        """The project file last saved or opened, where ``save_project`` saves by default."""
        project_file = self._call("get_design")["project_file"]
        return None if project_file is None else Path(project_file)

    @property
    def imported_from(self) -> str | None:
        """The name of the Zemax file the design was imported from; None if not imported."""
        source = self._call("get_design")["imported_from"]
        return None if source is None else source["file_name"]

    @property
    def modified_since_import(self) -> bool | None:
        """True if the bench has been changed since it was imported; None if not imported."""
        source = self._call("get_design")["imported_from"]
        return None if source is None else source["modified_since_import"]

    def new_design(self, discard_unsaved: bool = False, example: str | None = None) -> None:
        """Start again with no lenses and one default field, or from an example design.

        ``example`` is the name of one of ``raycrafter.core.examples.EXAMPLES``, such as
        ``"Basic microscope"``.

        Raises
        ------
        ValueError
            If there are unsaved changes and ``discard_unsaved`` is False, or there is no
            example of that name.
        """
        self._call("new_design", discard_unsaved=discard_unsaved, example=example)

    def save_project(
        self, file_path: str | Path | None = None, forget_import: bool = False
    ) -> Path:
        """Save the whole design as a RayCrafter project file and return where it went.

        A project keeps everything (all fields, assemblies, part details, and the Zemax
        file a design was imported from). Without ``file_path`` it saves to
        ``project_path``; ``.raycrafter`` is added to a name with no extension.

        With ``forget_import=True`` the project no longer keeps the Zemax file the design
        was imported from, nor the design as imported: from then on it is a design of its
        own rather than an import (see ``imported_from``).

        Raises
        ------
        ValueError
            If no path is given and the design has no project file yet.

        Example
        -------
        >>> Bench().save_project("my_design.raycrafter")  # doctest: +SKIP
        """
        absolute_path = None if file_path is None else _absolute(file_path)
        result = self._call("save_project", file_path=absolute_path, forget_import=forget_import)
        return Path(result["saved_to"])

    def load_project(self, file_path: str | Path, discard_unsaved: bool = False) -> None:
        """Replace the design with the one in a RayCrafter project file.

        Raises
        ------
        ValueError
            If the file is not a valid project, or there are unsaved changes and
            ``discard_unsaved`` is False.
        """
        self._call("load_project", file_path=_absolute(file_path), discard_unsaved=discard_unsaved)

    def load_zmx(self, file_path: str | Path, discard_unsaved: bool = False) -> list[str]:
        """Replace the design with the one in a Zemax .zmx file (an import).

        The imported design has no project file until ``save_project`` is called.

        Returns
        -------
        list of str
            Notes on anything RayCrafter simplified or could not represent.

        Raises
        ------
        ValueError
            If the file cannot be read, or there are unsaved changes and ``discard_unsaved``
            is False.

        Example
        -------
        >>> notes = bench.load_zmx("my_design.zmx", discard_unsaved=True)  # doctest: +SKIP
        """
        result = self._call(
            "load_zmx", file_path=_absolute(file_path), discard_unsaved=discard_unsaved
        )
        return result["notes"]

    def save_zmx(self, file_path: str | Path) -> Path:
        """Export the design as a Zemax .zmx file and return where it went.

        This does not count as saving the design, as a ZMX file keeps only the selected
        field and no assemblies; use ``save_project`` for that.

        Example
        -------
        >>> Bench().save_zmx("my_design.zmx")  # doctest: +SKIP
        """
        absolute_path = _absolute(file_path)
        self._call("save_zmx", file_path=absolute_path)
        return Path(absolute_path)


def _absolute(file_path: str | Path) -> str:
    """The absolute form of a path, so a connected window (whose working folder differs from
    the script's) reads and writes the file the script means."""
    return str(Path(file_path).expanduser().resolve())
