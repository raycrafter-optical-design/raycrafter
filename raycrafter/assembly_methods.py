"""The ``Bench`` methods for assemblies (such as Plossl pairs) and user parts.

``Bench`` (``raycrafter.api``) inherits these, so a script calls them on a bench:

>>> from raycrafter import Bench
>>> bench = Bench(empty=True)
>>> for focal_length_mm in (100.0, 100.0):
...     _ = bench.add_paraxial_lens(focal_length_mm)
>>> bench.set_gap(0, 200.0).gap_after_mm  # a 4f relay
200.0
>>> bench.make_assembly(0, 1, name="4f relay")
>>> [lens.assembly_name for lens in bench.lenses]
['4f relay', '4f relay']
>>> bench.save_user_part(0, "4f relay")  # doctest: +SKIP

An assembly is a run of neighbouring lenses that moves and flips as one; the gaps inside it
are locked. A user part is an assembly saved to the user's parts folder, from which it can
be inserted into any design, as a catalogue lens can.

They are kept apart from ``raycrafter.api`` to keep that file a manageable size.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from raycrafter.api import LensInfo


class AssemblyMethods:
    """Making, splitting and flipping assemblies, and user parts, for ``Bench`` to inherit.

    Every method goes through ``Bench._call``, so it works the same on a headless bench and
    on one connected to a RayCrafter window.
    """

    if TYPE_CHECKING:  # provided by Bench

        def _call(self, tool_name: str, **arguments) -> dict: ...

        @property
        def lenses(self) -> list[LensInfo]: ...

    def make_plossl(self, lens_index: int) -> None:
        """Turn one lens into a Plossl pair: two copies of it facing each other.

        The pair is centred on the optical centre of the original lens, with the positive
        elements (of a singlet, the more strongly curved faces) inside and 1 mm apart. A
        locked gap either side keeps its length, so what is joined to it moves outwards.
        The pair is an assembly called "Plossl pair", at indices ``lens_index`` and
        ``lens_index + 1``; ``split_assembly`` turns it back into two separate lenses.

        Raises
        ------
        ValueError
            If the lens is paraxial, diverging or already in an assembly, or the pair would
            overlap a neighbour.

        Example
        -------
        >>> from raycrafter import Bench
        >>> bench = Bench(empty=True)
        >>> _ = bench.add_lens("AC254-050-A")
        >>> bench.make_plossl(0)
        >>> [lens.assembly_name for lens in bench.lenses]
        ['Plossl pair', 'Plossl pair']
        """
        self._call("make_plossl", lens_index=int(lens_index))

    def make_assembly(
        self, first_lens_index: int, last_lens_index: int, name: str | None = None
    ) -> None:
        """Group lenses ``first_lens_index`` to ``last_lens_index`` into an assembly.

        The gaps between them are locked at their current lengths, so they move as one.
        Assemblies wholly inside the run are absorbed into the new one. Without a name it is
        called "Assembly <number>".

        Raises
        ------
        ValueError
            If the run has fewer than two lenses, or would cut an assembly in two.
        """
        self._call(
            "make_assembly",
            first_lens_index=int(first_lens_index),
            last_lens_index=int(last_lens_index),
            name=name,
        )

    def split_assembly(self, lens_index: int) -> None:
        """Dissolve the assembly this lens is in; its lenses stay where they are, unlocked."""
        self._call("split_assembly", lens_index=int(lens_index))

    def flip_assembly(self, lens_index: int) -> None:
        """Turn this lens's assembly round in place: lenses in reverse order, each flipped.

        The indices of the lenses inside it are reversed. Flipping twice restores it.

        Raises
        ------
        ValueError
            If the lens is in no assembly (use ``flip_lens``).
        """
        self._call("flip_assembly", lens_index=int(lens_index))

    def save_user_part(self, lens_index: int, name: str, overwrite: bool = False) -> Path:
        """Save this lens's assembly as a user part called ``name``; return the file.

        Parts are kept in the user's parts folder (see
        ``raycrafter.core.user_parts.default_parts_folder``) and can be inserted in any
        design with ``insert_user_part``.

        Raises
        ------
        ValueError
            If the lens is in no assembly, or a part of that name exists and ``overwrite``
            is False.
        """
        result = self._call(
            "save_user_part", lens_index=int(lens_index), name=name, overwrite=bool(overwrite)
        )
        return Path(result["file"])

    def user_parts(self) -> list[str]:
        """The names of the user parts that can be inserted, sorted."""
        return [part["name"] for part in self._call("list_user_parts")["user_parts"]]

    def insert_user_part(self, name: str, front_z_mm: float) -> list[LensInfo]:
        """Place a user part, as one assembly, with its front as near ``front_z_mm`` as fits.

        Returns
        -------
        list of LensInfo
            Snapshots of the new lenses.

        Raises
        ------
        ValueError
            If there is no such part, or it fits nowhere.
        """
        result = self._call("insert_user_part", name=name, front_z_mm=float(front_z_mm))
        lenses = self.lenses
        return [lenses[index] for index in result["new_lens_indices"]]
