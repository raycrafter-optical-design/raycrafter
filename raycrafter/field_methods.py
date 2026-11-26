"""The ``Bench`` methods for editing the fields: the light entering the bench.

``Bench`` (``raycrafter.api``) inherits these, so a script calls them on a bench. A field
is a beam (an object at infinity: a width, angle, vergence and offset at the source plane)
or, after ``set_finite_object``, a point of a finite object on the source plane (a height
and an NA). ``Bench.fields`` lists them. They are kept apart from ``raycrafter.api`` to keep
that file a manageable size.

>>> from raycrafter import Bench
>>> bench = Bench()
>>> bench.set_field(0, angle_deg=1.0).angle_deg
1.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from raycrafter.api import FieldInfo


class FieldMethods:
    """Adding, removing, selecting and changing fields, for ``Bench`` to inherit.

    Every method goes through ``Bench._call``, so it works the same on a headless bench and
    on one connected to a RayCrafter window.
    """

    if TYPE_CHECKING:  # provided by Bench

        def _call(self, tool_name: str, **arguments) -> dict: ...

        @property
        def fields(self) -> list[FieldInfo]: ...

        @property
        def selected_field_index(self) -> int: ...

    def set_field(
        self,
        field_index: int,
        diameter_mm: float | None = None,
        angle_deg: float | None = None,
        vergence_dioptres: float | None = None,
        offset_mm: float | None = None,
        wavelength_nm: float | None = None,
        colour: str | None = None,
        height_x_mm: float | None = None,
        height_y_mm: float | None = None,
        numerical_aperture: float | None = None,
    ) -> FieldInfo:
        """Change any of a field's properties; those left out stay as they are.

        Only a field's own kind of property can be set (see ``FieldInfo.kind``): the
        diameter, angle, vergence and offset of a beam, or the heights and NA of an object
        point. Both have a wavelength and colour.

        Parameters
        ----------
        field_index : int
            Which field (see ``fields``).
        diameter_mm : float, optional
            Beam diameter at the source plane (see ``source_gap_mm``); must be positive.
        angle_deg : float, optional
            Angle of the beam to the axis (positive rises).
        vergence_dioptres : float, optional
            Positive converging, negative diverging, 0 collimated.
        offset_mm : float, optional
            Height of the beam centre above the axis at the source plane.
        wavelength_nm : float, optional
            Must be positive.
        colour : str, optional
            A hex colour such as ``"#ff8800"``, or ``"wavelength"`` to go back to the colour
            of its wavelength.
        height_x_mm, height_y_mm : float, optional
            Where an object point is on the source plane.
        numerical_aperture : float, optional
            An object point's object-space NA; between 0 and 1.

        Returns
        -------
        FieldInfo
            The field after the change.

        Example
        -------
        >>> bench = Bench()
        >>> bench.set_field(0, angle_deg=2.0, colour="#ff8800").colour
        '#ff8800'
        """
        changes = {
            "diameter_mm": diameter_mm,
            "angle_deg": angle_deg,
            "vergence_dioptres": vergence_dioptres,
            "offset_mm": offset_mm,
            "wavelength_nm": wavelength_nm,
            "height_x_mm": height_x_mm,
            "height_y_mm": height_y_mm,
            "numerical_aperture": numerical_aperture,
        }
        arguments: dict[str, float | str] = {
            name: float(value) for name, value in changes.items() if value is not None
        }
        if colour is not None:
            arguments["colour"] = colour
        self._call("set_field", field_index=int(field_index), **arguments)
        return self.fields[int(field_index)]

    def set_finite_object(self, finite: bool = True) -> list[FieldInfo]:
        """Make every field a point of a finite object (``finite``), or a beam, and return them.

        A finite object sits on the source plane, which is then the object plane: each field
        is a point at a height there, sending a cone of rays of numerical aperture NA towards
        the first lens. Each field keeps its chief-ray angle, wavelength and colour. If the
        source plane is on the first lens, it is unlocked and moved to the lens's front focal
        plane (or as far forward as there is room).

        Example
        -------
        >>> bench = Bench()
        >>> [field.kind for field in bench.set_finite_object()]
        ['object_point']
        >>> bench.source_z_mm
        0.0
        """
        self._call("set_finite_object", finite=bool(finite))
        return self.fields

    def add_field(self, **changes) -> FieldInfo:
        """Add a field, a copy of the selected one, and select it.

        Starting from a copy means the new field already suits the bench; pass any of the
        ``set_field`` properties to change them at once. A new field is drawn in the colour
        of its wavelength.

        Example
        -------
        >>> bench = Bench()
        >>> new_field = bench.add_field(angle_deg=1.5)
        >>> new_field.index, new_field.angle_deg, new_field.diameter_mm, new_field.selected
        (1, 1.5, 10.0, True)
        """
        previously_selected = self.selected_field_index
        self._call("add_field")
        new_index = len(self.fields) - 1
        if not changes:
            return self.fields[new_index]
        try:
            return self.set_field(new_index, **changes)
        except (ValueError, TypeError):  # a bad value or property name: leave no half-made field
            self.remove_fields([new_index])
            self.select_field(previously_selected)
            raise

    def remove_fields(self, field_indices: list[int]) -> None:
        """Remove fields by index. At least one field always remains (the first, if all
        are listed).

        Example
        -------
        >>> bench = Bench()
        >>> _ = bench.add_field(wavelength_nm=486)
        >>> bench.remove_fields([0])
        >>> [field.wavelength_nm for field in bench.fields]
        [486.0]
        """
        self._call("remove_fields", field_indices=[int(index) for index in field_indices])

    def select_field(self, field_index: int) -> None:
        """Choose the field that autofocus and the single-field analyses use by default."""
        self._call("select_field", field_index=int(field_index))
