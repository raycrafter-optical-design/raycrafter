"""The state of the lens bench: the lenses on the axis, the beam and the image plane.

All z positions are absolute positions along the optical axis, in mm. The beam comes from
the left (negative z). These classes hold data only; editing is done by ``OpticalModel``
(``raycrafter.core.model``) and optics by ``raycrafter.core.optic_builder`` and
``raycrafter.core.analysis``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from raycrafter.core.prescription import LensSurface, flipped_prescription, lens_thickness

AXIS_START_Z = 0.0  # mm, where the beam starts


@dataclass
class PlacedLens:
    """One lens on the optical axis: its prescription, position, orientation and size.

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
    group_id : int or None
        Lenses that share an id form a Plossl pair and move together.
    is_negative : bool
        True for a diverging lens (negative focal length). These cannot form a Plossl pair.
    """

    part_number: str
    surfaces: list[LensSurface]
    semi_aperture: float
    z: float
    flipped: bool = False
    group_id: int | None = None
    is_negative: bool = False

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
        """Front-vertex to back-vertex distance (mm)."""
        return lens_thickness(self.surfaces)

    @property
    def back_z(self) -> float:
        """z of the back vertex (mm)."""
        return self.z + self.thickness


@dataclass
class OpticalState:
    """Everything the user can change: the lenses, the beam and the image plane.

    Attributes
    ----------
    lenses : list of PlacedLens
        Ordered left to right along the axis.
    next_group_id : int
        The id the next Plossl pair will get.
    beam_diameter : float
        Beam diameter at the first lens, in mm.
    field_angle : float
        Beam angle to the axis, in degrees.
    beam_vergence : float
        Dioptres (1/m): positive is converging, negative diverging, zero collimated.
    image_z : float
        z of the image plane, in mm.
    """

    lenses: list[PlacedLens] = field(default_factory=list)
    next_group_id: int = 1
    beam_diameter: float = 10.0
    field_angle: float = 0.0
    beam_vergence: float = 0.0
    image_z: float = 310.0

    def last_back_z(self) -> float:
        """z of the back vertex of the rightmost lens (0 if there are no lenses)."""
        return self.lenses[-1].back_z if self.lenses else AXIS_START_Z
