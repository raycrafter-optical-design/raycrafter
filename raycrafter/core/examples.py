"""Example designs to start from, such as a basic microscope.

Each example is a function that returns a new ``OpticalState``. The GUI offers them under
File > New from example (``EXAMPLES``); scripts and agents can call them directly.
"""

from __future__ import annotations

from collections.abc import Callable

from raycrafter.core.placement import make_paraxial_lens
from raycrafter.core.state import AnyField, ObjectPoint, OpticalState
from raycrafter.palette import field_colour

OBJECT_Z_MM = 20.0  # where an example's object sits, leaving room in front of it


def basic_microscope(
    objective_focal_length_mm: float = 10.0,
    tube_focal_length_mm: float = 200.0,
    lens_gap_mm: float = 100.0,
    numerical_aperture: float = 0.25,
    field_height_mm: float = 0.5,
) -> OpticalState:
    """An infinity-corrected microscope made of two paraxial lenses.

    The object (the sample) is one focal length in front of the objective, so each of its
    points leaves the objective as a collimated beam. The tube lens focuses those beams on
    the image plane, one focal length behind it. The gap between the two lenses does not
    change where the image is or how big it is: the magnification is
    -tube_focal_length / objective_focal_length (-20 by default).

    Parameters
    ----------
    objective_focal_length_mm, tube_focal_length_mm : float
        Focal lengths of the two lenses (both positive).
    lens_gap_mm : float
        Distance from the objective to the tube lens (the collimated space).
    numerical_aperture : float
        Object-space NA of every object point.
    field_height_mm : float
        Height of the second object point; the first is on the axis.

    Returns
    -------
    OpticalState
        The design, with two object points (on axis and at ``field_height_mm``) and the
        source plane, which is the object plane, unlocked from the objective.
    """
    objective_z = OBJECT_Z_MM + objective_focal_length_mm
    tube_z = objective_z + lens_gap_mm
    points: list[AnyField] = [
        ObjectPoint(height_y_mm=height_mm, numerical_aperture=numerical_aperture)
        for height_mm in (0.0, field_height_mm)
    ]
    for field_number, point in enumerate(points):
        point.colour = field_colour(field_number)
    return OpticalState(
        lenses=[
            make_paraxial_lens(objective_focal_length_mm, z=objective_z),
            make_paraxial_lens(tube_focal_length_mm, z=tube_z),
        ],
        fields=points,
        image_z=tube_z + tube_focal_length_mm,
        source_z=OBJECT_Z_MM,
        source_gap_locked=False,
    )


# The examples offered in the GUI, by the name shown in the menu.
EXAMPLES: dict[str, Callable[[], OpticalState]] = {
    "Basic microscope": basic_microscope,
}
