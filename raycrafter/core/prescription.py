"""Lens prescriptions: the surfaces of a single catalogue lens.

A prescription is a list of ``LensSurface`` objects, front to back, each holding the radius
of curvature, the distance to the next surface and the material that fills that gap. This
module reads prescriptions from Zemax ZMX text (through Optiland), flips them, and computes
simple geometry such as surface sag and lens thickness. It has no Qt code.
"""

from __future__ import annotations

import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from optiland.fileio import load_zemax_file


@dataclass
class LensSurface:
    """One lens surface: its radius of curvature and the material that follows it.

    Attributes
    ----------
    radius : float
        Radius of curvature in mm; ``inf`` for a flat surface.
    thickness_after : float
        Distance in mm from this surface's vertex to the next one.
    material_after : str
        Glass name of the material after this surface, or ``"air"``.
    """

    radius: float
    thickness_after: float
    material_after: str


def prescription_from_zmx_text(zmx_text: str) -> list[LensSurface]:
    """Parse ZMX text with Optiland and return the lens surfaces.

    The text is written to a temporary UTF-8 file because Optiland's reader needs a path.
    The object and image surfaces are dropped (the bench places its own image plane), and so
    is any leading flat air surface (the air-spaced doublets have one). A surface is followed
    by glass if Optiland gives its following material a name; otherwise it is air.

    Parameters
    ----------
    zmx_text : str
        Contents of a Zemax .zmx file for one lens.

    Returns
    -------
    list of LensSurface
        The lens surfaces, front to back, in the catalogue orientation.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "lens.zmx"
        temp_path.write_text(zmx_text, encoding="utf-8")
        with warnings.catch_warnings():
            # Optiland warns about catalogue details it cannot use; none matter for a single lens.
            warnings.simplefilter("ignore")
            loaded_optic = load_zemax_file(str(temp_path))
            surfaces = loaded_optic.surfaces.surfaces[1:-1]

    lens_surfaces = []
    for surface in surfaces:
        material_name = getattr(surface.material_post, "name", None)
        lens_surfaces.append(
            LensSurface(
                radius=float(surface.geometry.radius),
                thickness_after=float(surface.thickness),
                material_after=material_name if isinstance(material_name, str) else "air",
            )
        )

    while (
        lens_surfaces
        and np.isinf(lens_surfaces[0].radius)
        and lens_surfaces[0].material_after == "air"
    ):
        lens_surfaces.pop(0)
    return lens_surfaces


def surface_sag(radius: float, height: np.ndarray) -> np.ndarray:
    """Sag (z displacement from the vertex, mm) of a spherical surface at the given heights.

    A flat surface (infinite radius) has no sag. Heights beyond the sphere are clipped.
    """
    if np.isinf(radius):
        return np.zeros_like(height, dtype=float)
    return radius - np.sign(radius) * np.sqrt(np.maximum(radius**2 - height**2, 0.0))


def flipped_prescription(lens_surfaces: list[LensSurface]) -> list[LensSurface]:
    """Return the lens turned around: surface order reversed and radii negated.

    The glass that followed surface k now precedes it, so each surface takes the thickness
    and material that used to sit *before* it in the original order.
    """
    surface_count = len(lens_surfaces)
    flipped = []
    for new_index in range(surface_count):
        old_index = surface_count - 1 - new_index
        previous_old_index = old_index - 1
        if previous_old_index >= 0:
            thickness = lens_surfaces[previous_old_index].thickness_after
            material = lens_surfaces[previous_old_index].material_after
        else:
            thickness, material = 0.0, "air"
        flipped.append(LensSurface(-lens_surfaces[old_index].radius, thickness, material))
    return flipped


def lens_thickness(lens_surfaces: list[LensSurface]) -> float:
    """Distance (mm) from the front vertex to the back vertex: the sum of the glass thicknesses."""
    return sum(surface.thickness_after for surface in lens_surfaces[:-1])
