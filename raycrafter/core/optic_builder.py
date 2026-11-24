"""Build an Optiland ``Optic`` from the bench state, and export it as a Zemax file.

A fresh Optiland system is built from ``OpticalState`` for every update, rather than
editing a long-lived one (see log.md, "rebuild the Optiland system on every change").

A collimated beam is modelled as an object at infinity with an entrance pupil diameter
equal to the beam diameter. A converging or diverging beam moves the object to a finite
(possibly virtual) distance instead.

Every lens surface is given its clear aperture, so rays that pass outside a lens's rim are
blocked: Optiland sets their intensity to zero, and the analyses leave them out. Without
this, a beam wider than a lens was traced through the (non-existent) continuation of its
surfaces, giving wrong results or NaN where a sphere does not reach that far (issue #3).
"""

from __future__ import annotations

import warnings

import numpy as np
from optiland.fileio import save_zemax_file
from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture

from raycrafter.core.state import AXIS_START_Z, OpticalState

WAVELENGTH_UM = 0.555  # the single wavelength used throughout (green)


def object_distance_mm(beam_vergence: float) -> float:
    """Distance (mm) from the object to the first lens surface that gives the beam its vergence.

    A collimated beam (vergence 0) comes from infinity. A diverging beam (negative vergence)
    comes from a real point source 1000/|V| mm in front of the first surface. A converging
    beam (positive vergence) is aimed at a point 1000/V mm behind the first surface, which is
    a virtual object, i.e. a negative object distance.

    Parameters
    ----------
    beam_vergence : float
        Vergence in dioptres (1/m).
    """
    if beam_vergence == 0.0:
        return np.inf
    return -1000.0 / beam_vergence


def build_optic(
    state: OpticalState,
    field_angles: list[float] | None = None,
    beam_diameter_mm: float | None = None,
) -> Optic:
    """Create a fresh Optiland system from the bench state.

    Lenses are chained left to right. The last surface of each lens is followed by air up
    to the next lens (or the image plane). Each lens surface has the lens's clear aperture,
    which blocks rays outside the rim. The first surface is where the entrance pupil (the
    beam) is defined. With no lenses a flat air surface takes that role, so the beam still
    reaches the image.

    Parameters
    ----------
    state : OpticalState
        The lenses, beam and image plane.
    field_angles : list of float, optional
        Field angles in degrees. By default there is one field, at the current beam angle.
    beam_diameter_mm : float, optional
        Use this beam diameter instead of the state's (the MTF uses the part of the beam
        that gets through; see ``analysis.transmitted_beam_diameter``).

    Returns
    -------
    optiland.optic.Optic
        The system, ready to trace.
    """
    optic = Optic()
    optic.surfaces.add(index=0, radius=np.inf, thickness=object_distance_mm(state.beam_vergence))

    if state.lenses:
        surface_index = 1
        for lens_number, lens in enumerate(state.lenses):
            is_last_lens = lens_number == len(state.lenses) - 1
            next_front_z = state.image_z if is_last_lens else state.lenses[lens_number + 1].z
            gap_after_lens = max(next_front_z - lens.back_z, 0.0)

            surfaces = lens.surfaces_in_use()
            for surface_number, surface in enumerate(surfaces):
                is_last_surface = surface_number == len(surfaces) - 1
                surface_settings: dict[str, object] = dict(
                    index=surface_index,
                    radius=surface.radius,
                    thickness=gap_after_lens if is_last_surface else surface.thickness_after,
                    is_stop=(surface_index == 1),
                    aperture=RadialAperture(r_max=lens.semi_aperture),
                )
                # Optiland uses air when no material is given ("material=None" is rejected).
                if surface.material_after != "air" and not is_last_surface:
                    surface_settings["material"] = surface.material_after
                optic.surfaces.add(**surface_settings)
                surface_index += 1
        image_index = surface_index
    else:
        optic.surfaces.add(
            index=1,
            radius=np.inf,
            thickness=max(state.image_z - AXIS_START_Z, 0.0),
            is_stop=True,
        )
        image_index = 2

    optic.surfaces.add(index=image_index)
    optic.set_aperture("EPD", state.beam_diameter if beam_diameter_mm is None else beam_diameter_mm)
    optic.fields.set_type("angle")
    for angle in field_angles if field_angles is not None else [abs(state.field_angle)]:
        optic.fields.add(y=angle)
    optic.wavelengths.add(WAVELENGTH_UM, is_primary=True)
    return optic


def save_zmx(state: OpticalState, file_path: str) -> None:
    """Write the lens arrangement (lenses, gaps and image plane) to a Zemax .zmx file."""
    with warnings.catch_warnings():
        # Glasses missing from Zemax's catalogue are written as model glass; that warning is
        # expected.
        warnings.simplefilter("ignore")
        save_zemax_file(build_optic(state), file_path)
