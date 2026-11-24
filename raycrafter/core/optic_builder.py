"""Build an Optiland ``Optic`` from the bench state, and export it as a Zemax file.

A fresh Optiland system is built from ``OpticalState`` for every update, rather than
editing a long-lived one (see log.md, "rebuild the Optiland system on every change").

Most analyses launch each field's rays themselves (``raycrafter.core.ray_launch``), so the
system they use (``build_optic``) has its object at infinity and Optiland's own fields and
aperture play no part in them. The MTF and the Zemax export need Optiland's object and
entrance pupil model instead (``build_optic_for_beam``): there a collimated beam is an
object at infinity with an entrance pupil diameter equal to the beam diameter, and a
converging or diverging beam moves the object to a finite (possibly virtual) distance.

Every lens surface is given its clear aperture, so rays that pass outside a lens's rim are
blocked: Optiland sets their intensity to zero, and the analyses leave them out. Without
this, a beam wider than a lens was traced through the (non-existent) continuation of its
surfaces, giving wrong results or NaN where a sphere does not reach that far (issue #3).
"""

from __future__ import annotations

import warnings

import numpy as np
from optiland.fileio import save_zemax_file
from optiland.materials.warnings import OptilandMaterialWarning
from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture

from raycrafter.core.state import AXIS_START_Z, BeamField, OpticalState


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


def build_optic(state: OpticalState) -> Optic:
    """The Optiland system for tracing rays launched by ``ray_launch``.

    The object is at infinity, so that it does not get in the way of rays that start at
    the start of the axis. Optiland's single field (0 degrees), aperture and wavelength are
    set from the selected field only because its paraxial calculations (the effective and
    back focal lengths) need them.
    """
    beam = state.selected_field
    optic = Optic()
    optic.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    _add_lens_and_image_surfaces(optic, state)
    optic.set_aperture("EPD", beam.diameter_mm)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(beam.wavelength_um, is_primary=True)
    return optic


def build_optic_for_beam(
    state: OpticalState,
    beam: BeamField,
    field_angles: list[float] | None = None,
    beam_diameter_mm: float | None = None,
) -> Optic:
    """The Optiland system with Optiland's own object, pupil and field set up for one beam.

    The beam's offset cannot be described this way, so it is left out (the beam is centred
    on the first lens).

    Parameters
    ----------
    state : OpticalState
        The lenses and image plane.
    beam : BeamField
        Sets the object distance (from the vergence), the entrance pupil diameter and the
        wavelength.
    field_angles : list of float, optional
        Field angles in degrees. By default there is one field, at the beam's angle.
    beam_diameter_mm : float, optional
        Use this entrance pupil diameter instead of the beam's (the MTF uses the part of the
        beam that gets through; see ``analysis.transmitted_beam_diameter``).

    Returns
    -------
    optiland.optic.Optic
        The system, ready to trace.
    """
    optic = Optic()
    optic.surfaces.add(index=0, radius=np.inf, thickness=object_distance_mm(beam.vergence_dioptres))
    _add_lens_and_image_surfaces(optic, state)
    optic.set_aperture("EPD", beam.diameter_mm if beam_diameter_mm is None else beam_diameter_mm)
    optic.fields.set_type("angle")
    for angle in field_angles if field_angles is not None else [abs(beam.angle_deg)]:
        optic.fields.add(y=angle)
    optic.wavelengths.add(beam.wavelength_um, is_primary=True)
    return optic


def _add_lens_and_image_surfaces(optic: Optic, state: OpticalState) -> None:
    """Add every lens surface, then the image surface, after the object surface.

    Lenses are chained left to right; a paraxial lens is a single Optiland paraxial surface.
    The last surface of each lens is followed by air up to the next lens (or the image
    plane). Each lens surface has the lens's clear aperture, which blocks rays outside the
    rim. The first surface is the stop. With no lenses a flat air surface at the start of
    the axis takes that role, so the beam still reaches the image.
    """
    if not state.lenses:
        optic.surfaces.add(
            index=1,
            radius=np.inf,
            thickness=max(state.image_z - AXIS_START_Z, 0.0),
            is_stop=True,
        )
        optic.surfaces.add(index=2)
        return

    surface_index = 1
    for lens_number, lens in enumerate(state.lenses):
        is_last_lens = lens_number == len(state.lenses) - 1
        next_front_z = state.image_z if is_last_lens else state.lenses[lens_number + 1].z
        gap_after_lens = max(next_front_z - lens.back_z, 0.0)

        if lens.is_paraxial:
            # An ideal thin lens is one flat Optiland "paraxial" surface that bends every
            # ray as a perfect lens of focal length f would.
            optic.surfaces.add(
                index=surface_index,
                surface_type="paraxial",
                f=lens.paraxial_focal_length_mm,
                thickness=gap_after_lens,
                is_stop=(surface_index == 1),
                aperture=RadialAperture(r_max=lens.semi_aperture),
            )
            surface_index += 1
            continue

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
            with warnings.catch_warnings():
                # Some glass names (e.g. SF10) are in several of Optiland's catalogues. It
                # warns and picks Schott's, which is the right one for the Thorlabs lenses.
                warnings.simplefilter("ignore", OptilandMaterialWarning)
                optic.surfaces.add(**surface_settings)
            surface_index += 1
    optic.surfaces.add(index=surface_index)


def save_zmx(state: OpticalState, file_path: str) -> None:
    """Write the system to a Zemax .zmx file.

    Zemax gives every field the same object and entrance pupil, so those come from the
    selected field. Every field's angle and wavelength is written (each once), with the
    selected field's wavelength as the primary one. Field offsets are not exported.
    """
    selected = state.selected_field
    field_angles = sorted({abs(beam.angle_deg) for beam in state.fields})
    optic = build_optic_for_beam(state, selected, field_angles=field_angles)
    for wavelength_nm in sorted({beam.wavelength_nm for beam in state.fields}):
        if wavelength_nm != selected.wavelength_nm:
            optic.wavelengths.add(wavelength_nm / 1000.0)

    with warnings.catch_warnings():
        # Glasses missing from Zemax's catalogue are written as model glass; that warning is
        # expected.
        warnings.simplefilter("ignore")
        save_zemax_file(optic, file_path)
