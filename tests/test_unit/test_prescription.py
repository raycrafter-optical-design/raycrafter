"""Tests of lens prescriptions: reading ZMX text, flipping and geometry."""

import numpy as np

from raycrafter.core.prescription import (
    flipped_prescription,
    lens_thickness,
    prescription_from_zmx_text,
    surface_sag,
)
from raycrafter.lenscatalog import thorlabs


def act508_200_surfaces():
    """The ACT508-200-A prescription from the packaged catalogue."""
    entry = thorlabs.load_achromats()["ACT508-200-A"]
    return prescription_from_zmx_text(entry.zmx_text)


def test_cemented_doublet_has_three_surfaces_and_two_glasses():
    surfaces = act508_200_surfaces()
    assert len(surfaces) == 3
    assert [surface.material_after for surface in surfaces[:2]] != ["air", "air"]
    assert surfaces[-1].material_after == "air"


def test_flipping_twice_gives_the_original():
    # The last surface's thickness_after is the air gap behind the lens, which the bench sets
    # itself, so it is not part of the lens and is not kept by a flip.
    surfaces = act508_200_surfaces()
    twice_flipped = flipped_prescription(flipped_prescription(surfaces))
    assert [surface.radius for surface in twice_flipped] == [surface.radius for surface in surfaces]
    assert twice_flipped[:-1] == surfaces[:-1]


def test_flipping_keeps_the_thickness_and_reverses_the_radii():
    surfaces = act508_200_surfaces()
    flipped = flipped_prescription(surfaces)
    assert lens_thickness(flipped) == lens_thickness(surfaces)
    assert [surface.radius for surface in flipped] == [
        -surface.radius for surface in surfaces[::-1]
    ]


def test_sag_of_a_sphere():
    radius, height = 100.0, np.array([0.0, 30.0])
    expected = radius - np.sqrt(radius**2 - height**2)
    np.testing.assert_allclose(surface_sag(radius, height), expected)
    np.testing.assert_allclose(surface_sag(np.inf, height), 0.0)
