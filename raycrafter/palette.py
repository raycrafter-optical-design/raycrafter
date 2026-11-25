"""Colours shared by the GUI and the HTML report, so both look like the same instrument.

Plain hex strings, with no Qt or matplotlib dependency.
"""

import re

import numpy as np

LENS = "#3b7dd8"  # blue
LENS_HOVER = "#1c4f9c"  # darker blue for the lens under the pointer
PARAXIAL_ARROW = "#1d3e6c"  # arrowheads of a paraxial lens: LENS at half brightness
PLOSSL = "#7048e8"  # purple: a lens in a Plossl pair
PLOSSL_HOVER = "#4c2fb0"
PLOSSL_PREVIEW = "#e03131"  # dashed red outline of a proposed Plossl pair
GHOST_BLOCKED = "#e03131"  # red: a lens being placed where there is no room for it
RAY = "#e8a317"  # amber: the rays in the app icon
IMAGE = "#2f9e44"  # green: the image plane and "current image plane" markers
TANGENTIAL = "#3b7dd8"  # also the through-focus curve
SAGITTAL = "#e8590c"
AIRY = "#868e96"  # Airy disk, Airy radius and diffraction limit
TEXT = "#343a40"
FAINT = "#ced4da"  # zero lines and borders
AXIS_LINE = "#adb5bd"  # dashed optical axis in the side view
SIDE_GRID = "#e9ecef"  # faint square grid behind the side view
TIMING = "#adb5bd"  # trace/draw timing text
CANVAS_BORDER = "#adb5bd"  # thin frame around each plot

# Colours for telling field angles apart, in the order Zemax uses (blue, green, red, ...),
# dark enough to read on white. After the last one they repeat.
FIELD_COLOURS = [
    "#1c62d6",  # blue
    "#2b9a3e",  # green
    "#d62728",  # red
    "#d9a400",  # yellow
    "#8e44c9",  # purple
    "#0f9fb0",  # cyan
    "#e8590c",  # orange
    "#c2255c",  # pink
    "#5c3d2e",  # brown
    "#495057",  # grey
]


def is_hex_colour(text: str) -> bool:
    """True for a colour written as ``#rrggbb`` (case does not matter), e.g. ``"#3b7dd8"``."""
    return re.fullmatch(r"#[0-9a-fA-F]{6}", text) is not None


def field_colour(field_number: int) -> str:
    """The colour of the field with this (zero-based) number when fields are told apart."""
    return FIELD_COLOURS[field_number % len(FIELD_COLOURS)]


# Beam colours from wavelength are darkened to this fraction of full brightness, so that
# yellow and green beams still stand out against the white background.
BEAM_COLOUR_BRIGHTNESS = 0.8

# Spectral colour stops, (wavelength in nm, (red, green, blue) from 0 to 1), after the
# widely used approximation by Dan Bruton. Colours are interpolated linearly between stops;
# wavelengths outside the visible range take the colour of the nearest end.
_SPECTRUM_STOPS = [
    (380.0, (0.5, 0.0, 1.0)),  # violet
    (440.0, (0.0, 0.0, 1.0)),  # blue
    (490.0, (0.0, 1.0, 1.0)),  # cyan
    (510.0, (0.0, 1.0, 0.0)),  # green
    (580.0, (1.0, 1.0, 0.0)),  # yellow
    (645.0, (1.0, 0.0, 0.0)),  # red
    (780.0, (0.5, 0.0, 0.0)),  # deep red, used for the near infrared too
]


def wavelength_colour(wavelength_nm: float) -> str:
    """An approximate colour for light of the given wavelength, as a hex string.

    Parameters
    ----------
    wavelength_nm : float
        Wavelength in nanometres.

    Returns
    -------
    str
        A colour such as ``"#83cc00"`` (for 555 nm), darkened by BEAM_COLOUR_BRIGHTNESS.
    """
    stop_wavelengths = [stop_wavelength for stop_wavelength, _rgb in _SPECTRUM_STOPS]
    channels = []
    for channel in range(3):  # red, green, blue
        channel_values = [stop_rgb[channel] for _wavelength, stop_rgb in _SPECTRUM_STOPS]
        # np.interp holds the end values beyond either end of the spectrum.
        value = float(np.interp(wavelength_nm, stop_wavelengths, channel_values))
        channels.append(round(255 * BEAM_COLOUR_BRIGHTNESS * value))
    red, green, blue = channels
    return f"#{red:02x}{green:02x}{blue:02x}"
