"""Colours shared by the GUI and the HTML report, so both look like the same instrument.

Plain hex strings, with no Qt or matplotlib dependency.
"""

LENS = "#3b7dd8"  # blue
LENS_HOVER = "#1c4f9c"  # darker blue for the lens under the pointer
PARAXIAL_ARROW = "#1d3e6c"  # arrowheads of a paraxial lens: LENS at half brightness
PLOSSL = "#7048e8"  # purple: a lens in a Plossl pair
PLOSSL_HOVER = "#4c2fb0"
PLOSSL_PREVIEW = "#e03131"  # dashed red outline of a proposed Plossl pair
GHOST_BLOCKED = "#e03131"  # red: a lens being placed where there is no room for it
RAY = "#e8a317"  # amber
IMAGE = "#2f9e44"  # green: the image plane and "current image plane" markers
TANGENTIAL = "#3b7dd8"  # also the spot at the current beam angle
SAGITTAL = "#e8590c"
ON_AXIS = "#e8590c"  # the 0 degree spot, drawn under the current-angle spot
AIRY = "#868e96"  # Airy disk, Airy radius and diffraction limit
TEXT = "#343a40"
FAINT = "#ced4da"  # zero lines and borders
AXIS_LINE = "#adb5bd"  # dashed optical axis in the side view
SIDE_GRID = "#e9ecef"  # faint square grid behind the side view
TIMING = "#adb5bd"  # trace/draw timing text
CANVAS_BORDER = "#adb5bd"  # thin frame around each plot
