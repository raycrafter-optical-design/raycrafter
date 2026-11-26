"""Plain-language explanations of each analysis plot, shown by the report's "?" buttons.

Each entry is plot key -> (heading, list of paragraphs). The paragraphs may contain simple
HTML (<b>, <i>). Kept separate from the page code so the wording is easy to find and edit,
and so the GUI can reuse it later.
"""

EXPLANATIONS = {
    "spot": (
        "Spot diagram",
        [
            "Where a bundle of rays lands on the image plane, seen face-on. Each dot is one ray; the "
            "picture is centred on the middle of the cloud so you can judge its size and shape.",
            "A <b>small, round, tight</b> cloud means a sharp focus. Comet or fan shapes point to "
            "off-axis aberrations such as coma.",
            "The <b>dashed grey circle</b> is the Airy disk, the smallest spot even a perfect lens can make "
            "because of diffraction. If the cloud sits inside it, the system is <i>diffraction limited</i>.",
            "Each field (beam) is drawn in its own colour, centred on its own spot. The Airy disk and "
            "the RMS radius quoted are those of the selected field.",
        ],
    ),
    "through_focus": (
        "Through-focus RMS radius",
        [
            "How the spot size changes as the image plane is slid back and forth. The horizontal axis is the "
            "distance from the current image plane; the vertical axis is the RMS spot radius (smaller is "
            "better).",
            "The <b>green line</b> marks where the image plane is now. The <b>dotted blue line</b> marks "
            "the best focus, the lowest point of the curve. If they differ, moving the image plane to the "
            "dotted line would sharpen the image (the Autofocus button does this).",
            "The <b>dashed grey line</b> is the Airy radius, the diffraction limit. A curve that dips "
            "below it is diffraction limited at its best focus. A narrow dip means a shallow depth of focus.",
        ],
    ),
    "mtf": (
        "Modulation transfer function (MTF)",
        [
            "How much contrast the lens keeps at each level of fine detail. The horizontal axis is spatial "
            "frequency (cycles per millimetre at the image; high numbers are fine detail). The vertical axis "
            "is the contrast that survives, from 1 (perfect) down to 0 (detail is lost).",
            "The <b>dashed grey curve</b> is the diffraction limit, the best any lens could do for this "
            "aperture. The closer the blue (tangential) and orange (sagittal) curves lie to it, the better the "
            "lens.",
            "Tangential and sagittal refer to detail oriented across and along the line to the field centre; "
            "when they split apart the image is astigmatic.",
        ],
    ),
    "ray_fan": (
        "Ray fan",
        [
            "Shows where each ray across the aperture lands relative to the chief ray (the ray through the "
            "centre of the pupil). The horizontal axis is the position of the ray in the pupil, from one edge "
            "(-1) to the other (+1). The vertical axis is how far from ideal it lands, in micrometres.",
            "A perfect system gives a <b>flat line on zero</b>. A cubic S-shape means spherical aberration, a "
            "straight tilted line means the image plane is out of focus, and a curve that is not symmetric "
            "means coma.",
            "Blue is the tangential fan (rays in the plane containing the axis and beam); orange dashed is "
            "the sagittal fan (rays across it).",
        ],
    ),
    "field": (
        "Field curvature",
        [
            "Where the best focus lies for beams arriving at different angles. The vertical axis is the beam "
            "angle from the axis; the horizontal axis is how far the best focus is from the current image plane "
            "(zero is on the green line, negative is in front of it).",
            "A flat detector can only be at one distance, so the further these curves bend away from the "
            "green line, the more out of focus the edges of the field are when the centre is sharp.",
            "Blue (tangential) and orange (sagittal) separating from each other is astigmatism. The dashed "
            "horizontal line, when present, marks the selected field's angle.",
        ],
    ),
    "grid_distortion": (
        "Grid distortion",
        [
            "How a square grid on the object comes out in the image, seen face-on. The <b>dashed grey "
            "grid</b> is a perfectly scaled copy of the object grid; the <b>blue grid</b> shows where "
            "its points really land. The corners of the grid are as far from the axis as the "
            "farthest object point.",
            "Blue corners outside the grey ones is <i>pincushion</i> distortion; inside them is "
            "<i>barrel</i> distortion. Distortion moves image points without blurring them.",
            "The grey grid is scaled by the magnification measured next to the axis on the image "
            "plane. Each image point is that of the chief ray (the ray aimed at the centre of the "
            "first lens). A missing point means its chief ray is blocked.",
        ],
    ),
    "distortion": (
        "Distortion against object height",
        [
            "The vertical axis is the height of an object point above the axis; the horizontal axis is "
            "how much further from the axis its image lands than it would without distortion, in "
            "percent.",
            "Positive values are <b>pincushion</b> distortion, negative values <b>barrel</b> distortion. "
            "A system free of distortion gives a line on zero. The dashed horizontal line marks the "
            "selected field's height.",
        ],
    ),
}
