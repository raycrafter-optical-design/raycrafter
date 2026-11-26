"""Build a self-contained HTML report of the current lens system.

The report has the ray diagram across the top, a 3D view of the design that can be turned
round (``raycrafter.report.viewer_3d``), and the five analysis plots below them (spot
diagram, through-focus, MTF, ray fan, field curvature), with a row of summary tiles. Every
plot has a "?" button that opens a note explaining how to read it. Plots are inline SVG and
the style sheet and scripts (in ``assets``) are copied into the page, so the result is a
single HTML file that can be shared, and needs no other files or internet connection.

Example
-------
>>> from raycrafter.core import OpticalModel
>>> from raycrafter.report import build_report_html
>>> page = build_report_html(OpticalModel())  # doctest: +SKIP
"""

from __future__ import annotations

import html
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path

from raycrafter.core.analysis import airy_radius
from raycrafter.core.fields import chief_ray_angle_deg
from raycrafter.core.model import OpticalModel
from raycrafter.core.plot_geometry import (
    f_number_text,
    field_name,
    field_position_text,
    image_point_text,
    magnification_text,
    rms_radius_text,
)
from raycrafter.core.state import AnyField, ObjectPoint
from raycrafter.report import figures, viewer_3d
from raycrafter.report.explanations import EXPLANATIONS
from raycrafter.report.viewer_3d import json_for_script, read_asset


def card_html(
    key: str, title: str, subtitle_html: str, body: str, width_class: str, with_help: bool = True
) -> str:
    """One card: a heading, a one-line subtitle, the "?" button (optional) and the body (the
    plot, as inline SVG, or the 3D view)."""
    help_button = ""
    if with_help:
        help_button = (
            f'<button class="help" data-key="{key}" '
            f'aria-label="What does the {html.escape(title)} show?">?</button>'
        )
    return (
        f'<section class="card {width_class}">'
        f'<div class="card-head"><div><h2>{html.escape(title)}</h2>'
        f'<p class="sub">{subtitle_html}</p></div>'
        f"{help_button}</div>{body}</section>"
    )


def chip_html(label: str, value: str) -> str:
    """One small summary tile at the top of the page."""
    return (
        f'<div class="chip"><span class="label">{html.escape(label)}</span>'
        f'<span class="value">{value}</span></div>'
    )


def vergence_words(vergence: float) -> str:
    """Plain-words description of the beam vergence: collimated, converging or diverging."""
    if vergence == 0.0:
        return "collimated"
    return "converging" if vergence > 0 else "diverging"


def field_html(field_index: int, light: AnyField) -> str:
    """A one-line description of a field, in the field's colour."""
    if isinstance(light, ObjectPoint):
        text = (
            f"{field_name(field_index)}: object point at {field_position_text(light)}, "
            f"NA {light.numerical_aperture:g}, {light.wavelength_nm:g} nm"
        )
    else:
        text = (
            f"{field_name(field_index)}: {light.diameter_mm:g} mm "
            f"{vergence_words(light.vergence_dioptres)} at {light.angle_deg:+.1f}°, "
            f"{light.wavelength_nm:g} nm"
        )
        if light.offset_mm != 0.0:
            text += f", offset {light.offset_mm:+g} mm"
    return f'<span style="color:{light.display_colour()}">{html.escape(text)}</span>'


def build_report_html(model: OpticalModel) -> str:
    """Trace the current system, draw every plot and return the whole report as HTML.

    Parameters
    ----------
    model : OpticalModel
        The bench to report on. It is not changed.

    Returns
    -------
    str
        A complete, self-contained HTML document.
    """
    state = model.state
    fast_trace = model.trace_fast()
    through_focus = model.compute_through_focus(fast_trace)
    mtf = model.compute_mtf()
    ray_fan = model.compute_ray_fan()
    curvature = model.compute_field_curvature()

    spot = fast_trace.spot
    summary = fast_trace.summary
    airy = airy_radius(spot)
    selected = state.selected_field

    chips = [
        chip_html("Fields", str(len(state.fields))),
        chip_html("Image plane", f"z = {state.image_z:.1f} mm"),
        chip_html("RMS radius", rms_radius_text(spot, "µm")),
    ]
    if summary is not None:
        chips += [
            chip_html("EFL", f"{summary.effective_focal_length:.1f} mm"),
            chip_html("BFL", f"{summary.back_focal_length:.1f} mm"),
        ]
        if summary.magnification is not None:
            chips.append(chip_html("Magnification", magnification_text(summary.magnification)))
    if state.has_finite_object:
        image_x, image_y = fast_trace.chief_ray_image_points[state.selected_field_index]
        chips.append(chip_html("Image height", image_point_text(image_x, image_y)))
    f_number = model.compute_f_number()
    if f_number is not None:
        chips.append(chip_html("f-number", f_number_text(f_number)))
    if airy is not None:
        chips.append(chip_html("Airy radius", f"{airy * 1000:.1f} µm"))

    lens_names = ", ".join(
        html.escape(lens.part_number) + (" (flipped)" if lens.flipped else "")
        for lens in state.lenses
    )
    ray_subtitle = " &middot; ".join(
        [lens_names] + [field_html(index, beam) for index, beam in enumerate(state.fields)]
    )

    # The spot shows every field; the RMS radius quoted is the selected field's.
    spot_subtitle = f"RMS radius {rms_radius_text(spot, 'µm')}"
    if len(state.fields) > 1:
        spot_subtitle = (
            f'<span style="color:{selected.display_colour()}">'
            f"{field_name(state.selected_field_index)}: {spot_subtitle}</span>"
        )

    # "+ 0.0" turns a rounded -0.000 into 0.000.
    best_focus_text = f"{round(through_focus.best_offset, 3) + 0.0:+.3f}"
    angle_text = f"{field_name(state.selected_field_index)}, at {field_position_text(selected)}"
    cards = [
        card_html(
            "rays",
            "Ray diagram",
            ray_subtitle,
            figures.figure_to_inline_svg(figures.draw_ray_diagram(state, fast_trace)),
            "span-6",
            with_help=False,
        ),
        card_html(
            "view_3d",
            "3D view",
            "Drag to rotate &middot; scroll to zoom &middot; right-drag to pan",
            viewer_3d.viewer_html(),
            "span-6",
            with_help=False,
        ),
        card_html(
            "spot",
            "Spot diagram",
            spot_subtitle,
            figures.figure_to_inline_svg(figures.draw_spot_diagram(state, fast_trace)),
            "span-2",
        ),
        card_html(
            "through_focus",
            "Through-focus",
            f"Best focus {best_focus_text} mm from the image plane",
            figures.figure_to_inline_svg(figures.draw_through_focus(through_focus)),
            "span-2",
        ),
        card_html(
            "mtf", "MTF", angle_text, figures.figure_to_inline_svg(figures.draw_mtf(mtf)), "span-2"
        ),
        card_html(
            "ray_fan",
            "Ray fan",
            angle_text,
            figures.figure_to_inline_svg(figures.draw_ray_fan(ray_fan)),
            "span-3",
        ),
        card_html(
            "field",
            "Field curvature",
            "Best focus against field angle",
            figures.figure_to_inline_svg(
                figures.draw_field_curvature(curvature, chief_ray_angle_deg(state, selected))
            ),
            "span-3",
        ),
    ]

    explanation_data = {
        key: {"title": title, "paragraphs": paragraphs}
        for key, (title, paragraphs) in EXPLANATIONS.items()
    }
    explanation_json = json_for_script(explanation_data)

    generated = datetime.now().strftime("%d %B %Y, %H:%M")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lens system report</title>
<style>{read_asset("report.css")}</style>
</head>
<body>
<main>
  <header>
    <h1>Lens system report</h1>
    <p class="when">Generated {generated} by RayCrafter</p>
  </header>
  <div class="chips">{"".join(chips)}</div>
  <div class="grid">{"".join(cards)}</div>
  <footer>Click the ? beside any plot to see what it shows. Esc closes the note.</footer>
</main>

<div id="explain" role="dialog" hidden><button class="close" aria-label="Close">&times;</button>
  <h3></h3><div class="body"></div></div>
<script id="explanation-data" type="application/json">{explanation_json}</script>
<script>{read_asset("report.js")}</script>
{viewer_3d.scripts_html(model.build_scene())}
</body>
</html>
"""


def write_and_open_report(model: OpticalModel) -> Path:
    """Save the report to the temp folder with a time-stamped name and open it in the browser.

    Returns
    -------
    pathlib.Path
        Where the report was written.
    """
    report_path = (
        Path(tempfile.gettempdir()) / f"raycrafter_report_{datetime.now():%Y%m%d_%H%M%S}.html"
    )
    report_path.write_text(build_report_html(model), encoding="utf-8")
    webbrowser.open(report_path.as_uri())
    return report_path
