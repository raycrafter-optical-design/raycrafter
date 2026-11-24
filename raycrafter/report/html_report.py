"""Build a self-contained HTML report of the current lens system.

The report has the ray diagram across the top and the five analysis plots below it (spot
diagram, through-focus, MTF, ray fan, field curvature), with a row of summary tiles. Every
plot has a "?" button that opens a note explaining how to read it. Plots are inline SVG and
the style sheet and script (``assets/report.css``, ``assets/report.js``) are copied into the
page, so the result is a single HTML file that needs no other files or internet connection.

Example
-------
>>> from raycrafter.core import OpticalModel
>>> from raycrafter.report import build_report_html
>>> page = build_report_html(OpticalModel())  # doctest: +SKIP
"""

from __future__ import annotations

import html
import json
import tempfile
import webbrowser
from datetime import datetime
from importlib import resources
from pathlib import Path

from raycrafter import palette
from raycrafter.core.analysis import airy_radius
from raycrafter.core.model import OpticalModel
from raycrafter.core.plot_geometry import f_number_text, rms_radius_text
from raycrafter.report import figures
from raycrafter.report.explanations import EXPLANATIONS


def read_asset(file_name: str) -> str:
    """Text of one of the report's packaged asset files (CSS or JavaScript)."""
    return resources.files("raycrafter.report").joinpath("assets", file_name).read_text("utf-8")


def card_html(
    key: str, title: str, subtitle_html: str, svg: str, width_class: str, with_help: bool = True
) -> str:
    """One card: a heading, a one-line subtitle, the "?" button (optional) and the plot."""
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
        f"{help_button}</div>{svg}</section>"
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

    spot, on_axis_spot = fast_trace.spot, fast_trace.on_axis_spot
    summary = fast_trace.summary
    airy = airy_radius(spot.numerical_aperture)

    chips = [
        chip_html("Beam", f"{state.beam_diameter:.0f} mm"),
        chip_html("Angle", f"{state.field_angle:+.1f}°"),
        chip_html("Vergence", f"{state.beam_vergence:+.2f} D"),
        chip_html("Image plane", f"z = {state.image_z:.1f} mm"),
        chip_html("RMS radius", rms_radius_text(spot, "µm")),
    ]
    if summary is not None:
        chips += [
            chip_html("EFL", f"{summary.effective_focal_length:.1f} mm"),
            chip_html("BFL", f"{summary.back_focal_length:.1f} mm"),
        ]
    f_number = model.compute_f_number()
    if f_number is not None:
        chips.append(chip_html("f-number", f_number_text(f_number)))
    if airy is not None:
        chips.append(chip_html("Airy radius", f"{airy * 1000:.1f} µm"))

    lens_names = ", ".join(
        html.escape(lens.part_number) + (" (flipped)" if lens.flipped else "")
        for lens in state.lenses
    )
    ray_subtitle = f"{lens_names} &middot; {vergence_words(state.beam_vergence)} beam"

    if on_axis_spot is None:
        spot_subtitle = f"RMS radius {rms_radius_text(spot, 'µm')}"
    else:
        spot_subtitle = (
            f'<span style="color:{palette.TANGENTIAL}">{state.field_angle:+.1f}°: '
            f"{rms_radius_text(spot, 'µm')}</span> &nbsp; "
            f'<span style="color:{palette.ON_AXIS}">0°: '
            f"{rms_radius_text(on_axis_spot, 'µm')}</span>"
        )

    # "+ 0.0" turns a rounded -0.000 into 0.000.
    best_focus_text = f"{round(through_focus.best_offset, 3) + 0.0:+.3f}"
    angle_text = f"at {state.field_angle:+.1f}°"
    cards = [
        card_html(
            "rays",
            "Ray diagram",
            ray_subtitle,
            figures.draw_ray_diagram(state, fast_trace),
            "span-6",
            with_help=False,
        ),
        card_html(
            "spot", "Spot diagram", spot_subtitle, figures.draw_spot_diagram(fast_trace), "span-2"
        ),
        card_html(
            "through_focus",
            "Through-focus",
            f"Best focus {best_focus_text} mm from the image plane",
            figures.draw_through_focus(through_focus),
            "span-2",
        ),
        card_html("mtf", "MTF", angle_text, figures.draw_mtf(mtf), "span-2"),
        card_html("ray_fan", "Ray fan", angle_text, figures.draw_ray_fan(ray_fan), "span-3"),
        card_html(
            "field",
            "Field curvature",
            "Best focus against field angle",
            figures.draw_field_curvature(curvature, state.field_angle),
            "span-3",
        ),
    ]

    explanation_data = {
        key: {"title": title, "paragraphs": paragraphs}
        for key, (title, paragraphs) in EXPLANATIONS.items()
    }
    # "</" inside a script block would end it early, so escape it in the JSON.
    explanation_json = json.dumps(explanation_data).replace("</", "<\\/")

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
