"""matplotlib figures for the HTML report, each returned as an inline SVG string.

Each ``draw_*`` function takes analysis results from ``raycrafter.core`` and draws the same
picture as the matching GUI panel, in the same colours (``raycrafter.palette``).
"""

from __future__ import annotations

import io
import re

import numpy as np
from matplotlib import rcParams
from matplotlib.figure import Figure

from raycrafter import palette
from raycrafter.core.analysis import (
    FIELD_CURVATURE_MAX_ANGLE,
    FastTrace,
    FieldCurvatureResult,
    MtfResult,
    RayFanResult,
    ThroughFocusResult,
    airy_radius,
)
from raycrafter.core.plot_geometry import (
    AXIS_Y_RANGE,
    AXIS_Z_RANGE,
    lens_surface_arcs,
    ray_fan_limit_um,
    rays_from_axis_start,
    spot_view_half_width_um,
)
from raycrafter.core.state import OpticalState

# Keep text as text in the SVG, so the page's own font is used and stays crisp.
rcParams["svg.fonttype"] = "none"

SQUARE_FIGURE_SIZE = (4.6, 4.4)  # inches; the three plots in the middle row of the page
WIDE_FIGURE_SIZE = (6.6, 3.9)  # the two plots in the bottom row
SQUARE_MARGINS = dict(left=0.17, right=0.97, top=0.97, bottom=0.14)
WIDE_MARGINS = dict(left=0.11, right=0.98, top=0.97, bottom=0.16)

_figure_counter = 0  # gives each SVG its own id prefix, so inline SVGs on one page cannot clash


def style_axes(axes) -> None:
    """Apply the report's shared look: light grid, no top and right edges, soft grey text."""
    axes.set_facecolor("none")
    axes.grid(True, color="#e9ecef", linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(palette.FAINT)
    axes.tick_params(colors="#868e96", labelsize=9, length=3)
    for label in (axes.xaxis.label, axes.yaxis.label):
        label.set_color("#495057")
        label.set_size(10)


def new_axes(figure_size: tuple, margins: dict, x_label: str, y_label: str):
    """A styled figure with one set of labelled axes; returns (figure, axes)."""
    figure = Figure(figsize=figure_size)
    axes = figure.add_subplot(111)
    figure.subplots_adjust(**margins)
    style_axes(axes)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    return figure, axes


def figure_to_inline_svg(figure: Figure) -> str:
    """Render a matplotlib figure to an SVG string that scales to the width of its container."""
    global _figure_counter
    _figure_counter += 1
    rcParams["svg.hashsalt"] = f"figure-{_figure_counter}"  # unique clip-path ids per figure

    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", transparent=True)
    svg_text = buffer.getvalue()

    # Drop the XML header and doctype, and the fixed width and height so CSS controls the size.
    svg_text = svg_text[svg_text.index("<svg") :]
    root_tag_end = svg_text.index(">")
    root_tag = re.sub(r'\s(width|height)="[^"]*"', "", svg_text[:root_tag_end])
    return root_tag + svg_text[root_tag_end:]


def draw_ray_diagram(state: OpticalState, fast_trace: FastTrace) -> str:
    """The side view: lenses, rays and the image plane."""
    figure, axes = new_axes(
        (13.0, 4.4), dict(left=0.05, right=0.99, top=0.97, bottom=0.14), "z (mm)", "y (mm)"
    )
    axes.set_xlim(*AXIS_Z_RANGE)
    axes.set_ylim(*AXIS_Y_RANGE)
    axes.axhline(0, color=palette.AXIS_LINE, linewidth=0.9, linestyle="--", zorder=1)

    for ray_z, ray_y in rays_from_axis_start(state, fast_trace.paths):
        axes.plot(ray_z, ray_y, color=palette.RAY, linewidth=1.3, zorder=3)

    # Lenses: a filled glass body between the first and last surface, every surface outlined.
    for lens in state.lenses:
        colour = palette.PLOSSL if lens.group_id is not None else palette.LENS
        edge_y, surface_arcs_z = lens_surface_arcs(lens, point_count=80)
        front_z, back_z = surface_arcs_z[0], surface_arcs_z[-1]

        axes.fill_betweenx(edge_y, front_z, back_z, color=colour, alpha=0.27, linewidth=0, zorder=2)
        for arc_z in surface_arcs_z:
            axes.plot(arc_z, edge_y, color=colour, linewidth=1.6, zorder=4)
        for rim in (0, -1):  # bottom and top rims
            axes.plot(
                [front_z[rim], back_z[rim]],
                [edge_y[rim], edge_y[rim]],
                color=colour,
                linewidth=1.6,
                zorder=4,
            )

    # Image plane: a green bar with a white-on-green caption over it.
    axes.axvline(state.image_z, color=palette.IMAGE, linewidth=3, zorder=5)
    axes.text(
        state.image_z,
        AXIS_Y_RANGE[1] - 1.5,
        "image",
        color="white",
        fontsize=9.5,
        ha="center",
        va="top",
        zorder=6,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": palette.IMAGE, "edgecolor": "none"},
    )
    return figure_to_inline_svg(figure)


def draw_spot_diagram(fast_trace: FastTrace) -> str:
    """Spot diagram (micrometres about each centroid) with the Airy circle."""
    spot, on_axis_spot = fast_trace.spot, fast_trace.on_axis_spot
    figure, axes = new_axes(SQUARE_FIGURE_SIZE, SQUARE_MARGINS, "x (um)", "y (um)")
    axes.set_aspect("equal", adjustable="box")
    axes.axhline(0, color="#dee2e6", linewidth=1)
    axes.axvline(0, color="#dee2e6", linewidth=1)

    # The 0 degree spot goes underneath so the current-angle spot lies on top.
    if on_axis_spot is not None:
        axes.scatter(
            on_axis_spot.x * 1000.0,
            on_axis_spot.y * 1000.0,
            s=9,
            color=palette.ON_AXIS,
            linewidths=0,
        )
    axes.scatter(spot.x * 1000.0, spot.y * 1000.0, s=9, color=palette.TANGENTIAL, linewidths=0)

    airy = airy_radius(spot.numerical_aperture)
    if airy is not None:
        airy_um = airy * 1000.0
        angles = np.linspace(0.0, 2.0 * np.pi, 160)
        axes.plot(
            airy_um * np.cos(angles),
            airy_um * np.sin(angles),
            color=palette.AIRY,
            linewidth=1.3,
            linestyle="--",
        )
    half_width = spot_view_half_width_um(spot, on_axis_spot)
    axes.set_xlim(-half_width, half_width)
    axes.set_ylim(-half_width, half_width)
    return figure_to_inline_svg(figure)


def draw_through_focus(result: ThroughFocusResult) -> str:
    """RMS radius against image plane offset, with the best focus and the Airy radius."""
    figure, axes = new_axes(
        SQUARE_FIGURE_SIZE, SQUARE_MARGINS, "image plane offset (mm)", "RMS radius (um)"
    )

    rms_current_um = result.rms_current * 1000.0
    curve_maximum_um = rms_current_um.max()
    if result.rms_on_axis is not None:
        rms_on_axis_um = result.rms_on_axis * 1000.0
        axes.plot(result.offsets, rms_on_axis_um, color=palette.ON_AXIS, linewidth=2)
        curve_maximum_um = max(curve_maximum_um, rms_on_axis_um.max())
    axes.plot(result.offsets, rms_current_um, color=palette.TANGENTIAL, linewidth=2)

    axes.axvline(0, color=palette.IMAGE, linewidth=2)
    axes.axvline(result.best_offset, color=palette.TANGENTIAL, linewidth=1, linestyle=":")

    # Show the Airy radius line unless it is far above the curve.
    y_maximum_um = curve_maximum_um
    if result.airy_radius is not None:
        airy_um = result.airy_radius * 1000.0
        axes.axhline(airy_um, color=palette.AIRY, linewidth=1.3, linestyle="--")
        y_maximum_um = max(curve_maximum_um, min(airy_um, 3.0 * curve_maximum_um))
    axes.set_xlim(result.offsets[0], result.offsets[-1])
    axes.set_ylim(0, y_maximum_um * 1.08)
    return figure_to_inline_svg(figure)


def draw_mtf(mtf: MtfResult) -> str:
    """Tangential and sagittal MTF with the diffraction limit."""
    figure, axes = new_axes(
        SQUARE_FIGURE_SIZE, SQUARE_MARGINS, "spatial frequency (cycles/mm)", "modulation"
    )
    axes.plot(
        mtf.frequency,
        mtf.diffraction_limit,
        color=palette.AIRY,
        linewidth=1.3,
        linestyle="--",
        label="diffraction limit",
    )
    axes.plot(
        mtf.frequency, mtf.tangential, color=palette.TANGENTIAL, linewidth=2, label="tangential"
    )
    axes.plot(mtf.frequency, mtf.sagittal, color=palette.SAGITTAL, linewidth=2, label="sagittal")
    axes.set_xlim(0, mtf.frequency[-1])
    axes.set_ylim(0, 1.02)
    axes.legend(frameon=False, fontsize=9, labelcolor=palette.TEXT)
    return figure_to_inline_svg(figure)


def draw_ray_fan(fan: RayFanResult) -> str:
    """Transverse ray aberration against pupil coordinate."""
    figure, axes = new_axes(
        WIDE_FIGURE_SIZE, WIDE_MARGINS, "pupil coordinate", "image plane error (um)"
    )
    axes.axhline(0, color=palette.FAINT, linewidth=1)
    axes.plot(
        fan.pupil,
        fan.tangential_error * 1000.0,
        color=palette.TANGENTIAL,
        linewidth=2,
        label="tangential",
    )
    axes.plot(
        fan.pupil,
        fan.sagittal_error * 1000.0,
        color=palette.SAGITTAL,
        linewidth=2,
        linestyle="--",
        label="sagittal",
    )

    largest_error_mm = max(np.abs(fan.tangential_error).max(), np.abs(fan.sagittal_error).max())
    limit_um = ray_fan_limit_um(largest_error_mm)
    axes.set_xlim(-1.02, 1.02)
    axes.set_ylim(-limit_um, limit_um)
    axes.legend(frameon=False, fontsize=9, labelcolor=palette.TEXT)
    return figure_to_inline_svg(figure)


def draw_field_curvature(curvature: FieldCurvatureResult, current_angle: float) -> str:
    """Best-focus shift against field angle; a dashed line marks the current beam angle."""
    figure, axes = new_axes(
        WIDE_FIGURE_SIZE, WIDE_MARGINS, "focus shift from image plane (mm)", "field angle (deg)"
    )
    axes.axvline(0, color=palette.IMAGE, linewidth=2)
    axes.axhline(abs(current_angle), color=palette.FAINT, linewidth=1, linestyle="--")
    axes.plot(
        curvature.tangential_shift,
        curvature.angles,
        color=palette.TANGENTIAL,
        linewidth=2,
        marker="o",
        markersize=4,
        label="tangential",
    )
    axes.plot(
        curvature.sagittal_shift,
        curvature.angles,
        color=palette.SAGITTAL,
        linewidth=2,
        linestyle="--",
        marker="s",
        markersize=4,
        label="sagittal",
    )

    largest_shift = max(
        np.abs(curvature.tangential_shift).max(), np.abs(curvature.sagittal_shift).max(), 0.5
    )
    axes.set_xlim(-largest_shift * 1.1, largest_shift * 1.1)
    axes.set_ylim(0, FIELD_CURVATURE_MAX_ANGLE * 1.02)
    axes.legend(frameon=False, fontsize=9, labelcolor=palette.TEXT)
    return figure_to_inline_svg(figure)
