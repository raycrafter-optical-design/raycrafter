"""The side view: the optical axis with its lenses, rays and image plane.

Lenses and the image plane can be dragged along the axis. Hovering a lens darkens it and
shows its part number; that lens is the target of the right-click menu and the keyboard
shortcuts (handled by the main window). The view edits the model directly when something
is dragged, then emits ``changed`` so the main window recomputes and redraws.

To insert a lens, the main window calls ``start_placement``: a dashed ghost of the lens then
follows the pointer along the axis, snapped to the nearest gap it fits in (red where it fits
nowhere). A left click puts it on the bench; Escape or a right click cancels.

A paraxial (ideal thin) lens is drawn as a line with arrowheads. It has no thickness, so it
can be hovered and dragged a few pixels either side of its line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QCursor, QFont, QPainter, QTextOption

from raycrafter import palette
from raycrafter.core.analysis import RayPaths
from raycrafter.core.model import OpticalModel
from raycrafter.core.plot_geometry import (
    AXIS_Y_RANGE,
    AXIS_Z_RANGE,
    lens_surface_arcs,
    rays_from_axis_start,
)
from raycrafter.core.state import PlacedLens
from raycrafter.gui.style import (
    PLOT_BOTTOM_GAP_PX,
    colour_with_alpha,
    join_with_gaps,
    scaled_font,
    style_plot,
)

MIN_IMAGE_GAP_MM = 1.0  # the image plane stays at least this far behind the last lens
IMAGE_MAX_Z = AXIS_Z_RANGE[1] - 5.0  # the image plane cannot be dragged closer to the edge
SIDE_PLOT_MIN_HEIGHT_PX = 240  # the side view is never squeezed below this height
SIDE_PLOT_TOP_GAP_PX = 8  # space between the top edge and the top of the y axis
SIDE_GRID_CELL_PX = 40  # grid cells are this many screen pixels on each side
TIMING_FONT_SCALE = 2 / 3  # timing text is this fraction of the axis number size
LENS_NAME_GAP_PX = 3  # gap between the top of a lens and its name tag
LENS_NAME_FONT_SCALE = 0.9  # name tag text is this fraction of the default text size
LENS_NAME_TRIM_PX = 2  # the name tag's box is this much tighter than its text item, top and bottom
LENS_NAME_CORNER_PX = 4  # corner radius of the name tag
PLOSSL_PREVIEW_PAD_MM = 1.5  # margin of the dashed rectangle round a proposed Plossl pair
IMAGE_HOVER_TOLERANCE_PX = 6  # a pointer this close to the image line counts as over it
LENS_FILL_ALPHA = 70  # fill opacity (0-255) of a lens on the bench
GHOST_FILL_ALPHA = 35  # fill opacity of a lens being placed
GHOST_Z_VALUE = 80  # the ghost is drawn above the lenses on the bench
PARAXIAL_LINE_WIDTH_PX = 3  # a paraxial lens is drawn as a line this thick
PARAXIAL_ARROW_SIZE_PX = 11  # size of the arrowheads at each end of a paraxial lens
PARAXIAL_GRAB_HALF_WIDTH_PX = 5  # a paraxial lens can be hovered and dragged this far either side


class LensNameTag(pg.TextItem):
    """White text on a solid, rounded, coloured box.

    pyqtgraph's own TextItem fill is a sharp-cornered rectangle, so the box is drawn here.
    """

    def __init__(self) -> None:
        super().__init__(color="w", anchor=(0.5, 1))
        self.box_colour = QColor(palette.LENS_HOVER)

    def paint(self, painter, *args) -> None:
        """Let TextItem set the painter up (it draws no fill itself), then draw the box."""
        super().paint(painter, *args)
        box = self.textItem.mapToParent(self.textItem.boundingRect()).boundingRect()
        box.adjust(0, LENS_NAME_TRIM_PX, 0, -LENS_NAME_TRIM_PX)  # the text item has extra margin
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.box_colour))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.drawRoundedRect(box, LENS_NAME_CORNER_PX, LENS_NAME_CORNER_PX)


class LensShape:
    """The pyqtgraph items that draw a lens.

    A catalogue lens is a filled glass body with its outline. A paraxial lens is a thick
    vertical line with an arrowhead at each end: pointing outwards for a positive lens and
    inwards for a negative one.

    On its own this is the ghost of a lens being placed; ``LensGraphics`` adds dragging for
    the lenses on the bench.
    """

    def __init__(self, plot_item: pg.PlotItem, paraxial: bool = False) -> None:
        self.plot_item = plot_item
        self.paraxial = paraxial
        self.outline = pg.PlotCurveItem(connect="finite")

        if paraxial:
            self.arrowheads = pg.ScatterPlotItem(
                size=PARAXIAL_ARROW_SIZE_PX,
                pen=pg.mkPen(None),
                brush=pg.mkBrush(palette.PARAXIAL_ARROW),
            )
            self.items: list = [self.outline, self.arrowheads]
        else:
            # The front and back outlines drive the fill; the outline curve draws the shape.
            self.front_curve = pg.PlotCurveItem()
            self.back_curve = pg.PlotCurveItem()
            self.fill = pg.FillBetweenItem(self.front_curve, self.back_curve)
            self.items = [self.fill, self.outline]
        for item in self.items:
            plot_item.addItem(item)

    def remove(self) -> None:
        """Take all of this lens's items off the plot."""
        for item in self.items:
            self.plot_item.removeItem(item)

    def set_visible(self, visible: bool) -> None:
        """Show or hide the lens."""
        for item in self.items:
            item.setVisible(visible)

    def is_visible(self) -> bool:
        """True if the lens is drawn."""
        return self.outline.isVisible()

    def set_z_value(self, z_value: float) -> None:
        """Stacking order: items with a higher value are drawn on top."""
        for item in self.items:
            item.setZValue(z_value)

    def draw(self, lens: PlacedLens, colour: str, dashed: bool = False) -> None:
        """Draw the lens body and outline at its current position and orientation.

        A dashed outline and fainter fill mark the ghost of a lens that is being placed.
        """
        line_style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
        if self.paraxial:
            self._draw_paraxial(lens, colour, line_style)
            return

        edge_y, surface_arcs_z = lens_surface_arcs(lens)

        # The glass body runs from the first surface to the last; internal cemented surfaces
        # are drawn as lines on top.
        front_z, back_z = surface_arcs_z[0], surface_arcs_z[-1]
        self.front_curve.setData(front_z, edge_y)
        self.back_curve.setData(back_z, edge_y)

        rim_segments_z = [[front_z[0], back_z[0]], [front_z[-1], back_z[-1]]]
        rim_segments_y = [[edge_y[0]] * 2, [edge_y[-1]] * 2]
        outline_z, outline_y = join_with_gaps(
            surface_arcs_z + rim_segments_z, [edge_y] * len(surface_arcs_z) + rim_segments_y
        )
        self.outline.setData(outline_z, outline_y)

        self.outline.setPen(pg.mkPen(colour, width=1.8, style=line_style))
        fill_alpha = GHOST_FILL_ALPHA if dashed else LENS_FILL_ALPHA
        self.fill.setBrush(pg.mkBrush(colour_with_alpha(colour, fill_alpha)))

    def _draw_paraxial(self, lens: PlacedLens, colour: str, line_style: Qt.PenStyle) -> None:
        """A vertical line the height of the lens, with an arrowhead at the top and bottom."""
        half_height = lens.semi_aperture
        self.outline.setData([lens.z, lens.z], [-half_height, half_height])
        self.outline.setPen(pg.mkPen(colour, width=PARAXIAL_LINE_WIDTH_PX, style=line_style))

        # pyqtgraph's "t1" is a triangle pointing up and "t" one pointing down.
        top_symbol, bottom_symbol = ("t", "t1") if lens.is_negative else ("t1", "t")
        self.arrowheads.setData(
            x=[lens.z, lens.z], y=[half_height, -half_height], symbol=[top_symbol, bottom_symbol]
        )


class LensGraphics(LensShape):
    """A lens on the bench: its shape plus an invisible region that lets it be dragged.

    The region spans the lens from front to back vertex. A paraxial lens has no thickness,
    so its region is widened by ``grab_margin_mm`` on each side to give the pointer
    something to hold.
    """

    def __init__(self, plot_item: pg.PlotItem, paraxial: bool = False) -> None:
        super().__init__(plot_item, paraxial)
        self.grab_margin_mm = 0.0

        # The region is invisible; it only provides whole-lens dragging along z. pyqtgraph
        # derives a hover brush from the normal one, and with a transparent brush that came
        # out solid black, so the hover brush is set explicitly (also transparent; the
        # hovered lens itself turns darker instead).
        transparent = pg.mkBrush(0, 0, 0, 0)
        self.region = pg.LinearRegionItem(values=(0, 1), movable=True, brush=transparent)
        self.region.setHoverBrush(transparent)
        for edge_line in self.region.lines:
            edge_line.setPen(pg.mkPen(None))
            edge_line.setMovable(False)
        self.items.append(self.region)
        plot_item.addItem(self.region)

    def update(self, lens: PlacedLens, colour: str) -> None:
        """Redraw the lens and move its drag region to match."""
        self.draw(lens, colour)
        if self.paraxial:
            self.grab_margin_mm = paraxial_grab_margin_mm(self.plot_item)
        self.region.blockSignals(True)
        self.region.setRegion((lens.z - self.grab_margin_mm, lens.back_z + self.grab_margin_mm))
        self.region.blockSignals(False)

    def requested_front_z(self) -> float:
        """Where the user has dragged the lens's front vertex to (mm)."""
        return float(self.region.getRegion()[0]) + self.grab_margin_mm


def paraxial_grab_margin_mm(plot_item: pg.PlotItem) -> float:
    """How far either side of a paraxial lens (mm) the pointer can be and still hold it."""
    mm_per_pixel = plot_item.vb.viewPixelSize()[0]
    return PARAXIAL_GRAB_HALF_WIDTH_PX * mm_per_pixel


@dataclass
class Placement:
    """A new lens that is following the pointer, waiting for a click to be put on the bench.

    Attributes
    ----------
    lens : PlacedLens
        The new lens (not yet in the model); its z is kept at the ghost's position.
    ghost : LensShape
        Its dashed outline on the plot.
    fits : bool
        True if the lens fits at the ghost's current position.
    """

    lens: PlacedLens
    ghost: LensShape
    fits: bool = False


def ghost_colour(placement: Placement) -> str:
    """Lens blue where the new lens fits; red where it does not."""
    return palette.LENS if placement.fits else palette.GHOST_BLOCKED


class SideView(pg.PlotWidget):
    """The optical axis, drawn to scale, with draggable lenses and image plane.

    Signals
    -------
    changed
        A lens or the image plane was dragged; the model has already been updated.
    lens_menu_requested(int, QPoint)
        The user right-clicked lens ``index`` at the given global screen position.
    placement_finished(bool)
        Placing a lens ended: True if it was put on the bench, False if cancelled.
    drag_finished
        The mouse was released after dragging a lens or the image plane.
    """

    changed = pyqtSignal()
    lens_menu_requested = pyqtSignal(int, QPoint)
    placement_finished = pyqtSignal(bool)
    drag_finished = pyqtSignal()

    def __init__(self, model: OpticalModel) -> None:
        super().__init__()
        self.model = model
        self.lens_graphics: list[LensGraphics] = []  # one entry per lens in the model
        self.hovered_index: int | None = None  # lens under the pointer
        self.image_line_locked = False  # True while "Lock focus" sets the image plane
        self.lens_drag_active = (
            False  # True from the first drag movement until the mouse is released
        )

        self.placement: Placement | None = None  # set while a new lens follows the pointer

        style_plot(self, x_label="z (mm)", y_label="y (mm)")
        side = self.getPlotItem()
        # With no title the y axis would run right up to the top border; a few pixels of top
        # margin keep the plot visibly inside the frame.
        side.layout.setContentsMargins(0, SIDE_PLOT_TOP_GAP_PX, 0, PLOT_BOTTOM_GAP_PX)
        side.showGrid(x=False, y=False)  # turn off the default grid; a square one is drawn below
        side.setXRange(*AXIS_Z_RANGE, padding=0)
        side.setYRange(*AXIS_Y_RANGE, padding=0)
        side.addLine(y=0, pen=pg.mkPen(palette.AXIS_LINE, width=1, style=Qt.PenStyle.DashLine))
        self.setMinimumHeight(SIDE_PLOT_MIN_HEIGHT_PX)

        self._add_items(side)

        side.vb.sigResized.connect(self.update_grid)
        self.image_line.sigPositionChanged.connect(self.on_image_dragged)
        # The signal passes the line itself; drag_finished takes no arguments.
        self.image_line.sigPositionChangeFinished.connect(lambda _line: self.drag_finished.emit())
        self.image_line.sigPositionChanged.connect(self.place_image_label)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.on_right_click)
        self.scene().sigMouseMoved.connect(self.on_hover)
        self.scene().sigMouseClicked.connect(self.on_scene_clicked)
        self.viewport().installEventFilter(self)

    def _add_items(self, side: pg.PlotItem) -> None:
        """Create the grid, rays, image line, captions and Plossl preview outline."""
        # Square grid: drawn by hand (not with showGrid) so its cells stay square on screen
        # even though the z and y axes cover very different distances in mm.
        self.grid_curve = pg.PlotCurveItem(
            pen=pg.mkPen(palette.SIDE_GRID, width=1), connect="finite"
        )
        self.grid_curve.setZValue(-10)  # behind the lenses and rays
        side.addItem(self.grid_curve)

        self.ray_curve = pg.PlotCurveItem(pen=pg.mkPen(palette.RAY, width=1.3), connect="finite")
        side.addItem(self.ray_curve)

        self.image_line = pg.InfiniteLine(
            angle=90,
            movable=True,
            pen=self._image_line_pen(),
            hoverPen=pg.mkPen(palette.IMAGE, width=3),
        )
        side.addItem(self.image_line)

        # The "image" caption is a plain text item at a fixed height, moved with the line.
        # (The line's built-in label re-anchors itself on every move, so the text bounced.)
        self.image_label = pg.TextItem("image", color="w", fill=palette.IMAGE, anchor=(0.5, 0))
        self.image_label.setZValue(40)  # above the line, so it cannot cut across the text
        side.addItem(self.image_label, ignoreBounds=True)
        self.place_image_label()

        # Trace and draw timings: small grey text in the top-right corner. Its font size is
        # set in showEvent, once the style sheet has fixed the size of the axis numbers.
        self.timing_label = pg.TextItem(color=palette.TIMING, anchor=(1, 0))
        z_span = AXIS_Z_RANGE[1] - AXIS_Z_RANGE[0]
        y_span = AXIS_Y_RANGE[1] - AXIS_Y_RANGE[0]
        self.timing_label.setPos(AXIS_Z_RANGE[1] - 0.01 * z_span, AXIS_Y_RANGE[1] - 0.04 * y_span)
        self.timing_label.setZValue(50)
        side.addItem(self.timing_label, ignoreBounds=True)

        # Dashed red rectangle shown while "Make Plossl" is highlighted in the lens menu.
        self.plossl_preview_curve = pg.PlotCurveItem(
            pen=pg.mkPen(palette.PLOSSL_PREVIEW, width=2, style=Qt.PenStyle.DashLine)
        )
        self.plossl_preview_curve.setZValue(60)
        self.plossl_preview_curve.hide()
        side.addItem(self.plossl_preview_curve, ignoreBounds=True)

        # Lens name tag: centred just above the hovered lens, text a little smaller than default.
        self.name_tag = LensNameTag()
        self.name_tag.setFont(scaled_font(self.name_tag.textItem.font(), LENS_NAME_FONT_SCALE))
        self.name_tag.setZValue(100)
        self.name_tag.hide()
        # The tag is a child of the plot item, not of its view box: the view box clips its
        # children to the data area, which cut the corners off the tag when a tall lens reached
        # near the top edge. Its position is therefore in plot-item pixels (see place_name_tag).
        self.name_tag.setParentItem(side)

    @staticmethod
    def _image_line_pen():
        """The image line's normal look: dashed green (solid when hovered)."""
        return pg.mkPen(palette.IMAGE, width=3, style=Qt.PenStyle.DashLine)

    # ------------------------------------------------------------------ drawing

    def rebuild_lens_graphics(self) -> None:
        """Recreate one set of graphics items per lens (after the lens list changes)."""
        for graphics in self.lens_graphics:
            graphics.remove()
        self.lens_graphics = []
        self.hovered_index = None

        side = self.getPlotItem()
        for lens in self.model.state.lenses:
            graphics = LensGraphics(side, paraxial=lens.is_paraxial)
            # The signal passes the region as an argument; swallow it and bind our own object.
            graphics.region.sigRegionChanged.connect(
                lambda _region, graphics=graphics: self.on_lens_dragged(graphics)
            )
            graphics.region.sigRegionChangeFinished.connect(self.on_lens_drag_finished)
            self.lens_graphics.append(graphics)

    def update_view(self, paths: RayPaths) -> None:
        """Redraw the rays, every lens and the image line from the current model state."""
        rays = rays_from_axis_start(self.model.state, paths)
        ray_z, ray_y = join_with_gaps([z for z, _ in rays], [y for _, y in rays])
        self.ray_curve.setData(ray_z, ray_y)

        # Move the draggable image line without re-triggering its own change signal.
        self.image_line.blockSignals(True)
        self.image_line.setValue(self.model.state.image_z)
        self.image_line.blockSignals(False)
        self.place_image_label()

        self.refresh_lens_colours()

    def set_timings(self, trace_ms: float, draw_ms: float) -> None:
        """Show the trace and draw times, right-aligned in the top-right corner."""
        # TextItem only aligns text through HTML, which then also needs the colour.
        self.timing_label.setHtml(
            f'<div style="text-align: right; color: {palette.TIMING}">'
            f"trace {trace_ms:.0f} ms<br>draw {draw_ms:.0f} ms</div>"
        )
        # Alignment only works once the text box has a fixed width, so fix it to the text's
        # own. Wrapping is off and a few pixels of slack added, because idealWidth() can come
        # out slightly narrower than the text needs, which wrapped it onto many lines.
        document = self.timing_label.textItem.document()
        text_option = document.defaultTextOption()
        text_option.setWrapMode(QTextOption.WrapMode.NoWrap)
        document.setDefaultTextOption(text_option)
        self.timing_label.textItem.setTextWidth(document.idealWidth() + 12)

    def update_grid(self, *_args) -> None:
        """Redraw the grid so every cell is the same size on screen (a true square).

        The grid is anchored at the origin (z = 0, y = 0). Its spacing is set in data units
        per axis so that each cell spans SIDE_GRID_CELL_PX pixels both across and up-down.
        This runs whenever the plot is resized.
        """
        view_box = self.getPlotItem().vb
        width_px, height_px = view_box.width(), view_box.height()
        if width_px <= 0 or height_px <= 0:
            return

        z_min, z_max = AXIS_Z_RANGE
        y_min, y_max = AXIS_Y_RANGE
        cell_z_mm = SIDE_GRID_CELL_PX * (z_max - z_min) / width_px
        cell_y_mm = SIDE_GRID_CELL_PX * (y_max - y_min) / height_px

        # Lines of constant z (vertical) and constant y (horizontal), each as two end points.
        x_points, y_points = [], []
        z_steps = np.arange(np.ceil(z_min / cell_z_mm), np.floor(z_max / cell_z_mm) + 1)
        for z_position in z_steps * cell_z_mm:
            x_points += [z_position, z_position, np.nan]
            y_points += [y_min, y_max, np.nan]

        y_steps = np.arange(np.ceil(y_min / cell_y_mm), np.floor(y_max / cell_y_mm) + 1)
        for y_position in y_steps * cell_y_mm:
            x_points += [z_min, z_max, np.nan]
            y_points += [y_position, y_position, np.nan]

        self.grid_curve.setData(np.array(x_points), np.array(y_points))

    def show_plossl_preview(self, lens_index: int) -> None:
        """Outline, dashed red, the lens and the neighbour it would form a Plossl pair with."""
        partner_index = self.model.plossl_partner(lens_index)
        if partner_index is None:
            return
        lenses = self.model.state.lenses
        pair = [lenses[lens_index], lenses[partner_index]]
        z_low = min(lens.z for lens in pair) - PLOSSL_PREVIEW_PAD_MM
        z_high = max(lens.back_z for lens in pair) + PLOSSL_PREVIEW_PAD_MM
        half_height = max(lens.semi_aperture for lens in pair) + PLOSSL_PREVIEW_PAD_MM

        self.plossl_preview_curve.setData(
            [z_low, z_high, z_high, z_low, z_low],
            [-half_height, -half_height, half_height, half_height, -half_height],
        )
        self.plossl_preview_curve.show()

    def hide_plossl_preview(self) -> None:
        """Remove the dashed Plossl rectangle."""
        self.plossl_preview_curve.hide()

    def showEvent(self, event) -> None:
        """Once shown, size the timing text at 2/3 of the axis numbers (set by the style sheet)."""
        super().showEvent(event)
        axis_font = self.getPlotItem().getAxis("bottom").style["tickFont"] or self.viewport().font()
        self.timing_label.setFont(scaled_font(QFont(axis_font), TIMING_FONT_SCALE))

    # ------------------------------------------------------------- hover and colours

    def lens_index_at(self, view_point: QPointF) -> int | None:
        """Index of the (frontmost) lens whose outline contains the point, or None.

        A paraxial lens is a line with no thickness, so it counts as under the pointer a few
        pixels either side.
        """
        grab_margin_mm = paraxial_grab_margin_mm(self.getPlotItem())
        found_index = None
        for index, lens in enumerate(self.model.state.lenses):
            margin_mm = grab_margin_mm if lens.is_paraxial else 0.0
            if (
                lens.z - margin_mm <= view_point.x() <= lens.back_z + margin_mm
                and abs(view_point.y()) <= lens.semi_aperture
            ):
                found_index = index
        return found_index

    def lens_colour(self, index: int) -> str:
        """Blue (purple in a Plossl pair), darker when it, or its pair, is under the pointer."""
        in_pair = self.model.state.lenses[index].group_id is not None
        # A Plossl pair moves as one, so it is highlighted as one.
        hovered_group = []
        if self.hovered_index is not None:
            hovered_group = self.model.group_indices(self.hovered_index)
        if index in hovered_group:
            return palette.PLOSSL_HOVER if in_pair else palette.LENS_HOVER
        return palette.PLOSSL if in_pair else palette.LENS

    def refresh_lens_colours(self) -> None:
        """Redraw the lenses in their current colours (no re-trace needed), and the name tag."""
        for index, (lens, graphics) in enumerate(zip(self.model.state.lenses, self.lens_graphics)):
            graphics.update(lens, self.lens_colour(index))
        self.place_name_tag()

    def set_hovered_lens(self, lens_index: int | None) -> None:
        """Remember which lens the pointer is over and recolour the lenses if that changed."""
        if lens_index != self.hovered_index:
            self.hovered_index = lens_index
            self.refresh_lens_colours()

    def place_name_tag(self) -> None:
        """Show the hovered lens's part number centred above it, a few pixels clear of its top.

        The tag's box is a shade darker than the lens outline. It is hidden when no lens is
        hovered.
        """
        # While a lens is being placed, the tag names the ghost instead.
        if self.placement is not None:
            if self.placement.ghost.is_visible():
                self._show_name_tag(self.placement.lens, ghost_colour(self.placement))
            else:
                self.name_tag.hide()
            return

        # The tag is not drawn while a lens is being dragged: it would flicker as the hover
        # state changes under the moving lens. It reappears when the drag ends.
        if self.hovered_index is None or self.lens_drag_active:
            self.name_tag.hide()
            return

        lens = self.model.state.lenses[self.hovered_index]
        self._show_name_tag(lens, self.lens_colour(self.hovered_index))

    def _show_name_tag(self, lens: PlacedLens, lens_colour: str) -> None:
        """Put the name tag above ``lens``, in a box a shade darker than ``lens_colour``."""
        plot_item = self.getPlotItem()
        lens_top_centre = plot_item.mapFromScene(
            plot_item.vb.mapViewToScene(QPointF(0.5 * (lens.z + lens.back_z), lens.semi_aperture))
        )

        # The drawn box is LENS_NAME_TRIM_PX inside the text item at top and bottom, so allow
        # for that when leaving the gap above the lens; never let it poke out of the plot top.
        tag_height = self.name_tag.boundingRect().height()
        tag_bottom = max(
            lens_top_centre.y() - LENS_NAME_GAP_PX + LENS_NAME_TRIM_PX,
            tag_height - LENS_NAME_TRIM_PX + 2.0,
        )

        self.name_tag.setText(lens.part_number)
        self.name_tag.box_colour = QColor(lens_colour).darker(125)
        self.name_tag.setPos(lens_top_centre.x(), tag_bottom)
        self.name_tag.show()

    def is_over_image_line(self, scene_position: QPointF) -> bool:
        """True if the pointer is within a few pixels of the image line and it can be dragged."""
        if self.image_line_locked:
            return False
        view_box = self.getPlotItem().vb
        line_scene_x = view_box.mapViewToScene(QPointF(self.image_line.value(), 0.0)).x()
        return abs(scene_position.x() - line_scene_x) <= IMAGE_HOVER_TOLERANCE_PX

    def place_image_label(self, *_args) -> None:
        """Put the "image" caption just inside the top of the plot, centred on the image line."""
        y_span = AXIS_Y_RANGE[1] - AXIS_Y_RANGE[0]
        self.image_label.setPos(self.image_line.value(), AXIS_Y_RANGE[1] - 0.01 * y_span)

    def on_hover(self, scene_position: QPointF) -> None:
        """Track the lens under the pointer; show a left/right arrow over anything draggable.

        While a lens is being placed, move its ghost to the pointer instead.
        """
        if self.placement is not None:
            self.move_ghost(self.placement, scene_position)
            return

        view_point = self.getPlotItem().vb.mapSceneToView(scene_position)
        hovered_index = self.lens_index_at(view_point)
        self.set_hovered_lens(hovered_index)

        if hovered_index is not None or self.is_over_image_line(scene_position):
            self.viewport().setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.viewport().unsetCursor()
        self.place_name_tag()

    def eventFilter(self, watched, event) -> bool:
        """Clear the hover when the pointer leaves the view (no mouse-move event is sent then)."""
        if watched is self.viewport() and event.type() == QEvent.Type.Leave:
            self.set_hovered_lens(None)
            if self.placement is not None:
                self.placement.ghost.set_visible(False)
                self.place_name_tag()
            else:
                self.viewport().unsetCursor()
        return super().eventFilter(watched, event)

    # ------------------------------------------------------------ placing a new lens

    def start_placement(self, new_lens: PlacedLens) -> None:
        """Let the user put ``new_lens`` on the axis: its ghost follows the pointer.

        The lens is not in the model until the user clicks. Lenses on the bench cannot be
        dragged meanwhile, so a click near one places the new lens rather than moving it.
        """
        self.cancel_placement()  # in case a previous placement is still going on
        ghost = LensShape(self.getPlotItem(), paraxial=new_lens.is_paraxial)
        ghost.set_z_value(GHOST_Z_VALUE)
        ghost.set_visible(False)  # shown once the pointer is over the plot
        self.placement = Placement(lens=new_lens, ghost=ghost)

        for graphics in self.lens_graphics:
            graphics.region.setMovable(False)
        self.set_hovered_lens(None)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)

        # If the pointer is already over the plot, show the ghost there straight away.
        pointer_in_view = self.viewport().mapFromGlobal(QCursor.pos())
        if self.viewport().rect().contains(pointer_in_view):
            self.move_ghost(self.placement, self.mapToScene(pointer_in_view))

    def move_ghost(self, placement: Placement, scene_position: QPointF) -> None:
        """Centre the ghost on the pointer's z, snapped into the nearest gap it fits in.

        The ghost is hidden while the pointer is outside the plotting area (e.g. over an
        axis). Where the lens fits nowhere, the ghost is drawn red under the pointer.
        """
        view_box = self.getPlotItem().vb
        if not view_box.sceneBoundingRect().contains(scene_position):
            placement.ghost.set_visible(False)
            self.place_name_tag()
            return

        lens = placement.lens
        pointer_z = view_box.mapSceneToView(scene_position).x()
        requested_front_z = pointer_z - lens.thickness / 2.0
        free_front_z = self.model.free_front_z(
            lens.thickness, requested_front_z, max_back_z=IMAGE_MAX_Z - MIN_IMAGE_GAP_MM
        )
        placement.fits = free_front_z is not None
        lens.z = requested_front_z if free_front_z is None else free_front_z

        placement.ghost.draw(lens, ghost_colour(placement), dashed=True)
        placement.ghost.set_visible(True)
        self.place_name_tag()

    def on_scene_clicked(self, event) -> None:
        """While placing: a left click puts the lens where its ghost is, if it fits there."""
        placement = self.placement
        if placement is None or event.button() != Qt.MouseButton.LeftButton:
            return
        self.move_ghost(placement, event.scenePos())  # the ghost may lag a fast click
        if not (placement.fits and placement.ghost.is_visible()):
            return
        self._end_placement(placement)
        self.model.insert_lens(placement.lens, max_image_z=IMAGE_MAX_Z)
        self.rebuild_lens_graphics()
        self.changed.emit()
        self.placement_finished.emit(True)

    def cancel_placement(self) -> None:
        """Stop placing a lens without adding it (Escape or right click)."""
        if self.placement is not None:
            self._end_placement(self.placement)
            self.placement_finished.emit(False)

    def _end_placement(self, placement: Placement) -> None:
        """Remove the ghost and make the lenses on the bench draggable again."""
        placement.ghost.remove()
        self.placement = None
        for graphics in self.lens_graphics:
            graphics.region.setMovable(True)
        self.viewport().unsetCursor()
        self.name_tag.hide()

    # ------------------------------------------------------------ user interaction

    def set_image_line_locked(self, locked: bool) -> None:
        """Lock focus on: the image plane is set by the software, so the line cannot be dragged.

        A line that is not movable gets no hover highlight, so it stays dashed; the arrow
        cursor is suppressed in ``is_over_image_line``.
        """
        self.image_line_locked = locked
        self.image_line.setMovable(not locked)
        self.image_line.setPen(self._image_line_pen())  # clears any hover look
        if locked:
            self.viewport().unsetCursor()

    def on_right_click(self, _widget_position) -> None:
        """Ask for the lens menu if the pointer is over a lens; cancel placing a lens."""
        if self.placement is not None:
            self.cancel_placement()
            return
        global_position = QCursor.pos()
        scene_position = self.mapToScene(self.viewport().mapFromGlobal(global_position))
        lens_index = self.lens_index_at(self.getPlotItem().vb.mapSceneToView(scene_position))
        if lens_index is not None:
            self.lens_menu_requested.emit(lens_index, global_position)

    def on_lens_dragged(self, graphics: LensGraphics) -> None:
        """A lens was dragged: move it (with its Plossl partner), or swap it with a neighbour."""
        self.lens_drag_active = True
        index = self.lens_graphics.index(graphics)
        new_order = self.model.move_lens(index, graphics.requested_front_z())
        if new_order is not None:
            # The lens list was reordered; reorder the graphics (and the hover marker) to match,
            # so the item being dragged keeps showing the lens it belongs to.
            self.lens_graphics = [self.lens_graphics[old_index] for old_index in new_order]
            self.hovered_index = new_order.index(index)
        self.changed.emit()

    def on_lens_drag_finished(self) -> None:
        """The mouse was released after dragging a lens: bring the name tag back."""
        self.lens_drag_active = False
        self.place_name_tag()
        self.drag_finished.emit()

    def is_dragging(self) -> bool:
        """True while the mouse is holding a lens or the image plane."""
        return self.lens_drag_active or self.image_line.moving

    def on_image_dragged(self) -> None:
        """The image line was dragged: keep it behind the last lens and inside the view."""
        state = self.model.state
        minimum_z = state.last_back_z() + MIN_IMAGE_GAP_MM if state.lenses else AXIS_Z_RANGE[0]
        state.image_z = float(np.clip(self.image_line.value(), minimum_z, IMAGE_MAX_Z))
        self.changed.emit()
