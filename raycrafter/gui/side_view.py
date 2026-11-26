"""The side view: the optical axis with its lenses, rays and image plane.

Each field's rays are drawn in the field's colour.

Lenses and the image plane can be dragged along the axis. Hovering a lens darkens it and
shows its part number; that lens is the target of the right-click menu and the keyboard
shortcuts (handled by the main window). Clicking a lens selects it (it stays dark), so
that its settings can be edited in the context bar; clicking empty space clears that.
The view edits the model directly when something is dragged, then emits ``changed`` so the
main window recomputes and redraws.

To insert a lens (or a user part), the main window calls ``start_placement``: a dashed ghost
of the lens then
follows the pointer along the axis, snapped to the nearest gap it fits in (red where it fits
nowhere). A left click puts it on the bench; Escape or a right click cancels.

The air gaps are dimensioned along the bottom (``raycrafter.gui.gap_dimensions``). Clicking
a dimension opens a box in which to type the gap; right-clicking it asks the main window for
the menu that locks or unlocks it. Lenses joined by locked gaps are dragged, and darkened on
hover, as one. While the gap to the image plane is locked, the image line cannot be dragged.

The source plane, where every field is defined, is a dashed line captioned "source". While
its gap to the first lens is locked (the default, at 0 mm) it moves with the first lens;
right-clicking its caption unlocks it, and it can then be dragged or its gap typed. The
dimension of that gap is left out while it is locked at 0 mm, as it says nothing then. When
the fields are points of a finite object, the source plane is the object plane and is
captioned "object" instead.

The axes have no titles: a small glyph in the bottom-left corner (``raycrafter.gui.axis_glyph``)
shows that z runs right and y up, and every length on the bench is in mm.

A paraxial (ideal thin) lens is drawn as a line with arrowheads. It has no thickness, so it
can be hovered and dragged a few pixels either side of its line.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QFont, QTextOption

from raycrafter import palette
from raycrafter.core.analysis import RayPaths
from raycrafter.core.gaps import MIN_IMAGE_GAP_MM, SOURCE_GAP_INDEX, AirGap
from raycrafter.core.model import OpticalModel
from raycrafter.core.plot_geometry import (
    DEFAULT_AXIS_Y_RANGE,
    DEFAULT_AXIS_Z_RANGE,
    IMAGE_EDGE_GAP_MM,
    axis_ranges,
    ray_polylines,
    source_caption,
)
from raycrafter.core.state import AXIS_START_Z, PlacedLens
from raycrafter.gui.axis_glyph import AxisGlyph
from raycrafter.gui.gap_dimensions import EDITABLE, FIXED, LOCKED, GapDimensions, GapEditor
from raycrafter.gui.lens_shapes import (
    LENS_NAME_TRIM_PX,
    LensGraphics,
    LensNameTag,
    LensShape,
    Placement,
    ghost_colour,
    paraxial_grab_margin_mm,
)
from raycrafter.gui.style import PLOT_BOTTOM_GAP_PX, join_with_gaps, scaled_font, style_plot

SIDE_PLOT_MIN_HEIGHT_PX = 240  # the side view is never squeezed below this height
SIDE_PLOT_TOP_GAP_PX = 8  # space between the top edge and the top of the y axis
SIDE_GRID_CELL_PX = 40  # grid cells are this many screen pixels on each side
TIMING_FONT_SCALE = 2 / 3  # timing text is this fraction of the default axis number size
TICK_FONT_SCALE = 0.9  # the axis numbers are a little smaller than the default
LENS_NAME_GAP_PX = 3  # gap between the top of a lens and its name tag
LENS_NAME_FONT_SCALE = 0.9  # name tag text is this fraction of the default text size
IMAGE_HOVER_TOLERANCE_PX = 6  # a pointer this close to the image line counts as over it
GHOST_Z_VALUE = 80  # the ghost is drawn above the lenses on the bench
RAY_WIDTH_PX = 1.3
HIGHLIGHTED_RAY_WIDTH_PX = 3.0  # the rays of the field selected in the bench viewer


class SideView(pg.PlotWidget):
    """The optical axis, drawn to scale, with draggable lenses and image plane.

    The view shows the whole design: ``fit_to_design`` widens it for a design too long or
    tall for the default range (see ``plot_geometry.axis_ranges``). It is called by the main
    window after each change, but not during a drag, so the axes stay still under the mouse.

    Signals
    -------
    changed
        A lens or the image plane was dragged, a lens was placed or a gap was typed; the
        model has already been updated.
    message_requested(str)
        Something to tell the user, e.g. why a gap cannot be changed.
    lens_menu_requested(int, QPoint)
        The user right-clicked lens ``index`` at the given global screen position.
    gap_menu_requested(int, QPoint)
        The user right-clicked the dimension of the gap behind lens ``index``.
    placement_finished(bool)
        Placing a lens ended: True if it was put on the bench, False if cancelled.
    drag_finished
        The mouse was released after dragging a lens or the image plane.
    lens_selected(object)
        The selected lens changed: its index, or None when no lens is selected.
    """

    changed = pyqtSignal()
    lens_menu_requested = pyqtSignal(int, QPoint)
    gap_menu_requested = pyqtSignal(int, QPoint)
    placement_finished = pyqtSignal(bool)
    drag_finished = pyqtSignal()
    message_requested = pyqtSignal(str)
    lens_selected = pyqtSignal(object)

    def __init__(self, model: OpticalModel) -> None:
        super().__init__()
        self.model = model
        self.lens_graphics: list[LensGraphics] = []  # one entry per lens in the model
        self.hovered_index: int | None = None  # lens under the pointer
        self.selected_index: int | None = None  # lens last clicked (see set_selected_lens)
        # The lens and field picked in the bench viewer, drawn highlighted (see set_highlight).
        self.highlighted_lens_index: int | None = None
        self.highlighted_field_index: int | None = None
        self.image_line_locked = False  # True while "Lock focus" sets the image plane
        # True while the image line cannot be dragged: "Lock focus" is on or the gap to the
        # image plane is locked (see update_image_line_movable).
        self.image_line_fixed = False
        self.lens_drag_active = (
            False  # True from the first drag movement until the mouse is released
        )

        self.placement: Placement | None = None  # set while new lenses follow the pointer
        self.last_placed_lenses: list[PlacedLens] = []  # put on the bench by the last click
        self.gap_editor: GapEditor | None = None  # set while a gap is being typed

        # The (min, max) of each axis, in mm; changed only by fit_to_design.
        self.z_range: tuple[float, float] = DEFAULT_AXIS_Z_RANGE
        self.y_range: tuple[float, float] = DEFAULT_AXIS_Y_RANGE

        # No axis titles: the small glyph in the corner (AxisGlyph) shows which way z and y
        # run, and every length on the bench is in mm. The titles took a lot of room.
        style_plot(self)
        side = self.getPlotItem()
        side.getAxis("bottom").showLabel(False)
        side.getAxis("left").showLabel(False)
        self.tick_fonts_scaled = False  # see showEvent
        # With no title the y axis would run right up to the top border; a few pixels of top
        # margin keep the plot visibly inside the frame.
        side.layout.setContentsMargins(0, SIDE_PLOT_TOP_GAP_PX, 0, PLOT_BOTTOM_GAP_PX)
        side.showGrid(x=False, y=False)  # turn off the default grid; a square one is drawn below
        side.setXRange(*self.z_range, padding=0)
        side.setYRange(*self.y_range, padding=0)
        side.addLine(y=0, pen=pg.mkPen(palette.AXIS_LINE, width=1, style=Qt.PenStyle.DashLine))
        self.setMinimumHeight(SIDE_PLOT_MIN_HEIGHT_PX)

        self._add_items(side)

        side.vb.sigResized.connect(self.update_grid)
        side.vb.sigResized.connect(self.on_view_resized)
        side.vb.sigResized.connect(self.axis_glyph.place)
        self.image_line.sigPositionChanged.connect(self.on_image_dragged)
        # The signal passes the line itself; drag_finished takes no arguments.
        self.image_line.sigPositionChangeFinished.connect(lambda _line: self.drag_finished.emit())
        self.image_line.sigPositionChanged.connect(self.place_image_label)
        self.source_line.sigPositionChanged.connect(self.on_source_dragged)
        self.source_line.sigPositionChangeFinished.connect(lambda _line: self.drag_finished.emit())
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.on_right_click)
        self.scene().sigMouseMoved.connect(self.on_hover)
        self.scene().sigMouseClicked.connect(self.on_scene_clicked)
        self.viewport().installEventFilter(self)

    def _add_items(self, side: pg.PlotItem) -> None:
        """Create the grid, rays, image line, captions and lens name tag."""
        # Square grid: drawn by hand (not with showGrid) so its cells stay square on screen
        # even though the z and y axes cover very different distances in mm.
        self.grid_curve = pg.PlotCurveItem(
            pen=pg.mkPen(palette.SIDE_GRID, width=1), connect="finite"
        )
        self.grid_curve.setZValue(-10)  # behind the lenses and rays
        side.addItem(self.grid_curve)

        self.axis_glyph = AxisGlyph(side)

        self.ray_curves: list[pg.PlotCurveItem] = []  # one per field, made in update_view
        self.gap_dimensions = GapDimensions(side)

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

        # The source plane: dashed, and captioned at the top as the image plane is.
        self.source_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(palette.SOURCE, width=2, style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen(palette.SOURCE, width=2),
        )
        side.addItem(self.source_line)
        self.source_label = pg.TextItem("source", color="w", fill=palette.SOURCE, anchor=(0.5, 0))
        self.source_label.setZValue(40)
        side.addItem(self.source_label, ignoreBounds=True)

        # Trace and draw timings: small grey text in the top-right corner. Its font size is
        # set in showEvent, once the style sheet has fixed the size of the axis numbers.
        self.timing_label = pg.TextItem(color=palette.TIMING, anchor=(1, 0))
        self.place_timing_label()
        self.timing_label.setZValue(50)
        side.addItem(self.timing_label, ignoreBounds=True)

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

    def fit_to_design(self) -> None:
        """Set the axis ranges to show the whole design, and move the grid and labels to match.

        Nothing is redrawn if the ranges are unchanged, which is the usual case.
        """
        z_range, y_range = axis_ranges(self.model.state)
        if (z_range, y_range) == (self.z_range, self.y_range):
            return
        self.z_range, self.y_range = z_range, y_range
        side = self.getPlotItem()
        side.setXRange(*z_range, padding=0)
        side.setYRange(*y_range, padding=0)
        self.update_grid()
        self.place_timing_label()
        self.place_image_label()

    def image_max_z(self) -> float:
        """The furthest z (mm) the image plane can be dragged to: just inside the right edge."""
        return self.z_range[1] - IMAGE_EDGE_GAP_MM

    def rebuild_lens_graphics(self) -> None:
        """Recreate one set of graphics items per lens (after the lens list changes)."""
        self.close_gap_editor()  # its lens index may now mean another lens
        for graphics in self.lens_graphics:
            graphics.remove()
        self.lens_graphics = []
        self.hovered_index = None
        self.highlighted_lens_index = None  # the lens list changed, so the index may be wrong
        self.set_selected_lens(None)  # likewise

        side = self.getPlotItem()
        for lens in self.model.state.lenses:
            graphics = LensGraphics(side, paraxial=lens.is_paraxial)
            # The signal passes the region as an argument; swallow it and bind our own object.
            graphics.region.sigRegionChanged.connect(
                lambda _region, graphics=graphics: self.on_lens_dragged(graphics)
            )
            graphics.region.sigRegionChangeFinished.connect(self.on_lens_drag_finished)
            self.lens_graphics.append(graphics)

    def update_view(self, field_paths: list[RayPaths]) -> None:
        """Redraw the rays, lenses, gaps and image line from the current model state.

        Parameters
        ----------
        field_paths : list of RayPaths
            The side-view rays of each field, in the order of the model's fields.
        """
        fields = self.model.state.fields
        if len(fields) != len(self.ray_curves):
            # Fields were added or removed, so the highlighted index may be another field now.
            self.highlighted_field_index = None
        self._match_ray_curve_count(len(field_paths))
        for paths, curve in zip(field_paths, self.ray_curves):
            rays = ray_polylines(paths)
            ray_z, ray_y = join_with_gaps([z for z, _ in rays], [y for _, y in rays])
            curve.setData(ray_z, ray_y)
        self.restyle_ray_curves()

        # Move the draggable image line without re-triggering its own change signal.
        self.image_line.blockSignals(True)
        self.image_line.setValue(self.model.state.image_z)
        self.image_line.blockSignals(False)
        self.place_image_label()
        self.update_image_line_movable()
        self.update_source_line()

        self.refresh_lens_colours()
        self.update_gap_dimensions()

    def restyle_ray_curves(self) -> None:
        """Draw each field's rays in its colour, thicker (and on top) if it is highlighted."""
        for index, (beam, curve) in enumerate(zip(self.model.state.fields, self.ray_curves)):
            highlighted = index == self.highlighted_field_index
            width = HIGHLIGHTED_RAY_WIDTH_PX if highlighted else RAY_WIDTH_PX
            curve.setPen(pg.mkPen(beam.display_colour(), width=width))
            curve.setZValue(1 if highlighted else 0)

    def set_highlight(self, lens_index: int | None, field_index: int | None) -> None:
        """Highlight a lens (drawn as if hovered) and a field (thicker rays); None for neither."""
        self.highlighted_lens_index = lens_index
        self.highlighted_field_index = field_index
        self.refresh_lens_colours()
        self.restyle_ray_curves()

    def _match_ray_curve_count(self, field_count: int) -> None:
        """Add or remove ray curves so there is one per field."""
        side = self.getPlotItem()
        while len(self.ray_curves) < field_count:
            curve = pg.PlotCurveItem(connect="finite")
            side.addItem(curve)
            self.ray_curves.append(curve)
        while len(self.ray_curves) > field_count:
            side.removeItem(self.ray_curves.pop())

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

        z_min, z_max = self.z_range
        y_min, y_max = self.y_range
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

    # ------------------------------------------------------------------ gaps

    def update_gap_dimensions(self) -> None:
        """Redraw the dimension of every air gap, and of the gap from the source plane.

        The source gap is left out while it is locked at 0 mm (the fields are then simply
        defined at the first lens).
        """
        shown_gaps = self.model.air_gaps()
        source_gap = self.model.source_gap()
        if source_gap is not None and not (source_gap.locked and source_gap.width_mm == 0.0):
            shown_gaps = [source_gap, *shown_gaps]
        self.gap_dimensions.update(shown_gaps, self.gap_label_style)

    def gap_label_style(self, gap: AirGap) -> str:
        """How a gap's label is drawn (see ``GapDimensions.update``)."""
        if gap.locked:
            return LOCKED
        return EDITABLE if self.why_gap_is_fixed(gap) is None else FIXED

    def on_view_resized(self, *_args) -> None:
        """The dimensions are placed in screen pixels, so a new size moves them."""
        self.close_gap_editor()
        self.update_gap_dimensions()

    def why_gap_is_fixed(self, gap: AirGap) -> str | None:
        """Why this gap cannot be typed, in words for the user; None if it can."""
        if gap.in_assembly:
            return "The gaps inside an assembly are fixed: split it to change them"
        if gap.from_source and gap.locked:
            caption = source_caption(self.model.state)
            return f'The gap from the source plane is locked: right-click "{caption}" to unlock it'
        if gap.locked:
            return "This gap is locked: right-click it to unlock it"
        if gap.to_image and self.image_line_locked:
            return '"Lock focus" sets the distance to the image plane: turn it off to type it'
        return None

    def open_gap_editor(self, gap: AirGap) -> None:
        """Show the box for typing this gap, or say why it cannot be changed."""
        self.close_gap_editor()  # which may apply a value, so read the gap afresh
        if gap.from_source:
            source_gap = self.model.source_gap()
            if source_gap is None:
                return
            gap = source_gap
        else:
            gap = self.model.air_gaps()[gap.lens_index]
        reason = self.why_gap_is_fixed(gap)
        if reason is not None:
            self.message_requested.emit(reason)
            return
        scene_centre = self.gap_dimensions.label_scene_centre(gap.lens_index)
        centre = QPointF(self.mapFromScene(scene_centre))
        self.gap_editor = GapEditor(gap, centre, parent=self.viewport())
        self.gap_editor.accepted.connect(self.set_gap)
        self.gap_editor.closed.connect(self.on_gap_editor_closed)

    def close_gap_editor(self) -> None:
        """Remove the gap box, if open, without applying it."""
        if self.gap_editor is not None:
            self.gap_editor.close_editor()

    def on_gap_editor_closed(self) -> None:
        self.gap_editor = None

    def set_gap(self, lens_index: int, gap_mm: float) -> None:
        """Apply a typed gap to the model, or say why it is not allowed."""
        try:
            if lens_index == SOURCE_GAP_INDEX:
                self.model.set_source_gap(gap_mm)
            else:
                self.model.set_gap(lens_index, gap_mm)
        except ValueError as error:
            self.message_requested.emit(str(error))
            return
        self.changed.emit()

    def showEvent(self, event) -> None:
        """Once shown, shrink the axis numbers and size the timing text to match.

        The default axis font is only known once the style sheet has been applied, i.e. when
        the view is first shown. The numbers are scaled once; the timing text is 2/3 of the
        default size.
        """
        super().showEvent(event)
        side = self.getPlotItem()
        axis_font = QFont(side.getAxis("bottom").style["tickFont"] or self.viewport().font())
        if not self.tick_fonts_scaled:
            for axis_name in ("bottom", "left"):
                side.getAxis(axis_name).setStyle(tickFont=scaled_font(axis_font, TICK_FONT_SCALE))
            self.tick_fonts_scaled = True
            self.timing_label.setFont(scaled_font(axis_font, TIMING_FONT_SCALE))
        self.axis_glyph.place()

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
        """Blue (purple in an assembly); darker when it, or a lens joined to it by locked
        gaps, is under the pointer, selected or highlighted in the bench viewer."""
        in_assembly = self.model.state.lenses[index].assembly_id is not None
        # Lenses joined by locked gaps move as one, so they are highlighted as one.
        darkened = []
        for marked_index in (self.hovered_index, self.selected_index, self.highlighted_lens_index):
            if marked_index is not None and marked_index < len(self.model.state.lenses):
                darkened += self.model.block_indices(marked_index)
        if index in darkened:
            return palette.ASSEMBLY_HOVER if in_assembly else palette.LENS_HOVER
        return palette.ASSEMBLY if in_assembly else palette.LENS

    def refresh_lens_colours(self) -> None:
        """Redraw the lenses in their current colours (no re-trace needed), and the name tag."""
        for index, (lens, graphics) in enumerate(zip(self.model.state.lenses, self.lens_graphics)):
            graphics.update(lens, self.lens_colour(index))
        self.place_name_tag()

    def set_selected_lens(self, lens_index: int | None) -> None:
        """Select a lens (None: no lens), redraw the colours and emit ``lens_selected``."""
        if lens_index == self.selected_index:
            return
        self.selected_index = lens_index
        if len(self.lens_graphics) == len(self.model.state.lenses):
            self.refresh_lens_colours()
        self.lens_selected.emit(lens_index)

    def set_hovered_lens(self, lens_index: int | None) -> None:
        """Remember which lens the pointer is over and recolour the lenses if that changed."""
        if lens_index != self.hovered_index:
            self.hovered_index = lens_index
            self.refresh_lens_colours()

    def place_name_tag(self) -> None:
        """Show the hovered lens's part number centred above it, a few pixels clear of its top.

        A lens in an assembly is labelled with the assembly's name too. The tag's box is a
        shade darker than the lens outline. It is hidden when no lens is hovered.
        """
        # While lenses are being placed, the tag names the ghost instead.
        placement = self.placement
        if placement is not None:
            if placement.is_visible():
                self._show_name_tag(
                    placement.name,
                    placement.lenses[0].z,
                    placement.lenses[-1].back_z,
                    max(lens.semi_aperture for lens in placement.lenses),
                    ghost_colour(placement),
                )
            else:
                self.name_tag.hide()
            return

        # The tag is not drawn while a lens is being dragged: it would flicker as the hover
        # state changes under the moving lens. It reappears when the drag ends.
        if self.hovered_index is None or self.lens_drag_active:
            self.name_tag.hide()
            return

        lens = self.model.state.lenses[self.hovered_index]
        assembly_name = self.model.assembly_name(self.hovered_index)
        text = (
            lens.part_number if assembly_name is None else f"{lens.part_number} · {assembly_name}"
        )
        self._show_name_tag(
            text, lens.z, lens.back_z, lens.semi_aperture, self.lens_colour(self.hovered_index)
        )

    def _show_name_tag(
        self, text: str, front_z: float, back_z: float, top_y: float, lens_colour: str
    ) -> None:
        """Put the name tag above what runs from ``front_z`` to ``back_z`` and up to ``top_y``.

        Its box is a shade darker than ``lens_colour``.
        """
        plot_item = self.getPlotItem()
        lens_top_centre = plot_item.mapFromScene(
            plot_item.vb.mapViewToScene(QPointF(0.5 * (front_z + back_z), top_y))
        )

        # The drawn box is LENS_NAME_TRIM_PX inside the text item at top and bottom, so allow
        # for that when leaving the gap above the lens; never let it poke out of the plot top.
        tag_height = self.name_tag.boundingRect().height()
        tag_bottom = max(
            lens_top_centre.y() - LENS_NAME_GAP_PX + LENS_NAME_TRIM_PX,
            tag_height - LENS_NAME_TRIM_PX + 2.0,
        )

        self.name_tag.setText(text)
        self.name_tag.box_colour = QColor(lens_colour).darker(125)
        self.name_tag.setPos(lens_top_centre.x(), tag_bottom)
        self.name_tag.show()

    def is_over_image_line(self, scene_position: QPointF) -> bool:
        """True if the pointer is within a few pixels of the image line and it can be dragged."""
        if self.image_line_fixed:
            return False
        view_box = self.getPlotItem().vb
        line_scene_x = view_box.mapViewToScene(QPointF(self.image_line.value(), 0.0)).x()
        return abs(scene_position.x() - line_scene_x) <= IMAGE_HOVER_TOLERANCE_PX

    def is_over_source_line(self, scene_position: QPointF) -> bool:
        """True if the pointer is within a few pixels of the source line and it can be dragged."""
        if not self.source_line.movable:
            return False
        view_box = self.getPlotItem().vb
        line_scene_x = view_box.mapViewToScene(QPointF(self.source_line.value(), 0.0)).x()
        return abs(scene_position.x() - line_scene_x) <= IMAGE_HOVER_TOLERANCE_PX

    def update_source_line(self) -> None:
        """Put the source line and caption at the source plane; draggable unless its gap is
        locked. Hidden with no lenses, when there is no gap to set."""
        state = self.model.state
        has_lenses = bool(state.lenses)
        self.source_line.setVisible(has_lenses)
        self.source_label.setVisible(has_lenses)
        caption = source_caption(state)
        if self.source_label.toPlainText() != caption:
            self.source_label.setText(caption)
        self.source_line.blockSignals(True)
        self.source_line.setValue(state.source_z)
        self.source_line.blockSignals(False)
        movable = has_lenses and not state.source_gap_locked
        if movable != self.source_line.movable:
            self.source_line.setMovable(movable)
        y_span = self.y_range[1] - self.y_range[0]
        self.source_label.setPos(state.source_z, self.y_range[1] - 0.01 * y_span)

    def on_source_dragged(self) -> None:
        """The source line was dragged: keep it between the axis start and the first lens."""
        self.close_gap_editor()
        state = self.model.state
        if not state.lenses or state.source_gap_locked:
            return
        wanted_z = float(np.clip(self.source_line.value(), AXIS_START_Z, state.lenses[0].z))
        self.model.set_source_z(wanted_z)
        self.source_label.setPos(wanted_z, self.source_label.pos().y())
        self.changed.emit()

    def place_image_label(self, *_args) -> None:
        """Put the "image" caption just inside the top of the plot, centred on the image line."""
        y_span = self.y_range[1] - self.y_range[0]
        self.image_label.setPos(self.image_line.value(), self.y_range[1] - 0.01 * y_span)

    def place_timing_label(self) -> None:
        """Put the trace and draw timings just inside the top-right corner of the plot."""
        z_span = self.z_range[1] - self.z_range[0]
        y_span = self.y_range[1] - self.y_range[0]
        self.timing_label.setPos(self.z_range[1] - 0.01 * z_span, self.y_range[1] - 0.04 * y_span)

    def on_hover(self, scene_position: QPointF) -> None:
        """Track the lens under the pointer; show a left/right arrow over anything draggable.

        While a lens is being placed, move its ghost to the pointer instead.
        """
        if self.placement is not None:
            self.move_ghost(self.placement, scene_position)
            return

        view_point = self.getPlotItem().vb.mapSceneToView(scene_position)
        gap = self.gap_dimensions.gap_at(scene_position)
        hovered_index = None if gap is not None else self.lens_index_at(view_point)
        self.set_hovered_lens(hovered_index)

        if gap is not None and self.why_gap_is_fixed(gap) is None:
            self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
        elif (
            hovered_index is not None
            or self.is_over_image_line(scene_position)
            or self.is_over_source_line(scene_position)
        ):
            self.viewport().setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.viewport().unsetCursor()
        self.place_name_tag()

    def eventFilter(self, watched, event) -> bool:
        """Clear the hover when the pointer leaves the view (no mouse-move event is sent then)."""
        if watched is self.viewport() and event.type() == QEvent.Type.Leave:
            self.set_hovered_lens(None)
            if self.placement is not None:
                self.placement.set_visible(False)
                self.place_name_tag()
            else:
                self.viewport().unsetCursor()
        return super().eventFilter(watched, event)

    # ------------------------------------------------------------ placing a new lens

    def start_placement(
        self, new_lenses: list[PlacedLens], assembly_name: str | None = None
    ) -> None:
        """Let the user put new lenses on the axis: their ghost follows the pointer.

        ``new_lenses`` is one lens, or the lenses of a user part (``assembly_name`` is then
        the part's name), which are placed together as one assembly. They are not in the
        model until the user clicks. Lenses on the bench cannot be dragged meanwhile, so a
        click near one places the new lenses rather than moving it.
        """
        self.cancel_placement()  # in case a previous placement is still going on
        self.close_gap_editor()
        placement = Placement(lenses=new_lenses, assembly_name=assembly_name)
        for lens in new_lenses:
            ghost = LensShape(self.getPlotItem(), paraxial=lens.is_paraxial)
            ghost.set_z_value(GHOST_Z_VALUE)
            placement.ghosts.append(ghost)
        placement.set_visible(False)  # shown once the pointer is over the plot
        self.placement = placement

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
        axis). Where the lenses fit nowhere, the ghost is drawn red under the pointer.
        """
        view_box = self.getPlotItem().vb
        if not view_box.sceneBoundingRect().contains(scene_position):
            placement.set_visible(False)
            self.place_name_tag()
            return

        pointer_z = view_box.mapSceneToView(scene_position).x()
        requested_front_z = pointer_z - placement.length_mm / 2.0
        free_front_z = self.model.free_front_z(
            placement.length_mm,
            requested_front_z,
            max_back_z=self.image_max_z() - MIN_IMAGE_GAP_MM,
        )
        placement.fits = free_front_z is not None
        placement.move_to(requested_front_z if free_front_z is None else free_front_z)

        placement.draw()
        placement.set_visible(True)
        self.place_name_tag()

    def on_scene_clicked(self, event) -> None:
        """A left click types a gap or selects a lens; while placing, it puts the new lenses
        where their ghost is."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        placement = self.placement
        if placement is None:
            gap = self.gap_dimensions.gap_at(event.scenePos())
            if gap is not None:
                # Open the box once the mouse release has been handled: on Linux, focus
                # moving at the end of the release otherwise reached the new box as a
                # focus-out, which accepts and closes it at once.
                QTimer.singleShot(0, lambda: self.open_gap_editor(gap))
                return
            view_point = self.getPlotItem().vb.mapSceneToView(event.scenePos())
            self.set_selected_lens(self.lens_index_at(view_point))
            return
        self.move_ghost(placement, event.scenePos())  # the ghost may lag a fast click
        if not (placement.fits and placement.is_visible()):
            return
        self._end_placement(placement)
        self.last_placed_lenses = placement.lenses
        self.model.insert_lenses(
            placement.lenses, max_image_z=self.image_max_z(), assembly_name=placement.assembly_name
        )
        self.rebuild_lens_graphics()
        self.changed.emit()
        self.placement_finished.emit(True)

    def update_ghost_lenses(self, new_lenses: list[PlacedLens]) -> None:
        """Swap the lenses being placed for ones of the same kind (e.g. a new focal length)."""
        if self.placement is None:
            return
        self.placement.lenses = new_lenses
        self.placement.draw()
        self.place_name_tag()

    def cancel_placement(self) -> None:
        """Stop placing a lens without adding it (Escape or right click)."""
        if self.placement is not None:
            self._end_placement(self.placement)
            self.placement_finished.emit(False)

    def _end_placement(self, placement: Placement) -> None:
        """Remove the ghost and make the lenses on the bench draggable again."""
        placement.remove()
        self.placement = None
        for graphics in self.lens_graphics:
            graphics.region.setMovable(True)
        self.viewport().unsetCursor()
        self.name_tag.hide()

    # ------------------------------------------------------------ user interaction

    def set_image_line_locked(self, locked: bool) -> None:
        """Lock focus on: the image plane is set by the software, so the line cannot be dragged."""
        self.image_line_locked = locked
        self.update_image_line_movable()

    def update_image_line_movable(self) -> None:
        """Let the image line be dragged unless "Lock focus" is on or its gap is locked.

        A line that is not movable gets no hover highlight, so it stays dashed; the arrow
        cursor is suppressed in ``is_over_image_line``. Runs on every redraw, so it changes
        the line only when the answer changes.
        """
        state = self.model.state
        fixed = self.image_line_locked or (bool(state.lenses) and state.image_gap_locked)
        if fixed == self.image_line_fixed:
            return
        self.image_line_fixed = fixed
        self.image_line.setMovable(not fixed)
        self.image_line.setPen(self._image_line_pen())  # clears any hover look
        if fixed:
            self.viewport().unsetCursor()

    def on_right_click(self, _widget_position) -> None:
        """Ask for the gap or lens menu if the pointer is over one; cancel placing a lens."""
        if self.placement is not None:
            self.cancel_placement()
            return
        global_position = QCursor.pos()
        scene_position = self.mapToScene(self.viewport().mapFromGlobal(global_position))
        gap = self.gap_dimensions.gap_at(scene_position)
        if gap is not None:
            self.close_gap_editor()
            self.gap_menu_requested.emit(gap.lens_index, global_position)
            return
        if self.source_label.isVisible() and self.source_label.sceneBoundingRect().contains(
            scene_position
        ):
            self.gap_menu_requested.emit(SOURCE_GAP_INDEX, global_position)
            return
        lens_index = self.lens_index_at(self.getPlotItem().vb.mapSceneToView(scene_position))
        if lens_index is not None:
            self.lens_menu_requested.emit(lens_index, global_position)

    def on_lens_dragged(self, graphics: LensGraphics) -> None:
        """A lens was dragged: move it and its block, or swap it with a neighbouring block."""
        # pyqtgraph keeps sending drag events to the item under the mouse even after the
        # graphics were rebuilt (e.g. Del pressed mid-drag); that lens is no longer shown.
        if graphics not in self.lens_graphics:
            return
        self.close_gap_editor()
        self.lens_drag_active = True
        index = self.lens_graphics.index(graphics)
        new_order = self.model.move_lens(index, graphics.requested_front_z())
        if new_order is not None:
            # The lens list was reordered; reorder the graphics (and the hover marker) to match,
            # so the item being dragged keeps showing the lens it belongs to.
            self.lens_graphics = [self.lens_graphics[old_index] for old_index in new_order]
            self.hovered_index = new_order.index(index)
            if self.selected_index is not None:
                self.selected_index = new_order.index(self.selected_index)
            if self.highlighted_lens_index is not None:
                self.highlighted_lens_index = new_order.index(self.highlighted_lens_index)
        self.changed.emit()

    def on_lens_drag_finished(self) -> None:
        """The mouse was released after dragging a lens: bring the name tag back."""
        self.lens_drag_active = False
        self.place_name_tag()
        self.drag_finished.emit()

    def is_dragging(self) -> bool:
        """True while the mouse is holding a lens or the image plane."""
        return self.lens_drag_active or self.image_line.moving or self.source_line.moving

    def on_image_dragged(self) -> None:
        """The image line was dragged: keep it behind the last lens and inside the view."""
        self.close_gap_editor()
        state = self.model.state
        minimum_z = state.last_back_z() + MIN_IMAGE_GAP_MM if state.lenses else self.z_range[0]
        state.image_z = float(np.clip(self.image_line.value(), minimum_z, self.image_max_z()))
        self.changed.emit()
