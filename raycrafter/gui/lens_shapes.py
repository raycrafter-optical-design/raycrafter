"""How a lens is drawn on the side view: its shape, its drag handle and its name tag.

``LensShape`` draws a lens (a glass body, or for a paraxial lens a line with arrowheads);
``LensGraphics`` adds the invisible region by which a lens on the bench is dragged.
``Placement`` holds the dashed "ghost" of lenses being placed, before they are on the bench.
They are kept apart from ``raycrafter.gui.side_view``, which arranges them, to keep that file
a manageable size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QPainter

from raycrafter import palette
from raycrafter.core.plot_geometry import lens_surface_arcs
from raycrafter.core.state import PlacedLens
from raycrafter.gui.style import colour_with_alpha, join_with_gaps

LENS_NAME_TRIM_PX = 2  # the name tag's box is this much tighter than its text item, top and bottom
LENS_NAME_CORNER_PX = 4  # corner radius of the name tag
LENS_FILL_ALPHA = 70  # fill opacity (0-255) of a lens on the bench
GHOST_FILL_ALPHA = 35  # fill opacity of a lens being placed
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
    """New lenses following the pointer, waiting for a click to be put on the bench.

    The lenses are one catalogue or paraxial lens, or the lenses of a user part, which are
    placed together as one assembly.

    Attributes
    ----------
    lenses : list of PlacedLens
        The new lenses (not yet in the model); their z is kept at the ghost's position.
    ghosts : list of LensShape
        Their dashed outlines on the plot, one per lens.
    assembly_name : str or None
        For a user part, its name, which the new assembly is given.
    fits : bool
        True if the lenses fit at the ghost's current position.
    """

    lenses: list[PlacedLens]
    ghosts: list[LensShape] = field(default_factory=list)
    assembly_name: str | None = None
    fits: bool = False

    @property
    def name(self) -> str:
        """What the name tag shows: the part's name, or the lens's part number."""
        return self.assembly_name or self.lenses[0].part_number

    @property
    def length_mm(self) -> float:
        """From the first front vertex to the last back vertex (mm)."""
        return self.lenses[-1].back_z - self.lenses[0].z

    def move_to(self, front_z: float) -> None:
        """Move the lenses together so that the first front vertex is at ``front_z`` (mm)."""
        shift = front_z - self.lenses[0].z
        for lens in self.lenses:
            lens.z += shift

    def draw(self) -> None:
        """Draw every ghost dashed, in blue where the lenses fit and red where they do not."""
        colour = ghost_colour(self)
        for lens, ghost in zip(self.lenses, self.ghosts):
            ghost.draw(lens, colour, dashed=True)

    def set_visible(self, visible: bool) -> None:
        """Show or hide the ghosts."""
        for ghost in self.ghosts:
            ghost.set_visible(visible)

    def is_visible(self) -> bool:
        """True if the ghosts are drawn."""
        return self.ghosts[0].is_visible()

    def remove(self) -> None:
        """Take the ghosts off the plot."""
        for ghost in self.ghosts:
            ghost.remove()


def ghost_colour(placement: Placement) -> str:
    """Lens blue where the new lenses fit; red where they do not."""
    return palette.LENS if placement.fits else palette.GHOST_BLOCKED
