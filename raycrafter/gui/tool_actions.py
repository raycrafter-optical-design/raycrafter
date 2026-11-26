"""What the tools of the palette do: the link between the palette, the context bar and the bench.

Choosing a tool shows its settings in the context bar and starts it:

- Move: lenses can be dragged, hovered, right-clicked and selected (the default). While a
  paraxial lens is selected, the context bar shows its focal length and diameter, and
  changing them changes the lens at once.
- Add paraxial lens: a ghost of a paraxial lens of the focal length and diameter in the
  context bar follows the pointer; each click places one. Changing them changes the ghost.
  These values for the next lens are kept apart from those of a selected lens.
- Add user part: the same, for the part chosen in the context bar's drop-down.
- Add catalogue lens: for now, opens the lens picker; the chosen lens is placed once, then
  the palette goes back to Move. (The picker moves into the main window later, #70.)
- Add field: for now, opens the field window and goes back to Move (#70).

The add tools that place things stay active: after each placement a new ghost follows the
pointer. Escape, a right click, V or choosing another tool ends them. Going back to Move
selects the lens just placed, so its settings can be changed at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raycrafter.core.placement import PARAXIAL_SEMI_APERTURE_MM
from raycrafter.gui.lens_picker import LensPickerDialog
from raycrafter.gui.tool_palette import Tool

if TYPE_CHECKING:
    from raycrafter.core.state import PlacedLens
    from raycrafter.gui.main_window import MainWindow

DEFAULT_PARAXIAL_FOCAL_LENGTH_MM = 100.0
DEFAULT_PARAXIAL_DIAMETER_MM = 2.0 * PARAXIAL_SEMI_APERTURE_MM


class ToolActions:
    """Carries out the palette's tools for the main window.

    Parameters
    ----------
    window : MainWindow
        Its ``tool_palette``, ``context_bar`` and ``side_view`` are used.
    """

    def __init__(self, window: MainWindow) -> None:
        self.window = window
        self.palette = window.tool_palette
        self.bar = window.context_bar
        # True while a placement is replaced by a new one: the side view then reports the
        # old one as cancelled, which must not end the tool.
        self.restarting_placement = False
        # The focal length and diameter (mm) of the next paraxial lens to be placed.
        self.next_paraxial_mm = (DEFAULT_PARAXIAL_FOCAL_LENGTH_MM, DEFAULT_PARAXIAL_DIAMETER_MM)

        self.bar.set_paraxial_values_mm(*self.next_paraxial_mm)
        self.palette.tool_chosen.connect(self.choose_tool)
        self.palette.autofocus_requested.connect(window.run_autofocus)
        self.bar.paraxial_changed.connect(self.on_paraxial_changed)
        self.bar.user_part_activated.connect(lambda _name: self.start_user_part_placement())
        window.side_view.placement_finished.connect(self.on_placement_finished)
        window.side_view.lens_selected.connect(self.on_lens_selected)

    @property
    def tool(self) -> Tool:
        """The active tool."""
        return self.palette.current_tool()

    def choose_tool(self, tool: Tool) -> None:
        """Make ``tool`` the active tool: show its settings and start it."""
        side_view = self.window.side_view
        just_placed = side_view.last_placed_lenses
        side_view.last_placed_lenses = []
        self.palette.set_tool(tool)
        side_view.set_selected_lens(None)  # selecting is for the Move tool
        self.bar.show_page(tool)
        self._replace_placement(None)  # whatever was being placed belongs to the old tool
        if tool == Tool.MOVE and just_placed:
            self.select_lens(just_placed[0])
        if tool == Tool.ADD_PARAXIAL_LENS:
            self.bar.set_paraxial_values_mm(*self.next_paraxial_mm)
            self.start_paraxial_lens_placement()
        elif tool == Tool.ADD_USER_PART:
            self.bar.set_user_parts(self.user_part_labels())
            self.start_user_part_placement()
        elif tool == Tool.ADD_CATALOGUE_LENS:
            self.open_lens_picker()
        elif tool == Tool.ADD_FIELD:
            self.window.open_field_editor()
            self.choose_tool(Tool.MOVE)

    def select_lens(self, lens: PlacedLens) -> None:
        """Select a lens on the bench (if it is still there), as a click on it would."""
        lenses = self.window.model.state.lenses
        index = next((index for index, other in enumerate(lenses) if other is lens), None)
        self.window.side_view.set_selected_lens(index)

    def escape(self) -> None:
        """Escape: end the active tool and go back to Move."""
        self.choose_tool(Tool.MOVE)

    # ------------------------------------------------------------------ placing

    def start_paraxial_lens_placement(self) -> None:
        """A ghost of a paraxial lens with the context bar's values follows the pointer."""
        new_lens = self.paraxial_lens_from_bar()
        if new_lens is None:
            self._replace_placement(None)
            self.window.statusBar().showMessage("Set a focal length other than 0 mm")
            return
        self._replace_placement([new_lens])

    def paraxial_lens_from_bar(self, z: float = 0.0) -> PlacedLens | None:
        """A paraxial lens with the context bar's focal length and diameter; None if f is 0."""
        focal_length_mm, diameter_mm = self.bar.paraxial_values_mm()
        if focal_length_mm == 0.0:
            return None
        return self.window.model.make_paraxial_lens(focal_length_mm, z, diameter_mm)

    def on_paraxial_changed(self, focal_length_mm: float, diameter_mm: float) -> None:
        """New paraxial values: for the ghost being placed, or for the selected lens."""
        side_view = self.window.side_view
        if self.tool == Tool.ADD_PARAXIAL_LENS:
            self.next_paraxial_mm = (focal_length_mm, diameter_mm)
            ghost = side_view.placement
            new_lens = self.paraxial_lens_from_bar(ghost.lenses[0].z if ghost else 0.0)
            if ghost is not None and new_lens is not None:
                side_view.update_ghost_lenses([new_lens])  # cheap: no new placement
            else:
                self.start_paraxial_lens_placement()  # starts or ends the ghost
            return

        lens_index = side_view.selected_index
        if lens_index is None or focal_length_mm == 0.0:
            return  # a focal length of 0 is not a lens: the lens keeps its last one
        self.window.model.set_paraxial_lens(lens_index, focal_length_mm, diameter_mm)
        # A run of small steps (a slider drag) is one undo step.
        self.window.request_update(undo_merge_key=("paraxial lens", lens_index))

    def on_lens_selected(self, lens_index: int | None) -> None:
        """With Move active, a selected paraxial lens shows its values in the context bar."""
        if self.tool != Tool.MOVE:
            return
        lens = None if lens_index is None else self.window.model.state.lenses[lens_index]
        if lens is None or lens.paraxial_focal_length_mm is None:
            self.bar.show_page(Tool.MOVE)
            return
        self.bar.set_paraxial_values_mm(lens.paraxial_focal_length_mm, 2.0 * lens.semi_aperture)
        self.bar.show_page(Tool.ADD_PARAXIAL_LENS)  # the same controls edit the selected lens

    def user_part_labels(self) -> list[tuple[str, str]]:
        """(label, name) of each saved user part, read afresh from disk for the drop-down."""
        return [
            (f"{part.name}  ({part.length_mm:.1f} mm)", part.name)
            for part in self.window.model.user_parts()
        ]

    def start_user_part_placement(self) -> None:
        """The part chosen in the context bar follows the pointer, as one assembly."""
        name = self.bar.chosen_user_part()
        parts = {part.name: part for part in self.window.model.user_parts()}
        if name not in parts:
            self._replace_placement(None)
            return
        self._replace_placement(parts[name].lenses_at(0.0), name)

    def open_lens_picker(self) -> None:
        """Show the lens picker; after Insert, let the user place the chosen lens once."""
        window = self.window
        if window.lens_picker is None:
            window.lens_picker = LensPickerDialog(window.model.catalog, parent=window)
        window.lens_picker.set_design_wavelengths(
            [beam.wavelength_nm for beam in window.model.state.fields]
        )
        accepted = window.lens_picker.exec() == LensPickerDialog.DialogCode.Accepted
        part_number = window.lens_picker.selected_part_number() if accepted else None
        if part_number is None:
            self.choose_tool(Tool.MOVE)
            return
        self._replace_placement([window.model.make_lens(part_number)])

    def _replace_placement(self, new_lenses, assembly_name: str | None = None) -> None:
        """End any placement without ending the tool, then start placing ``new_lenses``."""
        self.restarting_placement = True
        try:
            self.window.side_view.cancel_placement()
            if new_lenses:
                self.window.start_lens_placement(new_lenses, assembly_name)
            else:
                self.window.statusBar().clearMessage()
        finally:
            self.restarting_placement = False

    def on_placement_finished(self, placed: bool) -> None:
        """After a click, a sticky tool places again; a cancel goes back to Move."""
        if self.restarting_placement:
            return
        self.window.statusBar().clearMessage()
        if not placed:
            self.choose_tool(Tool.MOVE)
        elif self.tool == Tool.ADD_PARAXIAL_LENS:
            self.start_paraxial_lens_placement()
        elif self.tool == Tool.ADD_USER_PART:
            self.start_user_part_placement()
        else:
            self.choose_tool(Tool.MOVE)  # a catalogue lens, placed once for now
