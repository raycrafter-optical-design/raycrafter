"""The lens bench model: the one object that front ends (GUI, report, scripts) talk to.

``OpticalModel`` owns the bench state (``raycrafter.core.state``), the lenses that can be
placed (the catalogue), the user parts folder, the undo history and the project file. Its
methods are the editing operations and the analyses, applied to its current state. It has
no Qt code.

Most methods are one-line wrappers: the work, and the rules it follows, are in the module
named in each docstring, as functions of an ``OpticalState``:

- ``placement``: making, adding, inserting and deleting lenses; the source plane.
- ``moving``: moving lenses along the axis and swapping neighbouring blocks.
- ``gaps``: setting and locking the air gaps.
- ``assemblies``: assemblies, Plossl pairs and flipping.
- ``replacement``: replacing a lens in place.
- ``gap_optimisation``: setting a gap to collimate the output or focus.
- ``analysis``: spots, MTF, ray fans and the other analyses.

What stays here is what needs more than the state: the catalogue, the user parts folder,
the undo history, and saving and opening files.

Example
-------
>>> from raycrafter.core.model import OpticalModel
>>> model = OpticalModel()  # starts with one ACT508-200-A on the axis
>>> model.autofocus()  # doctest: +SKIP
>>> model.trace_fast().spot.rms_radius  # doctest: +SKIP
"""

from __future__ import annotations

import copy
from collections.abc import Hashable
from dataclasses import replace
from pathlib import Path

import numpy as np

from raycrafter.core import (
    analysis,
    assemblies,
    gap_optimisation,
    gaps,
    moving,
    optic_builder,
    placement,
    project_file,
    replacement,
    user_parts,
    zmx_import,
)
from raycrafter.core.assemblies import PLOSSL_GAP_MM  # noqa: F401  (imported from here too)
from raycrafter.core.history import UndoHistory, same_design
from raycrafter.core.placement import (  # noqa: F401  (imported from here too)
    FIRST_LENS_MIN_Z,
    INITIAL_LENS_Z,
    LENS_GAP_MM,
)
from raycrafter.core.state import BeamField, OpticalState, PlacedLens
from raycrafter.core.user_parts import UserPart
from raycrafter.lenscatalog import thorlabs
from raycrafter.lenscatalog.catalog import CatalogEntry

DEFAULT_LENS_PART = "ACT508-200-A"  # the lens placed when the model is created


class OpticalModel:
    """A row of lenses on an optical axis, a beam and an image plane.

    Parameters
    ----------
    catalog : dict, optional
        Part number -> CatalogEntry for the lenses that can be placed. Defaults to the
        packaged Thorlabs catalogue.
    add_default_lens : bool
        If True (the default), start with one DEFAULT_LENS_PART on the axis.

    Attributes
    ----------
    state : OpticalState
        The lenses, fields (beams) and image plane. Front ends may set a field's properties
        and ``image_z`` directly; lenses, and which fields exist, should be changed through
        the methods here. ``undo``, ``redo``, ``new_design``, ``load_project`` and
        ``load_zmx`` replace this object, so do not keep a reference to it.
    catalog : dict
        Part number -> CatalogEntry.
    project_path : Path or None
        The project file last saved or opened, where ``save_project`` saves by default;
        None for a new or imported design.
    parts_folder : Path
        Where user parts are saved and read (``user_parts.default_parts_folder()`` unless
        changed).

    Notes
    -----
    Undo steps are not recorded automatically: a front end calls ``record_undo_step`` when a
    user action is complete.
    """

    def __init__(
        self, catalog: dict[str, CatalogEntry] | None = None, add_default_lens: bool = True
    ) -> None:
        self.state = OpticalState()
        self.catalog = catalog if catalog is not None else thorlabs.load_lenses()
        if add_default_lens and DEFAULT_LENS_PART in self.catalog:
            self.add_lens(DEFAULT_LENS_PART)
        self.saved_state: OpticalState = copy.deepcopy(
            self.state
        )  # the start-up bench counts as saved
        self.project_path: Path | None = None  # the project file last saved or opened
        self.parts_folder: Path = user_parts.default_parts_folder()
        self.history = UndoHistory(self.state)

    # --------------------------------------------------------------- adding and removing

    def make_lens(self, part_number: str, front_z: float = INITIAL_LENS_Z) -> PlacedLens:
        """A catalogue lens at the given front-vertex z, not yet on the bench.

        Front ends use this to preview a lens (e.g. the ghost that follows the pointer)
        before ``insert_lens`` puts it on the bench.
        """
        return placement.make_lens(self.catalog[part_number], front_z)

    @staticmethod
    def make_paraxial_lens(focal_length_mm: float, z: float = INITIAL_LENS_Z) -> PlacedLens:
        """An ideal thin lens at ``z`` (see ``placement.make_paraxial_lens``)."""
        return placement.make_paraxial_lens(focal_length_mm, z)

    def add_lens(self, part_number: str) -> PlacedLens:
        """Place a catalogue lens just after the last one and return it (``placement.append_lens``)."""
        new_lens = self.make_lens(part_number)
        placement.append_lens(self.state, new_lens)
        return new_lens

    def free_front_z(
        self, thickness_mm: float, requested_front_z: float, max_back_z: float = np.inf
    ) -> float | None:
        """Nearest front z where a new lens fits (see ``placement.free_front_z``)."""
        return placement.free_front_z(self.state, thickness_mm, requested_front_z, max_back_z)

    def insert_lens(self, new_lens: PlacedLens, max_image_z: float = np.inf) -> int:
        """Put one lens on the bench at its own z; return its index (``placement.insert_lenses``)."""
        return placement.insert_lenses(self.state, [new_lens], max_image_z)[0]

    def insert_lenses(
        self,
        new_lenses: list[PlacedLens],
        max_image_z: float = np.inf,
        assembly_name: str | None = None,
    ) -> list[int]:
        """Put a run of lenses on the bench as one block (see ``placement.insert_lenses``)."""
        return placement.insert_lenses(self.state, new_lenses, max_image_z, assembly_name)

    def delete_lens(self, lens_index: int) -> None:
        """Remove one lens and the locks either side of it (see ``placement.delete_lens``)."""
        placement.delete_lens(self.state, lens_index)

    def flip_lens(self, lens_index: int) -> None:
        """Turn one lens around, unless paraxial or in an assembly (``assemblies.flip_lens``)."""
        assemblies.flip_lens(self.state, lens_index)

    # ---------------------------------------------------------------- replacing lenses

    def lens_focal_length_mm(self, lens_index: int) -> float:
        """The focal length (mm) of one lens on its own, at the selected field's wavelength."""
        wavelength_nm = self.state.selected_field.wavelength_nm
        return replacement.focal_length_mm(self.state.lenses[lens_index], wavelength_nm)

    def in_plossl_pair(self, lens_index: int) -> bool:
        """True if the lens is one of a Plossl pair."""
        return assemblies.in_plossl_pair(self.state, lens_index)

    def replace_lens(
        self, lens_index: int, new_lens: PlacedLens, split_plossl_pair: bool = False
    ) -> None:
        """Put ``new_lens`` in place of a lens (see ``replacement.replace_lens``)."""
        replacement.replace_lens(self.state, lens_index, new_lens, split_plossl_pair)

    def replace_plossl_pair(self, lens_index: int, part_number: str) -> None:
        """Replace both lenses of a Plossl pair with ``part_number`` (``replacement``)."""
        replacement.replace_plossl_pair(self.state, lens_index, self.make_lens(part_number))

    # ---------------------------------------------------------------- air gaps

    def air_gaps(self) -> list[gaps.AirGap]:
        """The air gap behind every lens (see ``gaps.air_gaps``)."""
        return gaps.air_gaps(self.state)

    def set_gap(self, lens_index: int, gap_mm: float) -> None:
        """Set the gap behind a lens, moving everything downstream (see ``gaps.set_gap``)."""
        gaps.set_gap(self.state, lens_index, gap_mm)

    def set_gap_locked(self, lens_index: int, locked: bool) -> None:
        """Lock or unlock the gap behind a lens (see ``gaps.set_gap_locked``)."""
        gaps.set_gap_locked(self.state, lens_index, locked)

    def optimise_gap(self, lens_index: int, goal: str) -> gap_optimisation.GapOptimisation:
        """Find the best gap for a goal, changing nothing (``gap_optimisation.optimise_gap``)."""
        return gap_optimisation.optimise_gap(self.state, lens_index, goal)

    def apply_gap_optimisation(
        self, result: gap_optimisation.GapOptimisation, lock: bool = False, flip: bool = False
    ) -> None:
        """Set the gap that was found (see ``gap_optimisation.apply_optimisation``)."""
        gap_optimisation.apply_optimisation(self.state, result, lock, flip)

    # ---------------------------------------------------------------- the source plane

    def source_gap(self) -> gaps.AirGap | None:
        """The gap from the source plane to the first lens (see ``gaps.source_gap``)."""
        return gaps.source_gap(self.state)

    def set_source_gap(self, gap_mm: float) -> None:
        """Set the gap from the source plane (see ``placement.set_source_gap``)."""
        placement.set_source_gap(self.state, gap_mm)

    def set_source_gap_locked(self, locked: bool) -> None:
        """Lock or unlock the gap from the source plane to the first lens.

        While it is locked the source plane moves with the first lens, keeping its distance.
        """
        self.state.source_gap_locked = locked

    def set_source_z(self, source_z: float) -> None:
        """Move the source plane to ``source_z`` mm (see ``placement.set_source_z``)."""
        placement.set_source_z(self.state, source_z)

    def lowest_first_front_z(self) -> float:
        """The smallest z the first lens may have (``placement.lowest_first_front_z``)."""
        return placement.lowest_first_front_z(self.state)

    # ---------------------------------------------------------------- moving

    def block_indices(self, lens_index: int) -> list[int]:
        """Indices of the lenses that move with this one: those joined to it by locked gaps."""
        return gaps.block_indices(self.state, lens_index)

    def move_lens(self, lens_index: int, requested_front_z: float) -> list[int] | None:
        """Move a lens and its block, as a drag does (see ``moving.move_lens``)."""
        return moving.move_lens(self.state, lens_index, requested_front_z)

    def swap_neighbouring_blocks(
        self, left_members: list[int], right_members: list[int]
    ) -> list[int]:
        """Exchange two adjacent blocks (see ``moving.swap_neighbouring_blocks``)."""
        return moving.swap_neighbouring_blocks(self.state, left_members, right_members)

    # ---------------------------------------------------------------- assemblies

    def assembly_indices(self, lens_index: int) -> list[int]:
        """Indices of the lenses in this lens's assembly (just itself if it is in none)."""
        return assemblies.assembly_indices(self.state, lens_index)

    def assembly_name(self, lens_index: int) -> str | None:
        """The name of this lens's assembly, or None if it is in none."""
        return assemblies.assembly_name(self.state, lens_index)

    def make_assembly(self, first_index: int, last_index: int, name: str | None = None) -> int:
        """Group a run of neighbouring lenses into an assembly (see ``assemblies``)."""
        return assemblies.make_assembly(self.state, first_index, last_index, name)

    def split_assembly(self, lens_index: int) -> None:
        """Dissolve this lens's assembly and unlock the gaps inside it."""
        assemblies.split_assembly(self.state, lens_index)

    def flip_assembly(self, lens_index: int) -> list[int]:
        """Turn this lens's assembly round as a whole; returns the new order of old indices."""
        return assemblies.flip_assembly(self.state, lens_index)

    def plossl_partner(self, lens_index: int) -> int | None:
        """A neighbour this lens could form a Plossl pair with (``assemblies.plossl_partner``)."""
        return assemblies.plossl_partner(self.state, lens_index)

    def make_plossl(self, lens_index: int) -> None:
        """Snap the lens and its partner into a Plossl pair (see ``assemblies.make_plossl``)."""
        assemblies.make_plossl(self.state, lens_index)

    # ---------------------------------------------------------------- user parts

    def save_assembly_as_part(self, lens_index: int, name: str, overwrite: bool = False) -> Path:
        """Save this lens's assembly as a user part in ``parts_folder``; return the file.

        Raises
        ------
        ValueError
            If the lens is in no assembly, the name is blank, or (unless ``overwrite``)
            there is already a part of that name.
        """
        if self.state.lenses[lens_index].assembly_id is None:
            raise ValueError("Only an assembly can be saved as a user part")
        members = self.assembly_indices(lens_index)
        part = user_parts.part_from_lenses(name, [self.state.lenses[index] for index in members])
        return user_parts.save_part(part, self.parts_folder, overwrite)

    def user_parts(self) -> list[UserPart]:
        """The user parts in ``parts_folder``, sorted by name (unreadable files are skipped)."""
        return user_parts.load_parts(self.parts_folder)[0]

    def user_part(self, name: str) -> UserPart:
        """The user part with this name.

        Raises
        ------
        ValueError
            If there is none.
        """
        for part in self.user_parts():
            if part.name == name:
                return part
        raise ValueError(f"There is no user part called {name!r}")

    # ---------------------------------------------------------------- fields

    def add_field(self) -> BeamField:
        """Add a field: a copy of the selected one, which then becomes the selected field.

        Starting from a copy means a new field is already sensible for the bench, and the
        user changes only what should differ. It gets the colour of its wavelength, even if
        the copied field had a colour of its own, so the two can be told apart once edited.

        Returns
        -------
        BeamField
            The new field (the last in ``state.fields``).
        """
        new_field = replace(self.state.selected_field, colour=None)
        self.state.fields.append(new_field)
        self.state.selected_field_index = len(self.state.fields) - 1
        return new_field

    def remove_fields(self, field_indices: list[int]) -> None:
        """Remove the given fields, always keeping at least one.

        If every field is listed, the first field is kept. The selection moves to the field
        just before the first one removed (or to the first field).

        Raises
        ------
        IndexError
            If any index is not a field; nothing is removed then.
        """
        fields = self.state.fields
        for field_index in field_indices:
            if not 0 <= field_index < len(fields):
                raise IndexError(f"There is no field {field_index}")
        to_remove = sorted(set(field_indices))
        if len(to_remove) >= len(fields):
            to_remove = to_remove[1:]  # keep the first field
        if not to_remove:
            return
        for index in reversed(to_remove):
            del fields[index]
        self.state.selected_field_index = min(max(to_remove[0] - 1, 0), len(fields) - 1)

    def select_field(self, field_index: int) -> None:
        """Make this field the one that autofocus and the single-field analyses use."""
        if not 0 <= field_index < len(self.state.fields):
            raise IndexError(f"There is no field {field_index}")
        self.state.selected_field_index = field_index

    # ---------------------------------------------------------------- optics

    def autofocus(self) -> float:
        """Move the image plane to the smallest RMS spot of the selected field, and return its z.

        Raises
        ------
        ValueError
            If the gap to the image plane is locked.
        """
        placement.refuse_if_image_gap_locked(self.state, "focus")
        self.state.image_z = analysis.find_best_focus_z(self.state)
        return self.state.image_z

    def trace_fast(self) -> analysis.FastTrace:
        """Every field's spot and rays, and the first-order summary (``analysis.trace_fast``)."""
        return analysis.trace_fast(self.state)

    def compute_f_number(self) -> analysis.FNumberResult | None:
        """f-number from the transmitted beam (see ``analysis.compute_f_number``)."""
        return analysis.compute_f_number(self.state)

    def compute_through_focus(self, fast_trace: analysis.FastTrace) -> analysis.ThroughFocusResult:
        """RMS radius against image plane position (see ``analysis.compute_through_focus``)."""
        return analysis.compute_through_focus(self.state, fast_trace)

    def compute_ray_fan(self) -> analysis.RayFanResult:
        """Transverse ray aberration (see ``analysis.compute_ray_fan``)."""
        return analysis.compute_ray_fan(self.state)

    def compute_mtf(self) -> analysis.MtfResult:
        """FFT MTF of the selected field (see ``analysis.compute_mtf``)."""
        return analysis.compute_mtf(self.state)

    def compute_field_curvature(self) -> analysis.FieldCurvatureResult:
        """Best focus against field angle (see ``analysis.compute_field_curvature``)."""
        return analysis.compute_field_curvature(self.state)

    # ---------------------------------------------------------------- undo and redo

    def record_undo_step(self, merge_key: Hashable | None = None) -> None:
        """Make the changes since the last step undoable as one (see ``UndoHistory.record``)."""
        self.history.record(self.state, merge_key)

    def amend_undo_step(self) -> None:
        """Take the bench as it is into the current step, without making a new one."""
        self.history.amend(self.state)

    def clear_undo_history(self) -> None:
        """Forget every undo step: the bench as it is now becomes the starting point."""
        self.history.reset(self.state)

    def undo(self) -> bool:
        """Go back one step. Returns False (changing nothing) if there is none."""
        return self._show_history_state(self.history.undo())

    def redo(self) -> bool:
        """Redo the step last undone. Returns False (changing nothing) if there is none."""
        return self._show_history_state(self.history.redo())

    def _show_history_state(self, state: OpticalState | None) -> bool:
        if state is None:
            return False
        self.state = state
        return True

    # ---------------------------------------------------------------- files

    def has_unsaved_changes(self) -> bool:
        """True if the design differs from the one last saved, opened, imported or started.

        Which field is selected is saved in a project file, but choosing another one does
        not count as a change. Nor does an image plane moved by less than
        ``history.IMAGE_Z_TOLERANCE_MM``, as "Lock focus" refocusing an unchanged bench does.
        """
        return not same_design(self.state, self.saved_state)

    def mark_saved(self) -> None:
        """Remember the current design as the saved one.

        Called by ``save_project``, ``load_project``, ``load_zmx`` and ``new_design``. A
        front end calls it itself after changes that count as part of starting up (e.g. the
        first autofocus).
        """
        self.saved_state = copy.deepcopy(self.state)

    def new_design(self) -> None:
        """Start again: no lenses, one default field, no project file.

        Counts as saved; clears undo.
        """
        self.state = OpticalState()
        self.project_path = None
        self.mark_saved()
        self.clear_undo_history()

    def save_project(
        self, file_path: str | Path | None = None, forget_import: bool = False
    ) -> Path:
        """Save the whole design as a RayCrafter project file, and count it as saved.

        Parameters
        ----------
        file_path : str or Path, optional
            Where to save. Defaults to ``project_path``, the file last saved or opened.
            ``project_file.FILE_EXTENSION`` is added if the name has no extension.
        forget_import : bool
            If True, drop the record of the file the design was imported from
            (``OpticalState.imported_from``), so that from now on it is a design of its own
            rather than an import. The record is dropped only if the save succeeds.

        Returns
        -------
        Path
            The file written, which becomes ``project_path``.

        Raises
        ------
        ValueError
            If no path is given and the design has no project file yet.
        """
        if file_path is None:
            if self.project_path is None:
                raise ValueError("The design has no project file yet, so give a file path.")
            file_path = self.project_path
        target = Path(file_path)
        if not target.suffix:
            target = target.with_suffix(project_file.FILE_EXTENSION)
        state_to_save = replace(self.state, imported_from=None) if forget_import else self.state
        project_file.save_project(state_to_save, target)
        self.state.imported_from = state_to_save.imported_from
        if forget_import:
            self.history.forget_imported_source()  # or undo would bring the record back
        self.project_path = target
        self.mark_saved()
        return target

    def load_project(self, file_path: str | Path) -> None:
        """Replace the design with the one in a project file, clearing the undo history.

        Raises
        ------
        project_file.ProjectFileError
            If the file cannot be read or is not a valid project. The design is then left
            as it was.
        """
        self.state = project_file.load_project(file_path)
        self.project_path = Path(file_path)
        self.mark_saved()
        self.clear_undo_history()

    def save_zmx(self, file_path: str) -> None:
        """Export the current system to a Zemax .zmx file.

        This does not count as saving the design: a ZMX file keeps only the selected field
        and no assemblies or part metadata, so the design would be lost on quitting.
        """
        optic_builder.save_zmx(self.state, file_path)

    def load_zmx(self, file_path: str) -> list[str]:
        """Replace the bench with the system in a Zemax .zmx file, clearing the undo history.

        The imported design counts as saved, and has no project file until it is saved.

        Returns
        -------
        list of str
            Notes on anything that was simplified (see ``zmx_import.load_zmx``).

        Raises
        ------
        zmx_import.ZmxImportError
            If the file cannot be read or describes something RayCrafter cannot show. The
            bench is then left as it was.
        """
        imported = zmx_import.load_zmx(file_path, self.catalog)
        self.state = imported.state
        self.project_path = None
        self.mark_saved()
        self.clear_undo_history()
        return imported.notes
