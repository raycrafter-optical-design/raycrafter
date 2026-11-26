"""The File menu of the main window: New, Open, Save, Save As, Import ZMX and Export ZMX.

A design is saved as a RayCrafter project (``raycrafter.core.project_file``), which keeps
everything. A Zemax file is a lossy exchange format: Import ZMX replaces the design with the
file's system, and Export ZMX writes the system out but does not count as saving the design,
so quitting after an export still asks to save. The window title shows the project file's
name, and marks it as modified when there are unsaved changes.

An imported design keeps the original ZMX file and the bench as imported, in its project
file too. "Save Without Original ZMX" saves the project without them, so from then on the
design is no longer an import.

New, Open, Import ZMX and Quit ask first if the design has unsaved changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QFileDialog, QMenu, QMessageBox

from raycrafter.core.project_file import FILE_EXTENSION, ProjectFileError
from raycrafter.core.zmx_import import ZmxImportError

if TYPE_CHECKING:
    from raycrafter.gui.main_window import MainWindow

PROJECT_FILTER = f"RayCrafter projects (*{FILE_EXTENSION})"
ZMX_FILTER = "Zemax files (*.zmx)"
UNTITLED_NAME = "Untitled"
STATUS_MESSAGE_MS = 5000


class FileActions:
    """Carries out the File menu on the window's model, asking the user where needed.

    Parameters
    ----------
    window : MainWindow
        The window whose design is saved and replaced, and which is redrawn afterwards.
    """

    def __init__(self, window: MainWindow) -> None:
        self.window = window
        self.save_without_import_action: QAction | None = None  # set by build_menu

    def build_menu(self, file_menu: QMenu) -> None:
        """Fill the window's File menu with these actions."""

        def add_action(text: str, slot, shortcut=None) -> QAction:
            action = file_menu.addAction(text)
            action.triggered.connect(slot)
            if shortcut is not None:
                action.setShortcut(QKeySequence(shortcut))
            return action

        add_action("New", self.new_design, QKeySequence.StandardKey.New)
        add_action("Open...", self.open_project, QKeySequence.StandardKey.Open)
        add_action("Save", self.save, QKeySequence.StandardKey.Save)
        add_action("Save As...", self.save_as, QKeySequence.StandardKey.SaveAs)
        self.save_without_import_action = add_action(
            "Save Without Original ZMX...", self.save_without_import
        )
        file_menu.addSeparator()
        add_action("Import ZMX...", self.import_zmx)
        add_action("Export ZMX...", self.export_zmx)

    @property
    def model(self):
        """The window's model (looked up each time, so it is never a stale reference)."""
        return self.window.model

    # ------------------------------------------------------------------ asking first

    def confirm_discard_changes(self, action_text: str) -> bool:
        """If there are unsaved changes, ask whether to save them first.

        Parameters
        ----------
        action_text : str
            What is about to happen, e.g. "starting a new design".

        Returns
        -------
        bool
            True to go ahead (nothing to lose, saved, or discarded); False to stop.
        """
        if not self.model.has_unsaved_changes():
            return True
        answer = QMessageBox.question(
            self.window,
            "Unsaved changes",
            f"The design has changed since it was last saved. Save it before {action_text}?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    # --------------------------------------------------------------- project files

    def new_design(self) -> None:
        """File > New: an empty bench with one default field."""
        if not self.confirm_discard_changes("starting a new design"):
            return
        self.model.new_design()
        self._show_replaced_design()

    def open_project(self) -> None:
        """File > Open: replace the design with a saved project."""
        if not self.confirm_discard_changes("opening another design"):
            return
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self.window, "Open RayCrafter project", self._start_folder(), PROJECT_FILTER
        )
        if not file_path:
            return  # dialog cancelled
        try:
            self.model.load_project(file_path)
        except ProjectFileError as error:
            QMessageBox.critical(
                self.window, "Open failed", f"Could not open {file_path}:\n{error}"
            )
            return
        self._show_replaced_design()
        self._show_status(f"Opened {file_path}")

    def save(self) -> bool:
        """File > Save: save to the project file, asking where if there is none yet.

        Returns
        -------
        bool
            True if the design was saved.
        """
        if self.model.project_path is None:
            return self.save_as()
        return self._save_to(self.model.project_path)

    def save_as(self) -> bool:
        """File > Save As: ask where to save the project, then save it there.

        Returns
        -------
        bool
            True if the design was saved.
        """
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self.window, "Save RayCrafter project", self._suggested_path(), PROJECT_FILTER
        )
        if not file_path:
            return False  # dialog cancelled
        if not file_path.lower().endswith(FILE_EXTENSION):
            file_path += FILE_EXTENSION
        return self._save_to(Path(file_path))

    def save_without_import(self) -> bool:
        """File > Save Without Original ZMX: Save As, dropping the record of the import.

        The project then keeps neither the original ZMX file nor the bench as imported, and
        the design stops being an import. The user is asked first, as this cannot be undone
        from the saved file.

        Returns
        -------
        bool
            True if the design was saved.
        """
        source = self.model.state.imported_from
        if source is None:
            return self.save_as()
        answer = QMessageBox.question(
            self.window,
            "Save without the original ZMX",
            f"The project will no longer keep {source.file_name} or the design as it was "
            "imported, so it cannot be compared with the import or imported again. "
            "Save without them?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer != QMessageBox.StandardButton.Save:
            return False
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self.window, "Save RayCrafter project", self._suggested_path(), PROJECT_FILTER
        )
        if not file_path:
            return False  # dialog cancelled
        if not file_path.lower().endswith(FILE_EXTENSION):
            file_path += FILE_EXTENSION
        return self._save_to(Path(file_path), forget_import=True)

    def _save_to(self, file_path: Path, forget_import: bool = False) -> bool:
        try:
            self.model.save_project(file_path, forget_import=forget_import)
        except Exception as error:  # show any problem (e.g. permissions) instead of crashing
            QMessageBox.critical(
                self.window, "Save failed", f"Could not save {file_path}:\n{error}"
            )
            return False
        self.update_title()
        self._show_status(f"Saved {file_path}")
        return True

    # ------------------------------------------------------------------- Zemax files

    def import_zmx(self) -> None:
        """File > Import ZMX: replace the design with the system in a Zemax file."""
        if not self.confirm_discard_changes("importing another design"):
            return
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self.window, "Import Zemax file", self._start_folder(), ZMX_FILTER
        )
        if not file_path:
            return  # dialog cancelled
        try:
            notes = self.model.load_zmx(file_path)
        except ZmxImportError as error:
            QMessageBox.critical(
                self.window, "Import failed", f"Could not import {file_path}:\n{error}"
            )
            return
        self._show_replaced_design()
        if notes:
            QMessageBox.information(
                self.window,
                "Imported with changes",
                "\n\n".join(["Some details were simplified:"] + notes),
            )
        self._show_status(f"Imported {file_path}")

    def export_zmx(self) -> bool:
        """File > Export ZMX: write the system as a Zemax file (the design stays unsaved).

        Returns
        -------
        bool
            True if the file was written.
        """
        suggested = Path(self._suggested_path()).with_suffix(".zmx")
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self.window, "Export Zemax file", str(suggested), ZMX_FILTER
        )
        if not file_path:
            return False  # dialog cancelled
        if not file_path.lower().endswith(".zmx"):
            file_path += ".zmx"
        try:
            self.model.save_zmx(file_path)
        except Exception as error:  # show any export problem instead of crashing the GUI
            QMessageBox.critical(
                self.window, "Export failed", f"Could not export {file_path}:\n{error}"
            )
            return False
        self._show_status(f"Exported {file_path}")
        return True

    # ---------------------------------------------------------------------- helpers

    def update_title(self) -> None:
        """Show the project's name in the title bar, marked if there are unsaved changes.

        Comparing the design with the saved one is too slow for every drag event, so the
        window calls this once a change is complete.
        """
        project_path = self.model.project_path
        name = project_path.name if project_path is not None else UNTITLED_NAME
        # "[*]" is where Qt shows the modified mark (an asterisk; a dot in the close button
        # on macOS).
        self.window.setWindowTitle(f"{name}[*] - RayCrafter")
        self.window.setWindowModified(self.model.has_unsaved_changes())
        if self.save_without_import_action is not None:
            self.save_without_import_action.setEnabled(self.model.state.imported_from is not None)

    def _show_replaced_design(self) -> None:
        self.window.show_replaced_design()
        self.update_title()

    def _show_status(self, message: str) -> None:
        self.window.statusBar().showMessage(message, STATUS_MESSAGE_MS)

    def _start_folder(self) -> str:
        """The folder of the current project, where the file dialogs open."""
        project_path = self.model.project_path
        return str(project_path.parent) if project_path is not None else ""

    def _suggested_path(self) -> str:
        """The project file's path, or a name based on the imported file, for Save As."""
        if self.model.project_path is not None:
            return str(self.model.project_path)
        source = self.model.state.imported_from
        stem = Path(source.file_name).stem if source is not None else "lens_system"
        return stem + FILE_EXTENSION
