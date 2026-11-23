"""Start the RayCrafter GUI. Installed as the ``raycrafter`` command."""

from __future__ import annotations

import sys


def main() -> None:
    """Create the Qt application, show the main window and run until it is closed.

    Start-up can be slow, so progress is printed to the terminal. The heavy imports are done
    inside this function, after the first message, so the user sees output straight away
    instead of a silent pause.
    """
    print("Starting RayCrafter: loading libraries...", flush=True)
    from PyQt6.QtWidgets import QApplication

    from raycrafter.gui.compile_notice import announce_numba_compilation
    from raycrafter.gui.main_window import MainWindow
    from raycrafter.gui.style import make_app_icon

    app = QApplication(sys.argv)
    app.setApplicationName("RayCrafter")
    app.setWindowIcon(make_app_icon())

    # The first ray trace can make numba compile the optics code, which takes minutes. It does so
    # only when the compiled code is not already cached on disk, so only then is the user told.
    print("Setting up the optical system...", flush=True)
    with announce_numba_compilation():
        window = MainWindow()
    window.show()
    print("Ready.", flush=True)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
