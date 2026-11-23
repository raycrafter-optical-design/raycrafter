"""Start the RayCrafter GUI. Installed as the ``raycrafter`` command."""

from __future__ import annotations

import sys


def main() -> None:
    """Create the Qt application, show the main window and run until it is closed.

    Start-up can take minutes the first time (see the messages below), so progress is printed
    to the terminal. The heavy imports are done inside this function, after the first message,
    so the user sees output straight away instead of a silent pause.
    """
    print("Starting RayCrafter: loading libraries...", flush=True)
    from PyQt6.QtWidgets import QApplication

    from raycrafter.gui.main_window import MainWindow
    from raycrafter.gui.style import make_app_icon

    app = QApplication(sys.argv)
    app.setApplicationName("RayCrafter")
    app.setWindowIcon(make_app_icon())

    # The first ray trace makes numba compile the optics code to machine code. This can take
    # several minutes on the first run only: the result is cached on disk for later runs.
    print(
        "Setting up the optical system. On the first run this compiles the ray-tracing code\n"
        "and can take several minutes with no further output. It is not stuck, so please\n"
        "wait. Later runs start much faster.",
        flush=True,
    )
    window = MainWindow()
    window.show()
    print("Ready.", flush=True)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
