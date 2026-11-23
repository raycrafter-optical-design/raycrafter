"""Start the RayCrafter GUI. Installed as the ``raycrafter`` command."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from raycrafter.gui.main_window import MainWindow
from raycrafter.gui.style import make_app_icon


def main() -> None:
    """Create the Qt application, show the main window and run until it is closed."""
    app = QApplication(sys.argv)
    app.setApplicationName("RayCrafter")
    app.setWindowIcon(make_app_icon())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
