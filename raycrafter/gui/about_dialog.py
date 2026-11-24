"""The "About RayCrafter" box: the app icon, name, version and a link to the project."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

import raycrafter
from raycrafter.gui.style import make_app_icon

PROJECT_URL = "https://github.com/raycrafter-optical-design/raycrafter"
ICON_SIZE_PX = 96


class AboutDialog(QDialog):
    """A small box with the icon on the left and the name, version and link on the right."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About RayCrafter")

        icon_label = QLabel()
        icon_label.setPixmap(make_app_icon().pixmap(ICON_SIZE_PX, ICON_SIZE_PX))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignTop)

        # The version comes from the git tag when the package is installed (setuptools-scm).
        version = getattr(raycrafter, "__version__", "unknown")
        self.text_label = QLabel(
            "<h2>RayCrafter</h2>"
            f"<p>Version {version}</p>"
            "<p>Lens-level optical design: place off-the-shelf lenses on an optical axis<br>"
            "and see spots, MTF and aberrations update live.</p>"
            f'<p><a href="{PROJECT_URL}">{PROJECT_URL}</a></p>'
        )
        self.text_label.setTextFormat(Qt.TextFormat.RichText)
        self.text_label.setOpenExternalLinks(True)  # the link opens in the web browser

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)  # Close has the "reject" role

        content_row = QHBoxLayout()
        content_row.setSpacing(18)
        content_row.addWidget(icon_label)
        content_row.addWidget(self.text_label)
        layout = QVBoxLayout(self)
        layout.addLayout(content_row)
        layout.addWidget(buttons)
