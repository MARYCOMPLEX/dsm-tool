"""
Input Images Tab Plugin

Provides the tab that displays the two input images side by side.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSplitter
from PyQt6.QtCore import Qt

from tabs.base import TabPlugin


class InputImagesTab(TabPlugin):
    """Tab showing the two input images."""

    @property
    def id(self) -> str:
        return "input_images"

    @property
    def title_en(self) -> str:
        return "Input Images"

    @property
    def title_zh(self) -> str:
        return "输入影像"

    def create_widget(self, main_window: "MainWindow") -> QWidget:  # type: ignore[name-defined]
        """
        Build a splitter that hosts the two pre-created input viewer containers.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal, container)
        # These containers are created and owned by MainWindow
        splitter.addWidget(main_window._viewer1_container)  # noqa: SLF001
        splitter.addWidget(main_window._viewer2_container)  # noqa: SLF001
        splitter.setSizes([500, 500])

        layout.addWidget(splitter)
        return container




