"""
Output Preview Tab Plugin

Provides the tab that displays one or two output images side by side.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSplitter
from PyQt6.QtCore import Qt

from tabs.base import TabPlugin


class OutputPreviewTab(TabPlugin):
    """Tab showing up to two selected output images."""

    @property
    def id(self) -> str:
        return "output_preview"

    @property
    def title_en(self) -> str:
        return "Output Preview"

    @property
    def title_zh(self) -> str:
        return "输出预览"

    def create_widget(self, main_window: "MainWindow") -> QWidget:  # type: ignore[name-defined]
        """
        Build a splitter that hosts the two pre-created output viewer containers.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal, container)
        splitter.addWidget(main_window._output_viewer1_container)  # noqa: SLF001
        splitter.addWidget(main_window._output_viewer2_container)  # noqa: SLF001
        splitter.setSizes([500, 500])

        layout.addWidget(splitter)
        return container




