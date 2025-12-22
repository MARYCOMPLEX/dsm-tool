"""
Tab Plugin Base Module

Defines the abstract base class for all tab plugins.
"""

from abc import ABC, abstractmethod
from typing import Optional

from PyQt6.QtWidgets import QWidget


class TabPlugin(ABC):
    """
    Abstract base class for tab plugins.

    Each plugin provides one tab in the main window.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Stable identifier for this tab (e.g. 'input', 'output')."""
        ...

    @property
    @abstractmethod
    def title_en(self) -> str:
        """Tab title (English)."""
        ...

    @property
    @abstractmethod
    def title_zh(self) -> str:
        """Tab title (Chinese)."""
        ...

    @abstractmethod
    def create_widget(self, main_window: "MainWindow") -> QWidget:
        """
        Create the central widget for this tab.

        Args:
            main_window: The application's MainWindow instance. Plugins
                can use it to access shared viewers, state, etc.
        """
        ...

    def on_language_changed(self, lang: str) -> None:
        """
        Called when UI language changes ('en' or 'zh').

        Default implementation does nothing. Override if your tab has
        internal texts that need to be updated.
        """
        _ = lang
        return




