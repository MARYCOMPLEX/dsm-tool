# tabs/dl_inspect.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from tabs.base import TabPlugin


class DLInspectTab(TabPlugin):
    @property
    def id(self) -> str:
        return "About"

    @property
    def title_en(self) -> str:
        return "About"

    @property
    def title_zh(self) -> str:
        return "关于"

    def create_widget(self, main_window: "MainWindow") -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("关于"))
        return w
