"""
Tabs Package

Automatically discovers and registers all tab plugins.
"""

import importlib
import pkgutil
from pathlib import Path
from typing import List, Type

from tabs.base import TabPlugin
from tabs.config import TAB_ORDER


def discover_tabs() -> List[Type[TabPlugin]]:
    """
    Automatically discover all TabPlugin subclasses in the tabs package.

    Returns:
        List of TabPlugin subclasses.
    """
    tabs: List[Type[TabPlugin]] = []
    tabs_dir = Path(__file__).parent

    for _, name, ispkg in pkgutil.iter_modules([str(tabs_dir)]):
        if name.startswith("_"):
            continue

        try:
            module = importlib.import_module(f"tabs.{name}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Failed to import tab module '{name}': {exc}")
            continue

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, TabPlugin)
                and attr is not TabPlugin
            ):
                tabs.append(attr)

    # Order tabs: first by TAB_ORDER, then remaining by class name
    order_index = {tab_id: idx for idx, tab_id in enumerate(TAB_ORDER)}

    def sort_key(cls: Type[TabPlugin]) -> tuple[int, str]:
        try:
            tmp = cls()
            tab_id = tmp.id
        except Exception:
            tab_id = cls.__name__
        primary = order_index.get(tab_id, len(order_index))
        return primary, cls.__name__

    tabs.sort(key=sort_key)
    return tabs


AVAILABLE_TABS = discover_tabs()


