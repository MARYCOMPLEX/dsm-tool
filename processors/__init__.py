"""
Processors Package

Automatically discovers and registers all Processor implementations.
"""

import importlib
import pkgutil
from pathlib import Path
from typing import List, Type

from processors.base import Processor


def discover_processors() -> List[Type[Processor]]:
    """
    Automatically discover all Processor subclasses in the processors package.
    
    Returns:
        List of Processor subclasses
    """
    processors = []
    
    # Get the processors package directory
    processors_dir = Path(__file__).parent
    
    # Import all modules in the processors package
    for finder, name, ispkg in pkgutil.iter_modules([str(processors_dir)]):
        if name.startswith('_'):
            continue
        
        try:
            module = importlib.import_module(f"processors.{name}")
            
            # Find all Processor subclasses in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, Processor) and
                    attr is not Processor):
                    processors.append(attr)
        
        except Exception as e:
            print(f"Warning: Failed to import processor module '{name}': {e}")
            continue
    
    return processors


# Auto-discover processors on import
AVAILABLE_PROCESSORS = discover_processors()

