import importlib
from typing import Any

from tensorcas.errors import AdapterError


def require(library: str) -> Any:
    """Import and return library, raising AdapterError if not installed."""
    try:
        return importlib.import_module(library)
    except ImportError:
        raise AdapterError(f"{library} is not installed")
