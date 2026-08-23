"""Public traffic_risk package (v0.2 API)."""

from .app import create_app
from .config import Settings

__version__ = "0.2.0"
__all__ = ["Settings", "create_app"]

