"""Image-based traffic scene risk assessment research demo."""

__version__ = "0.1.0"
"""Public traffic_risk package (v0.2 API)."""

from .app import create_app
from .config import Settings

__all__ = ["Settings", "create_app"]

