"""Public traffic_risk package (v0.2 API)."""

__version__ = "0.2.2"
__all__ = ["Settings", "create_app"]


def __getattr__(name):
    # Diagnostics must run even when runtime dependencies/configuration are broken.
    if name == "Settings":
        from .config import Settings
        return Settings
    if name == "create_app":
        from .app import create_app
        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

