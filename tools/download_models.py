"""Compatibility wrapper for :mod:`traffic_risk.download_models`.

The implementation and packaged manifest live in ``traffic_risk`` so wheel
installs and source checkouts cannot drift.  This module remains for existing
research scripts that import ``tools.download_models``.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from traffic_risk.download_models import download as _download


MANIFEST = files("traffic_risk").joinpath("model_manifest.json")


def download(destination_root: Path, force: bool = False) -> None:
    _download(destination_root, force, manifest_path=Path(str(MANIFEST)))


def main() -> None:
    from traffic_risk.download_models import main as package_main

    package_main()


if __name__ == "__main__":
    main()
