"""Download v0.1.0 model assets and verify every SHA-256 digest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from importlib.resources import files
from pathlib import Path


MANIFEST = files("tools").joinpath("model_manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination_root: Path, force: bool = False) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for asset in manifest["assets"]:
        destination = destination_root / asset["destination"]
        if destination.is_file() and sha256(destination) == asset["sha256"]:
            print(f"Verified {destination}")
            continue
        if destination.exists() and not force:
            raise RuntimeError(f"Digest mismatch for {destination}; use --force to replace it")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            with urllib.request.urlopen(asset["url"], timeout=60) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            if partial.stat().st_size != asset["size"] or sha256(partial) != asset["sha256"]:
                raise RuntimeError(f"Downloaded asset failed verification: {asset['name']}")
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
        print(f"Downloaded and verified {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path.cwd() / "models")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(args.destination, args.force)


if __name__ == "__main__":
    main()
