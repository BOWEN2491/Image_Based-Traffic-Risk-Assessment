"""List images which do not yet have an Ultralytics label file."""

from __future__ import annotations

import argparse
from pathlib import Path


def find_missing(source: Path, labels: Path) -> list[Path]:
    completed = {path.stem for path in labels.glob("*.txt")}
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and path.stem not in completed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    missing = find_missing(args.source, args.labels)
    args.output.write_text("\n".join(map(str, missing)), encoding="utf-8")
    print(f"Wrote {len(missing)} paths to {args.output}")


if __name__ == "__main__":
    main()
