"""Generate weak labels from an explicit feature CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .weak_label import generate_weak_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labeled = generate_weak_label(pd.read_csv(args.features))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
