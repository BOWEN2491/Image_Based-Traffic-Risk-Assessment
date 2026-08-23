"""Convert Ultralytics text detections to small JSON records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def convert(source: Path, labels: Path, output: Path) -> int:
    images = {path.stem: path for path in source.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}}
    output.mkdir(parents=True, exist_ok=True)
    converted = 0
    for label_path in labels.glob("*.txt"):
        image_path = images.get(label_path.stem)
        if image_path is None:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        detections = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id, cx, cy, box_width, box_height = map(float, parts[:5])
            detections.append({"class_id": int(class_id), "bbox_cxcywh": [cx, cy, box_width, box_height], "confidence": float(parts[5]) if len(parts) > 5 else None})
        (output / f"{label_path.stem}.json").write_text(json.dumps({"image_size": [width, height], "detections": detections}, indent=2), encoding="utf-8")
        converted += 1
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Converted {convert(args.source, args.labels, args.output)} label files")


if __name__ == "__main__":
    main()
