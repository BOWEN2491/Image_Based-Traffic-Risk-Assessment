"""Build bounded, deterministic, source-grouped exploratory evaluation manifests.

Source files are read only. Local annotations have not been authenticated against
an upstream archive, so these manifests explicitly prohibit model promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from PIL import Image

from .sample_identity import sha256_file

COLORS = ("green", "red", "unknown", "yellow")
PROVENANCE = "local-unverified"


def stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def group_split(group: str, source_split: str, seed: int) -> str:
    fraction = int(stable_key(group, seed)[:8], 16) / 2**32
    if source_split == "val":
        return "development" if fraction < 0.5 else "holdout"
    return "train" if fraction < 0.8 else "train-validation"


def pixel_hash(image: Image.Image) -> str:
    return hashlib.sha256(f"RGB:{image.size}:".encode() + image.tobytes()).hexdigest()


def _load_record(root: Path, split: str, path: Path) -> tuple[dict, Image.Image]:
    identifier = path.stem
    labels = root / "bdd100k_labels" / "100k" / split / f"{identifier}.json"
    payload = json.loads(labels.read_text(encoding="utf-8-sig"))
    if Path(payload.get("name", "")).stem != identifier:
        raise ValueError("annotation image name mismatch")
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != 1 or not isinstance(frames[0].get("objects"), list):
        raise ValueError("expected one annotated frame with an objects array")
    with Image.open(path) as source:
        source.load()
        image = source.convert("RGB")
    width, height = image.size
    objects = []
    for index, obj in enumerate(frames[0]["objects"]):
        if "box2d" not in obj:  # Segmentation-only objects are not box ground truth.
            continue
        box = obj["box2d"]
        coords = [float(box[key]) for key in ("x1", "y1", "x2", "y2")]
        x1, y1, x2, y2 = coords
        if not all(math.isfinite(v) for v in coords) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("invalid or out-of-image bounding box")
        category = obj.get("category")
        if not isinstance(category, str):
            raise ValueError("object category must be text")
        color = obj.get("attributes", {}).get("trafficLightColor", "unknown")
        if color in {"none", None, ""}:
            color = "unknown"
        if category == "traffic light" and color not in COLORS:
            raise ValueError("unsupported traffic light color")
        objects.append({"object_id": str(obj.get("id", index)), "category": category,
                        "bbox": coords, "color": color})
    info = root / "bdd100k_info" / "100k" / split / f"{identifier}.json"
    group, grouping = f"image:{identifier}", "image-id-fallback"
    info_hash = None
    if info.is_file():
        metadata = json.loads(info.read_text(encoding="utf-8-sig"))
        ride = metadata.get("rideID")
        if isinstance(ride, str) and ride:
            group, grouping = f"ride:{ride}", "rideID"
        info_hash = sha256_file(info)
    record = {"sample_id": f"{split}/{identifier}", "image_id": identifier,
              "source_split": split, "image_path": path.relative_to(root).as_posix(),
              "label_path": labels.relative_to(root).as_posix(),
              "image_sha256": sha256_file(path), "label_sha256": sha256_file(labels),
              "pixel_sha256": pixel_hash(image), "info_sha256": info_hash,
              "width": width, "height": height, "group_id": group, "grouping": grouping,
              "provenance": PROVENANCE, "promotion_allowed": False, "objects": objects}
    return record, image


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def prepare(root: Path, output: Path, *, seed: int = 42,
            max_images: int = 1000, max_crops: int = 2000,
            max_train_images: int = 10000, max_seconds: float = 1800) -> dict:
    root, output = root.resolve(), output.resolve()
    if not 1 <= max_images <= 1000 or not 4 <= max_crops <= 2000:
        raise ValueError("max_images must be 1..1000 and max_crops 4..2000")
    if not 1 <= max_train_images <= 10000 or not 0 < max_seconds <= 1800:
        raise ValueError("max_train_images must be 1..10000 and max_seconds in (0, 1800]")
    if output == root or root in output.parents:
        raise ValueError("output must be outside the read-only dataset")
    if output.exists():
        raise FileExistsError("Use a new output directory to preserve fixed experiment manifests")
    for split in ("train", "val"):
        if not (root / "bdd100_images_100k" / "100k" / split).is_dir():
            raise ValueError(f"missing {split} image directory")
    output.mkdir(parents=True)
    images, crops, excluded = [], [], []
    seen_images, seen_crops, eval_groups = set(), set(), set()
    train_counts = Counter()
    selected_counts = Counter()
    scanned_counts = Counter()
    started = time.monotonic()
    stop_reason = "completed available samples or crop limits"

    def crop_objects(record: dict, image: Image.Image, training: bool) -> None:
        for index, obj in enumerate(record["objects"]):
            if obj["category"] != "traffic light":
                continue
            color = obj["color"]
            if training and train_counts[color] >= max_crops // 4:
                continue
            x1, y1, x2, y2 = obj["bbox"]
            box = [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]
            if box[2] - box[0] < 6 or box[3] - box[1] < 6:
                continue
            crop = image.crop(box)
            digest = pixel_hash(crop)
            if digest in seen_crops:
                excluded.append({"sample_id": record["sample_id"], "object_id": obj["object_id"],
                                 "reason": "duplicate crop pixels"})
                continue
            seen_crops.add(digest)
            relative = Path("crops") / record["split"] / color / f"{record['image_id']}-{index}.png"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            crop.save(target)
            crops.append({"sample_id": f"{record['sample_id']}/{index}",
                          "source_image_id": record["sample_id"], "object_id": obj["object_id"],
                          "group_id": record["group_id"], "split": record["split"],
                          "source_split": record["source_split"], "label": color, "bbox": box,
                          "path": relative.as_posix(), "sha256": sha256_file(target),
                          "pixel_sha256": digest, "parent_image_sha256": record["image_sha256"],
                          "parent_label_sha256": record["label_sha256"],
                          "provenance": PROVENANCE, "promotion_allowed": False})
            if training:
                train_counts[color] += 1

    # Reserve evaluation groups and pixel hashes before considering training data.
    for source_split in ("val", "train"):
        directory = root / "bdd100_images_100k" / "100k" / source_split
        paths = sorted(directory.glob("*.jpg"), key=lambda p: stable_key(p.stem, seed))
        for path in paths:
            if time.monotonic() - started >= max_seconds:
                stop_reason = "preparation time budget reached"
                break
            if source_split == "train" and scanned_counts["train"] >= max_train_images:
                stop_reason = "training scan limit reached"
                break
            if source_split == "val" and selected_counts["val"] >= max_images:
                break
            if source_split == "train" and all(train_counts[c] >= max_crops // 4 for c in COLORS):
                break
            scanned_counts[source_split] += 1
            if scanned_counts[source_split] % 250 == 0:
                print(json.dumps({"scanning": source_split, "scanned": scanned_counts[source_split],
                                  "selected": selected_counts[source_split], "training_crops": dict(train_counts)}), flush=True)
            try:
                if source_split == "train":
                    # Avoid decoding large images once their color quotas are full.
                    annotation = root / "bdd100k_labels" / "100k" / "train" / f"{path.stem}.json"
                    data = json.loads(annotation.read_text(encoding="utf-8-sig"))
                    objects = data["frames"][0]["objects"]
                    def needed(obj):
                        color = obj.get("attributes", {}).get("trafficLightColor", "unknown")
                        color = "unknown" if color in {None, "", "none"} else color
                        return obj.get("category") == "traffic light" and color in COLORS and train_counts[color] < max_crops // 4
                    if not any(needed(obj) for obj in objects):
                        continue
                record, image = _load_record(root, source_split, path)
            except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError, Image.DecompressionBombError) as exc:
                excluded.append({"sample_id": f"{source_split}/{path.stem}", "reason": str(exc)})
                continue
            if source_split == "train" and record["group_id"] in eval_groups:
                excluded.append({"sample_id": record["sample_id"], "reason": "group reserved for evaluation"})
                continue
            if record["pixel_sha256"] in seen_images:
                excluded.append({"sample_id": record["sample_id"], "reason": "duplicate image pixels"})
                continue
            if source_split == "train" and not any(
                obj["category"] == "traffic light" and train_counts[obj["color"]] < max_crops // 4
                for obj in record["objects"]
            ):
                continue
            seen_images.add(record["pixel_sha256"])
            record["split"] = group_split(record["group_id"], source_split, seed)
            if source_split == "val":
                eval_groups.add(record["group_id"])
            images.append(record)
            selected_counts[source_split] += 1
            crop_objects(record, image, source_split == "train")
    _write_jsonl(output / "images.jsonl", images)
    _write_jsonl(output / "crops.jsonl", crops)
    _write_jsonl(output / "excluded.jsonl", excluded)
    summary = {"schema_version": 1, "seed": seed, "provenance": PROVENANCE,
               "promotion_allowed": False, "promotion_blockers": ["annotation source archive not authenticated"],
               "max_images": max_images, "max_training_crops": max_crops,
               "max_train_images": max_train_images, "max_seconds": max_seconds,
               "scanned_counts": dict(scanned_counts), "stop_reason": stop_reason,
               "training_class_shortfalls": {c: max_crops // 4 - train_counts[c] for c in COLORS},
               "image_counts": dict(Counter(r["split"] for r in images)),
               "crop_counts": dict(Counter(f"{r['split']}/{r['label']}" for r in crops)),
               "training_class_counts": dict(train_counts), "exclusions": len(excluded),
               "fallback_groups": sum(r["grouping"] != "rideID" for r in images),
               "manifests": {p.name: sha256_file(p) for p in (output / "images.jsonl", output / "crops.jsonl")},
               "notes": ["Selected samples decoded and boxes validated; not a whole-dataset integrity audit.",
                         "Train crops capped per class; total includes train-validation.",
                         "Historical residual annotations excluded because they were used in training.",
                         "Ground-truth crops are separate from later detector-matched crop evaluation."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--max-crops", type=int, default=2000)
    parser.add_argument("--max-train-images", type=int, default=10000)
    parser.add_argument("--max-seconds", type=float, default=1800)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    summary = prepare(Path(config["dataset_root"]), args.output, seed=args.seed,
                      max_images=args.max_images, max_crops=args.max_crops,
                      max_train_images=args.max_train_images, max_seconds=args.max_seconds)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
