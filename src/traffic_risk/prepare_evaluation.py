"""Build bounded, deterministic, source-grouped exploratory evaluation manifests.

Source files are read only. Local annotations have not been authenticated against
an upstream archive, so these manifests explicitly prohibit model promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.parse import urlparse

from PIL import Image

from .sample_identity import sha256_file

COLORS = ("green", "red", "unknown", "yellow")
PROVENANCE = "local-unverified"
SOURCE_STATUSES = {"local-unverified", "source-verified"}
ANNOTATION_REVIEW_STATUSES = {"awaiting-human-review", "not-reviewed", "reviewed"}
UNKNOWN = "unknown"


def _attribute(value):
    if value is None or value == "" or value == "none":
        return UNKNOWN
    return value


def _area_bucket(area_ratio: float) -> str:
    if area_ratio < 0.001:
        return "small"
    if area_ratio < 0.01:
        return "medium"
    return "large"


def default_provenance_record() -> dict:
    """Return a deliberately unverified source/annotation record."""
    return {
        "schema_version": 1,
        "source_status": PROVENANCE,
        "provider": None,
        "archive_id": None,
        "source_url": None,
        "acquired_date": None,
        "acquisition_and_conversion": [],
        "source_archive_sha256": None,
        "verified_by": None,
        "verified_date": None,
        "annotation_review": {
            "status": "awaiting-human-review",
            "reviewer": None,
            "reviewed_date": None,
            "scope": None,
        },
    }


def validate_provenance_record(value: dict | None) -> dict:
    """Validate supplied provenance metadata without asserting it is true.

    A `source-verified` record is a human attestation, not automated provenance
    discovery. This preparer always keeps promotion disabled pending an
    independently selected and reviewed holdout.
    """
    record = default_provenance_record() if value is None else value
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("provenance record must be a schema_version 1 object")
    if record.get("source_status") not in SOURCE_STATUSES:
        raise ValueError("source_status must be local-unverified or source-verified")
    required = (
        "provider", "archive_id", "source_url", "acquired_date",
        "acquisition_and_conversion", "source_archive_sha256", "verified_by", "verified_date",
    )
    if record["source_status"] == "source-verified":
        if any(not record.get(key) for key in required):
            raise ValueError("source-verified requires provider, archive, acquisition, checksum, and verifier fields")
    else:
        record = {**default_provenance_record(), **record}

    for field in ("provider", "archive_id", "verified_by"):
        if record.get(field) is not None and (not isinstance(record[field], str) or not record[field].strip()):
            raise ValueError(f"{field} must be non-empty text when provided")
    if record.get("source_url") is not None:
        if not isinstance(record["source_url"], str):
            raise ValueError("source_url must be text when provided")
        parsed = urlparse(record["source_url"])
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValueError("source_url must be an http(s) URL")
    for field in ("acquired_date", "verified_date"):
        if record.get(field) is not None:
            try:
                date.fromisoformat(record[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must use ISO YYYY-MM-DD format") from exc
    if record.get("acquisition_and_conversion") is not None:
        steps = record["acquisition_and_conversion"]
        if not isinstance(steps, list) or any(not isinstance(step, str) or not step.strip() for step in steps):
            raise ValueError("acquisition_and_conversion must be a list of non-empty documented steps")
    if record.get("source_archive_sha256") is not None:
        digest = record["source_archive_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError("source_archive_sha256 must be a 64-character SHA-256 digest")

    review = record.get("annotation_review")
    if not isinstance(review, dict) or review.get("status") not in ANNOTATION_REVIEW_STATUSES:
        raise ValueError("annotation_review must have a supported status")
    if review["status"] == "reviewed":
        if not review.get("reviewer") or not review.get("reviewed_date") or not review.get("scope"):
            raise ValueError("reviewed annotation status requires reviewer, reviewed_date, and scope")
        try:
            date.fromisoformat(review["reviewed_date"])
        except (TypeError, ValueError) as exc:
            raise ValueError("annotation_review.reviewed_date must use ISO YYYY-MM-DD format") from exc
    return record


def audit_cross_manifest(current: list[dict], prior_rows: list[tuple[str, list[dict]]]) -> dict:
    """Report cross-run role conflicts by image identity, ride group, and pixels."""
    fields = ("sample_id", "group_id", "pixel_sha256")
    current_index: dict[str, dict[str, set[str]]] = {field: {} for field in fields}
    prior_index: dict[str, dict[str, set[str]]] = {field: {} for field in fields}
    for rows, index in ((current, current_index),):
        for row in rows:
            for field in fields:
                value = row.get(field)
                if value:
                    index[field].setdefault(str(value), set()).add(row["split"])
    for _name, rows in prior_rows:
        for row in rows:
            for field in fields:
                value = row.get(field)
                if value:
                    prior_index[field].setdefault(str(value), set()).add(row["split"])

    conflicts, overlaps = [], []
    for field in fields:
        for value in sorted(current_index[field].keys() & prior_index[field].keys()):
            prior_roles = prior_index[field][value]
            current_roles = current_index[field][value]
            details = {"field": field, "identity_sha256": hashlib.sha256(value.encode()).hexdigest(),
                       "prior_roles": sorted(prior_roles), "current_roles": sorted(current_roles)}
            if prior_roles & current_roles:
                overlaps.append(details)
            if any(prior_role != current_role for prior_role in prior_roles for current_role in current_roles):
                conflicts.append(details)
    return {
        "schema_version": 1,
        "prior_manifest_names": sorted(name for name, _rows in prior_rows),
        "same_role_overlap_count": len(overlaps),
        "cross_role_conflict_count": len(conflicts),
        "same_role_overlaps": overlaps,
        "cross_role_conflicts": conflicts,
        "promotion_allowed": False,
    }


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
    frame_attributes = payload.get("attributes") or {}
    if not isinstance(frame_attributes, dict):
        frame_attributes = {}
    context = {key: _attribute(frame_attributes.get(key)) for key in ("weather", "scene", "timeofday")}
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
        object_attributes = obj.get("attributes") or {}
        if not isinstance(object_attributes, dict):
            object_attributes = {}
        width_px, height_px = x2 - x1, y2 - y1
        area_ratio = width_px * height_px / (width * height)
        objects.append({"object_id": str(obj.get("id", index)), "category": category,
                        "bbox": coords, "color": color, "bbox_area_ratio": area_ratio,
                        "bbox_size": _area_bucket(area_ratio),
                        "bbox_shortest_edge_px": min(width_px, height_px),
                        "tiny": min(width_px, height_px) < 6,
                        "attributes": {"occluded": _attribute(object_attributes.get("occluded")),
                                       "truncated": _attribute(object_attributes.get("truncated"))}})
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
              "frame_attributes": context, "provenance": PROVENANCE,
              "annotation_review": "awaiting-human-review", "promotion_allowed": False, "objects": objects}
    return record, image


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_prior_manifests(manifests: list[Path]) -> list[tuple[str, list[dict]]]:
    output = []
    for path in manifests:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        output.append((path.name, rows))
    return output


def _evaluation_support(images: list[dict]) -> dict:
    support = {}
    target = 100
    for split in ("development", "holdout"):
        rows = [row for row in images if row["split"] == split and row["source_split"] == "val"]
        color_counts = Counter(obj["color"] for row in rows for obj in row["objects"]
                               if obj["category"] == "traffic light")
        size_counts = Counter(obj["bbox_size"] for row in rows for obj in row["objects"])
        context_counts = {
            key: dict(Counter(row["frame_attributes"].get(key, UNKNOWN) for row in rows))
            for key in ("weather", "scene", "timeofday")
        }
        object_attribute_counts = {
            key: dict(Counter(obj["attributes"].get(key, UNKNOWN)
                              for row in rows for obj in row["objects"]))
            for key in ("occluded", "truncated")
        }
        support[split] = {
            "traffic_light_color_counts": {color: color_counts[color] for color in COLORS},
            "traffic_light_color_target": target,
            "traffic_light_color_shortfalls": {color: max(0, target - color_counts[color]) for color in COLORS},
            "ground_truth_bbox_area_counts": {name: size_counts[name] for name in ("small", "medium", "large")},
            "ground_truth_tiny_object_count": sum(obj["tiny"] for row in rows for obj in row["objects"]),
            "traffic_light_tiny_count": sum(obj["tiny"] for row in rows for obj in row["objects"]
                                              if obj["category"] == "traffic light"),
            "context_counts": context_counts,
            "ground_truth_object_attribute_counts": object_attribute_counts,
        }
    return support


def prepare(root: Path, output: Path, *, seed: int = 42,
            max_images: int = 1000, max_crops: int = 2000,
            max_train_images: int = 10000, max_seconds: float = 1800,
            provenance_record: dict | None = None,
            compare_manifests: list[Path] | None = None) -> dict:
    root, output = root.resolve(), output.resolve()
    if not 1 <= max_images <= 1000 or not 4 <= max_crops <= 2000:
        raise ValueError("max_images must be 1..1000 and max_crops 4..2000")
    if not 1 <= max_train_images <= 10000 or not 0 < max_seconds <= 1800:
        raise ValueError("max_train_images must be 1..10000 and max_seconds in (0, 1800]")
    if output == root or root in output.parents:
        raise ValueError("output must be outside the read-only dataset")
    if output.exists():
        raise FileExistsError("Use a new output directory to preserve fixed experiment manifests")
    provenance = validate_provenance_record(provenance_record)
    prior_rows = _read_prior_manifests(compare_manifests or [])
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
                          "provenance": provenance["source_status"],
                          "annotation_review": provenance["annotation_review"]["status"],
                          "promotion_allowed": False})
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
            record["provenance"] = provenance["source_status"]
            record["annotation_review"] = provenance["annotation_review"]["status"]
            selected_counts[source_split] += 1
            crop_objects(record, image, source_split == "train")
    _write_jsonl(output / "images.jsonl", images)
    _write_jsonl(output / "crops.jsonl", crops)
    _write_jsonl(output / "excluded.jsonl", excluded)
    provenance_path = output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    provenance_hash = sha256_file(provenance_path)
    for row in images:
        row["provenance_record_sha256"] = provenance_hash
    # Re-write after adding the source-record identity to each row.
    _write_jsonl(output / "images.jsonl", images)
    support = _evaluation_support(images)
    cross_manifest = audit_cross_manifest(images, prior_rows) if prior_rows else None
    if cross_manifest is not None:
        (output / "cross_manifest_audit.json").write_text(json.dumps(cross_manifest, indent=2), encoding="utf-8")
    blockers = ["a new independent source/holdout has not been established"]
    if provenance["source_status"] != "source-verified":
        blockers.append("source provenance is local-unverified")
    if provenance["annotation_review"]["status"] != "reviewed":
        blockers.append("independent annotation review is incomplete")
    if cross_manifest and cross_manifest["cross_role_conflict_count"]:
        blockers.append("cross-manifest role conflicts exist")
    summary = {"schema_version": 2, "seed": seed, "provenance": provenance["source_status"],
               "annotation_review": provenance["annotation_review"]["status"],
               "provenance_manifest": {"file": provenance_path.name, "sha256": provenance_hash},
               "promotion_allowed": False, "promotion_blockers": blockers,
               "max_images": max_images, "max_training_crops": max_crops,
               "max_train_images": max_train_images, "max_seconds": max_seconds,
               "scanned_counts": dict(scanned_counts), "stop_reason": stop_reason,
               "training_class_shortfalls": {c: max_crops // 4 - train_counts[c] for c in COLORS},
               "image_counts": dict(Counter(r["split"] for r in images)),
               "crop_counts": dict(Counter(f"{r['split']}/{r['label']}" for r in crops)),
               "training_class_counts": dict(train_counts), "exclusions": len(excluded),
               "fallback_groups": sum(r["grouping"] != "rideID" for r in images),
               "evaluation_support": support,
               "cross_manifest_audit": cross_manifest,
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
    parser.add_argument("--provenance", type=Path, help="Human-maintained source and annotation provenance JSON")
    parser.add_argument("--compare-manifest", type=Path, action="append", default=[],
                        help="Prior images.jsonl manifest to audit for cross-role leakage (repeatable)")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    source_record = json.loads(args.provenance.read_text(encoding="utf-8-sig")) if args.provenance else None
    summary = prepare(Path(config["dataset_root"]), args.output, seed=args.seed,
                      max_images=args.max_images, max_crops=args.max_crops,
                      max_train_images=args.max_train_images, max_seconds=args.max_seconds,
                      provenance_record=source_record, compare_manifests=args.compare_manifest)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
