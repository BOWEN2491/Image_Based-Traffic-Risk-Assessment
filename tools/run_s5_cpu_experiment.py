"""Run a serial, bounded YOLO11n-only S5 exploratory parameter comparison.

The tool selects at most 200 fixed development frames from an existing S4
manifest, verifies their file hashes, and stores all per-image predictions
under a local output directory. No training, holdout access, or downloads occur.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import random
import time

from PIL import Image

from traffic_risk.evaluate_detection import (
    CLASSES,
    average_precision,
    category,
    iou,
    match,
)

SEED = 42
CONFIGS = (
    {"name": "B0", "imgsz": 640, "confidence": .25, "roi": False},
    {"name": "C1", "imgsz": 960, "confidence": .25, "roi": False},
    {"name": "C2", "imgsz": 640, "confidence": .15, "roi": False},
    {"name": "C3", "imgsz": 960, "confidence": .15, "roi": False},
    {"name": "C4-640", "imgsz": 640, "confidence": .25, "roi": True},
    {"name": "C4-960", "imgsz": 960, "confidence": .25, "roi": True},
    {"name": "C5-640", "imgsz": 640, "confidence": .15, "roi": True},
    {"name": "C5-960", "imgsz": 960, "confidence": .15, "roi": True},
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tl_count(row: dict) -> int:
    return sum(category(obj.get("category", "")) == "traffic light" for obj in row.get("objects", []))


def _small_tl_count(row: dict) -> int:
    return sum(category(obj.get("category", "")) == "traffic light"
               and (obj.get("tiny") is True or obj.get("bbox_size") == "small")
               for obj in row.get("objects", []))


def select_development_samples(rows: list[dict], *, max_images: int = 200,
                               small_tl_target: int = 150, nonlight_target: int = 50) -> list[dict]:
    """Choose one deterministic image per ride group, prioritizing small/tiny TLs and no-TL frames."""
    if not 1 <= max_images <= 200 or small_tl_target < 0 or nonlight_target < 0:
        raise ValueError("max_images must be 1..200 and stratum targets nonnegative")
    dev = [row for row in rows if row.get("source_split") == "val" and row.get("split") == "development"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in dev:
        groups[row["group_id"]].append(row)
    def rank(group: str) -> tuple[str, str]:
        return hash_text(f"{SEED}:{group}"), group
    chosen: list[dict] = []
    used_groups: set[str] = set()

    def representative(group: str, *, require_small: bool = False, require_nonlight: bool = False) -> dict:
        candidates = groups[group]
        if require_small:
            candidates = [row for row in candidates if _small_tl_count(row)]
        if require_nonlight:
            candidates = [row for row in candidates if _tl_count(row) == 0]
        return sorted(candidates, key=lambda row: (-_small_tl_count(row), hash_text(f"{SEED}:{row['sample_id']}")))[0]

    max_light = min(small_tl_target, max_images - min(nonlight_target, max_images))
    priority_groups = sorted((g for g, values in groups.items()
                              if any(_small_tl_count(row) for row in values)), key=rank)
    for group in priority_groups[:max_light]:
        chosen.append(representative(group, require_small=True))
        used_groups.add(group)

    # If some groups cannot yield a distinct representative, fill remaining light
    # quota from other TL groups without splitting ride groups.
    light_target = max_light
    if len(chosen) < light_target:
        other_light = sorted((g for g, values in groups.items() if g not in used_groups
                              and any(_tl_count(row) for row in values)), key=rank)
        for group in other_light[:light_target - len(chosen)]:
            chosen.append(representative(group))
            used_groups.add(group)

    nonlight_limit = min(nonlight_target, max_images - len(chosen))
    nonlight_groups = sorted((g for g, values in groups.items() if g not in used_groups
                              and all(_tl_count(row) == 0 for row in values)), key=rank)
    for group in nonlight_groups[:nonlight_limit]:
        chosen.append(representative(group, require_nonlight=True))
        used_groups.add(group)

    # Fill any remaining slots from unused groups using deterministic hash order;
    # this avoids silently dropping frames when strata have fewer groups than requested.
    if len(chosen) < max_images:
        remaining = sorted((g for g in groups if g not in used_groups), key=rank)
        for group in remaining[:max_images - len(chosen)]:
            chosen.append(representative(group))
            used_groups.add(group)
    return sorted(chosen, key=lambda row: hash_text(f"{SEED}:{row['sample_id']}"))[:max_images]


def _raw_detections(result, *, x_offset: float = 0, y_offset: float = 0,
                    width: int, height: int, allowed_names: set[str] | None = None) -> list[dict]:
    output = []
    for box in result.boxes:
        name = category(result.names[int(box.cls.item())])
        if name not in CLASSES or (allowed_names is not None and name not in allowed_names):
            continue
        x1, y1, x2, y2 = box.xyxy.cpu().numpy().reshape(-1).tolist()
        output.append({"category": name,
                       "bbox": map_crop_box((x1, y1, x2, y2), x_offset, y_offset, width, height),
                       "confidence": float(box.conf.item())})
    return output


def map_crop_box(box: tuple[float, float, float, float], origin_x: float, origin_y: float,
                 frame_width: int, frame_height: int) -> list[float]:
    """Translate crop xyxy to original-image coordinates and clip to the frame."""
    x1, y1, x2, y2 = box
    return [max(0.0, min(float(frame_width), x1 + origin_x)),
            max(0.0, min(float(frame_height), y1 + origin_y)),
            max(0.0, min(float(frame_width), x2 + origin_x)),
            max(0.0, min(float(frame_height), y2 + origin_y))]


def _nms_traffic_lights(predictions: list[dict], threshold: float = .45) -> list[dict]:
    kept, light_boxes = [], []
    for prediction in sorted(predictions, key=lambda row: -row["confidence"]):
        if prediction["category"] != "traffic light":
            kept.append(prediction)
            continue
        if all(iou(prediction["bbox"], prior["bbox"]) <= threshold for prior in light_boxes):
            light_boxes.append(prediction)
    return kept + light_boxes


def predict_one(model, image_path: Path, config: dict) -> list[dict]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    full = model.predict(source=image, device="cpu", imgsz=config["imgsz"], conf=.001,
                         iou=.45, max_det=300, verbose=False)[0]
    detections = _raw_detections(full, width=width, height=height)
    if config["roi"]:
        light_class = next((class_id for class_id, name in full.names.items()
                            if category(name) == "traffic light"), None)
        if light_class is None:
            raise ValueError("Public YOLO11n checkpoint lacks a traffic-light category")
        roi_bottom = max(1, round(height * .60))
        roi = image.crop((0, 0, width, roi_bottom))
        cropped = model.predict(source=roi, device="cpu", imgsz=config["imgsz"], conf=.001,
                                iou=.45, max_det=300, classes=[light_class], verbose=False)[0]
        # Crop is top-left anchored, so both offsets are 0; clip to original frame bounds.
        detections.extend(_raw_detections(cropped, x_offset=0, y_offset=0,
                                          width=width, height=roi_bottom,
                                          allowed_names={"traffic light"}))
        detections = _nms_traffic_lights(detections, .45)
    return detections


def metrics_for_threshold(records: list[dict], confidence_threshold: float) -> dict:
    per_class = {}
    for name in CLASSES:
        total = sum(category(gt["category"]) == name for row in records for gt in row["ground_truth"])
        ap_values = []
        for step in range(10):
            scored = []
            for row in records:
                predictions = [p for p in row["objects"] if category(p["category"]) == name]
                truth = [gt for gt in row["ground_truth"] if category(gt["category"]) == name]
                matched = match(predictions, truth, .5 + step * .05)
                scored.extend((p["confidence"], int(index in matched)) for index, p in enumerate(predictions))
            ap_values.append(average_precision(scored, total))
        hits = predicted = 0
        for row in records:
            candidates = [p for p in row["objects"] if category(p["category"]) == name
                          and p["confidence"] >= confidence_threshold]
            truth = [gt for gt in row["ground_truth"] if category(gt["category"]) == name]
            hits += len(match(candidates, truth, .5))
            predicted += len(candidates)
        per_class[name] = {"gt_count": total, "prediction_count": predicted,
                           "map50": ap_values[0],
                           "map50_95": sum(value for value in ap_values if value is not None) / 10 if total else None,
                           "recall": hits / total if total else None,
                           "precision": hits / predicted if predicted else None}
    supported = [value for value in per_class.values() if value["gt_count"]]
    return {"per_class": per_class,
            "map50": sum(row["map50"] for row in supported) / len(supported) if supported else None,
            "map50_95": sum(row["map50_95"] for row in supported) / len(supported) if supported else None}


def run(config_path: Path, model_path: Path, output: Path, *,
        max_images: int = 200, max_per_config_seconds: float = 900,
        max_total_seconds: float = 5400) -> dict:
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    config_data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    dataset_root = Path(config_data["dataset_root"]).resolve(strict=True)
    manifest_path = Path(config_data["images_manifest"]).resolve(strict=True)
    manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = select_development_samples(manifest_rows, max_images=max_images)
    if not selected:
        raise ValueError("No development images available for S5 comparison")
    output.mkdir(parents=True, exist_ok=False)
    for row in selected:
        image_path = (dataset_root / row["image_path"]).resolve(strict=True)
        if not image_path.is_relative_to(dataset_root):
            raise ValueError(f"manifest image escaped dataset root: {row['sample_id']}")
        if sha256(image_path) != row["image_sha256"]:
            raise ValueError(f"image SHA-256 mismatch: {row['sample_id']}")
    with (output / "selection.jsonl").open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps({key: row[key] for key in (
                "sample_id", "group_id", "split", "source_split", "image_sha256", "objects",
                "width", "height", "frame_attributes")}, sort_keys=True) + "\n")
    model_hash = sha256(model_path)
    model = YOLO(str(model_path))  # Local explicit path; no implicit downloads.
    if not hasattr(model, "names") or not isinstance(model.names, dict):
        raise ValueError("Model checkpoint does not expose category metadata")
    if "traffic light" not in {category(name) for name in model.names.values()}:
        raise ValueError("Model checkpoint does not contain a traffic-light class")
    try:
        import psutil
        process = psutil.Process()

        def rss_mb():
            return process.memory_info().rss / (1024 ** 2)
    except ImportError:

        def rss_mb():
            return None
    started = time.monotonic()
    report = {
        "schema": "s5-cpu-exploration/1", "seed": SEED, "model": str(model_path.name),
        "model_sha256": model_hash, "model_version": str(ultralytics.__version__),
        "manifest_sha256": sha256(manifest_path), "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "selected_image_set_sha256": hash_text("\n".join(
            f"{row['sample_id']}:{row['image_sha256']}" for row in selected)),
        "source_status": "local-unverified", "promotion_allowed": False,
        "promotion_blockers": ["dataset provenance and independent holdout are not established"],
        "device": "cpu", "torch_threads": torch.get_num_threads(),
        "selected_images": len(selected), "unique_groups": len({r["group_id"] for r in selected}),
        "small_or_tiny_traffic_light_images": sum(any(category(o["category"]) == "traffic light"
            and (o.get("tiny") or o.get("bbox_size") == "small") for o in r["objects"]) for r in selected),
        "no_traffic_light_images": sum(_tl_count(r) == 0 for r in selected),
        "selection_policy": "one row per development ride group; deterministic SHA-256 order; small/tiny TL groups prioritized, then no-TL groups, then remaining groups",
        "metric_definition": "101-point interpolated AP at IoU .50:.05:.95 from conf .001 predictions; not official COCO; operational TP IoU .5",
        "started_utc": datetime.now(UTC).isoformat(), "max_per_config_seconds": max_per_config_seconds,
        "max_total_seconds": max_total_seconds, "configs": [],
    }
    for config in CONFIGS:
        elapsed_total = time.monotonic() - started
        if elapsed_total >= max_total_seconds:
            report["configs"].append({"name": config["name"], "status": "not_started_total_budget",
                                      "elapsed_seconds": 0, "settings": config})
            continue
        config_budget = min(max_per_config_seconds, max_total_seconds - elapsed_total)
        config_start = time.monotonic()
        rows, peak_rss = [], rss_mb() or 0
        timed_out = False
        failures = 0
        warmup_count = 0
        # Warm once per configuration, then measure all selected frames.
        try:
            predict_one(model, dataset_root / selected[0]["image_path"], config)
            warmup_count = 1
        except Exception:
            # The corresponding measured row below retains the concrete error.
            warmup_count = 0
        for sample in selected:
            if time.monotonic() - config_start >= config_budget:
                timed_out = True
                break
            image_path = dataset_root / sample["image_path"]
            result = {"sample_id": sample["sample_id"], "group_id": sample["group_id"],
                      "split": sample["split"], "ground_truth": sample["objects"],
                      "objects": [], "error": None}
            inference_start = time.perf_counter()
            try:
                result["objects"] = predict_one(model, image_path, config)
            except Exception as exc:
                failures += 1
                result["error"] = f"{type(exc).__name__}: {exc}"
            result["prediction_ms"] = (time.perf_counter() - inference_start) * 1000
            current_rss = rss_mb()
            if current_rss is not None:
                peak_rss = max(peak_rss, current_rss)
            rows.append(result)
            if len(rows) % 20 == 0:
                print(f"{config['name']}: {len(rows)}/{len(selected)}", flush=True)
        elapsed = time.monotonic() - config_start
        prediction_path = output / f"{config['name']}.jsonl"
        prediction_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        result = {"name": config["name"], "status": "timed_out_partial" if timed_out else "complete",
                  "settings": {**config, "device": "cpu", "threads": 2, "seed": SEED,
                               "ap_confidence": .001, "nms_iou": .45, "max_det_per_pass": 300,
                               "roi_region": "top 60%; traffic-light-only" if config["roi"] else None,
                               "roi_merge": "same-class traffic-light NMS IoU .45" if config["roi"] else None},
                  "completed_images": len(rows), "failed_images": failures,
                  "failure_rate": failures / len(rows) if rows else None,
                  "elapsed_seconds": elapsed, "images_per_second": len(rows) / elapsed if elapsed else None,
                  "warmup_images": warmup_count,
                  "prediction_p50_ms": float(np.percentile([row["prediction_ms"] for row in rows], 50)) if rows else None,
                  "prediction_p95_ms": float(np.percentile([row["prediction_ms"] for row in rows], 95)) if rows else None,
                  "peak_observed_rss_mb": peak_rss,
                  "prediction_file": prediction_path.name,
                  "metrics": metrics_for_threshold(rows, config["confidence"]),
                  "timeout_limit_seconds": config_budget}
        report["configs"].append(result)
        (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["ended_utc"] = datetime.now(UTC).isoformat()
    report["elapsed_total_seconds"] = time.monotonic() - started
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--per-config-seconds", type=float, default=900)
    parser.add_argument("--total-seconds", type=float, default=5400)
    args = parser.parse_args(argv)
    if not 1 <= args.max_images <= 200 or args.per_config_seconds <= 0 or args.total_seconds <= 0:
        parser.error("image cap must be 1..200 and time budgets must be positive")
    result = run(args.config, args.model, args.output, max_images=args.max_images,
                 max_per_config_seconds=args.per_config_seconds, max_total_seconds=args.total_seconds)
    print(json.dumps({"output": str(args.output), "completed_configs": sum(
        item.get("status") == "complete" for item in result["configs"]),
        "elapsed_total_seconds": result["elapsed_total_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
