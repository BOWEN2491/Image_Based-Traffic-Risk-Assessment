"""CPU detector comparison on immutable, source-grouped local evaluation records.

AP uses 101 recall points and IoU thresholds .50:.05:.95, without COCO area
or crowd handling. It must not be labelled an official COCO benchmark.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import time

CLASSES = ("person", "bicycle", "motorcycle", "car", "bus", "truck", "train", "traffic light")
ALIASES = {"pedestrian": "person", "bike": "bicycle", "motor": "motorcycle"}
LOGGER = logging.getLogger(__name__)


def pipeline_failed(response):
    from .inference_outcome import is_processing_failure

    reason_code = response.get("reason_code")
    if reason_code is None:
        # Historical local reports predate reason_code; retain their two known
        # normal abstention reasons and treat every other legacy result as failure.
        LOGGER.warning("Classifying a historical pipeline response without reason_code")
        return response.get("status") == "uncertain" and response.get("reason") not in {
            "No supported road users or signals were detected",
            "No conservative high-risk rule triggered; statistical models are disabled",
        }
    return is_processing_failure(reason_code)


def category(value):
    value = str(value).lower().replace("_", " ")
    return ALIASES.get(value, value)


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def match(predictions, truth, threshold=0.5):
    """Confidence-ordered, class-aware one-to-one matching; returns GT indices."""
    used, matches = set(), {}
    for index in sorted(range(len(predictions)), key=lambda i: -predictions[i]["confidence"]):
        prediction = predictions[index]
        candidates = [(iou(prediction["bbox"], gt["bbox"]), j) for j, gt in enumerate(truth)
                      if j not in used and category(gt["category"]) == category(prediction["category"])]
        overlap, target = max(candidates, default=(0, -1))
        if target >= 0 and overlap >= threshold:
            matches[index] = target
            used.add(target)
    return matches


def average_precision(scored_hits, total):
    if total == 0:
        return None
    hits = 0
    curve = []
    for rank, (_, hit) in enumerate(sorted(scored_hits, key=lambda pair: -pair[0]), 1):
        hits += hit
        curve.append((hits / total, hits / rank))
    return sum(max((precision for recall, precision in curve if recall >= point / 100), default=0)
               for point in range(101)) / 101


def metrics(records):
    per_class = {}
    for name in CLASSES:
        total = sum(category(gt["category"]) == name for row in records for gt in row["ground_truth"])
        aps = []
        for step in range(10):
            scored = []
            for row in records:
                predictions = [p for p in row["objects"] if category(p["category"]) == name]
                truth = [gt for gt in row["ground_truth"] if category(gt["category"]) == name]
                matches = match(predictions, truth, .5 + step * .05)
                scored.extend((p["confidence"], int(i in matches)) for i, p in enumerate(predictions))
            aps.append(average_precision(scored, total))
        hits, predicted = 0, 0
        for row in records:
            predictions = [p for p in row["objects"] if category(p["category"]) == name and p["confidence"] >= .25]
            truth = [gt for gt in row["ground_truth"] if category(gt["category"]) == name]
            hits += len(match(predictions, truth))
            predicted += len(predictions)
        per_class[name] = {"gt_count": total, "prediction_count_at_025": predicted,
                           "ap50": aps[0], "ap50_95": sum(aps) / 10 if total else None,
                           "recall_at_025_iou50": hits / total if total else None,
                           "precision_at_025_iou50": hits / predicted if predicted else None}
    active = [v for v in per_class.values() if v["gt_count"]]
    return {"images": len(records), "per_class": per_class,
            "map50": sum(v["ap50"] for v in active) / len(active) if active else None,
            "map50_95": sum(v["ap50_95"] for v in active) / len(active) if active else None}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_sample_splits(samples):
    identities, groups, pixels = set(), {}, {}
    for sample in samples:
        if sample["sample_id"] in identities:
            raise ValueError("Duplicate sample identity in evaluation manifest")
        identities.add(sample["sample_id"])
        if sample["split"] not in {"development", "holdout"}:
            raise ValueError("Validation images require development or holdout roles")
        for mapping, field in ((groups, "group_id"), (pixels, "pixel_sha256")):
            key = sample.get(field)
            if not key:
                raise ValueError(f"Missing evaluation identity: {field}")
            if mapping.setdefault(key, sample["split"]) != sample["split"]:
                raise ValueError(f"Cross-split evaluation leakage: {field}")


def evaluate(config):
    import numpy as np
    import psutil
    import torch
    from PIL import Image
    from ultralytics import YOLO
    from .config import Settings
    from .model_runtime import ModelRuntime
    from .pipeline import run_single_image

    torch.set_num_threads(int(config.get("threads", 2)))
    root = Path(config["dataset_root"])
    manifest = Path(config["images_manifest"])
    samples = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = [s for s in samples if s["source_split"] == "val"][:1000]
    if not samples:
        raise ValueError("No validation samples in image manifest")
    validate_sample_splits(samples)
    for sample in samples:
        if sha256(root / sample["image_path"]) != sample["image_sha256"]:
            raise ValueError(f"Image checksum mismatch: {sample['sample_id']}")
        if sha256(root / sample["label_path"]) != sample["label_sha256"]:
            raise ValueError(f"Label checksum mismatch: {sample['sample_id']}")
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "detection-evaluation/1", "manifest_sha256": sha256(manifest),
              "promotion_allowed": False, "promotion_reason": "Local label provenance is not independently verified; retain baseline.",
              "classes": list(CLASSES), "excluded_classes": ["rider", "traffic sign"],
              "settings": {"imgsz": 640, "device": "cpu", "threads": torch.get_num_threads(),
                           "ap_conf": .001, "runtime_conf": .25, "nms_iou": .45, "max_det": 300},
              "metric_definition": "101-point interpolated AP; no area/crowd handling; not official COCO metrics",
              "models": {}}
    for model_name, model_path in config["models"].items():
        if not Path(model_path).is_file():
            raise FileNotFoundError(model_path)
        destination = out / model_name
        destination.mkdir(exist_ok=True)
        model = YOLO(str(model_path))
        runtime = ModelRuntime(Settings(mode="rules"))
        runtime.yolo = model
        model.predict(source=str(root / samples[0]["image_path"]), device="cpu", imgsz=640, verbose=False)
        rows, elapsed, peak_rss, failures = [], [], 0, 0
        with (destination / "predictions.jsonl").open("w", encoding="utf-8") as stream, \
                (destination / "matched_traffic_lights.jsonl").open("w", encoding="utf-8") as crop_stream:
            for sample in samples:
                image_path = root / sample["image_path"]
                row = {key: sample[key] for key in ("sample_id", "image_id", "split", "image_path")}
                row.update(ground_truth=sample["objects"], objects=[], error=None)
                crop_records = []
                try:
                    result = model.predict(source=str(image_path), device="cpu", imgsz=640, conf=.001, iou=.45, max_det=300, verbose=False)[0]
                    row["objects"] = [{"category": category(result.names[int(box.cls.item())]),
                                       "bbox": box.xyxy.cpu().numpy().reshape(-1).tolist(),
                                       "confidence": float(box.conf.item())} for box in result.boxes
                                      if category(result.names[int(box.cls.item())]) in CLASSES]
                    start = time.perf_counter()
                    response = run_single_image(image_path, runtime)
                    row["pipeline_ms"] = (time.perf_counter() - start) * 1000
                    elapsed.append(row["pipeline_ms"])
                    row["risk"] = response["risk"]
                    row["pipeline_status"] = response["status"]
                    row["pipeline_reason"] = response["reason"]
                    if pipeline_failed(response):
                        raise RuntimeError(response["reason"])
                    predictions = [p for p in row["objects"] if p["category"] == "traffic light" and p["confidence"] >= .25]
                    truth = [gt for gt in row["ground_truth"] if category(gt["category"]) == "traffic light"]
                    matched = {gt: pred for pred, gt in match(predictions, truth).items()}
                    with Image.open(image_path) as image:
                        for j, gt in enumerate(truth):
                            record = {"sample_id": sample["sample_id"], "split": sample["split"],
                                      "object_id": gt["object_id"], "color": gt["color"], "crop_path": None, "iou": 0}
                            if j in matched:
                                prediction = predictions[matched[j]]
                                box = prediction["bbox"]
                                x1, y1, x2, y2 = [round(v) for v in box]
                                box = (max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2))
                                if box[2] > box[0] and box[3] > box[1]:
                                    filename = f"{sample['image_id']}-{gt['object_id']}.png"
                                    crop_dir = destination / "crops"
                                    crop_dir.mkdir(exist_ok=True)
                                    image.crop(box).convert("RGB").save(crop_dir / filename)
                                    record.update(crop_path=f"crops/{filename}", iou=iou(prediction["bbox"], gt["bbox"]))
                            crop_records.append(record)
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    failures += 1
                    crop_records = []
                    for gt in row["ground_truth"]:
                        if category(gt["category"]) == "traffic light":
                            crop_records.append({"sample_id": sample["sample_id"], "split": sample["split"],
                                "object_id": gt["object_id"], "color": gt["color"], "crop_path": None, "iou": 0,
                                "error": row["error"], "reason": "inference_failure"})
                for record in crop_records:
                    crop_stream.write(json.dumps(record) + "\n")
                peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                rows.append(row)
                if len(rows) % 50 == 0:
                    print(f"{model_name}: {len(rows)}/{len(samples)}", flush=True)
        model_report = {"sha256": sha256(model_path), "failure_count": failures,
                        "failure_rate": failures / len(samples), "peak_observed_rss_mb": peak_rss / 1024**2,
                        "pipeline_p50_ms": float(np.percentile(elapsed, 50)) if elapsed else None,
                        "pipeline_p95_ms": float(np.percentile(elapsed, 95)) if elapsed else None,
                        "splits": {split: metrics([r for r in rows if r["split"] == split])
                                   for split in ("development", "holdout")}}
        report["models"][model_name] = model_report
        for split in ("development", "holdout"):
            split_times = [r["pipeline_ms"] for r in rows if r["split"] == split and "pipeline_ms" in r]
            model_report["splits"][split]["pipeline_p95_ms"] = float(np.percentile(split_times, 95)) if split_times else None
        with (destination / "per_class.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["split", "category", *next(iter(model_report["splits"]["holdout"]["per_class"].values()))])
            writer.writeheader()
            for split, split_metrics in model_report["splits"].items():
                for name, values in split_metrics["per_class"].items():
                    writer.writerow({"split": split, "category": name, **values})
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    names = list(report["models"])
    if len(names) == 2:
        baseline = report["models"][names[0]]["splits"]["holdout"]
        candidate = report["models"][names[1]]["splits"]["holdout"]
        sufficient = all(baseline["per_class"][name]["gt_count"] >= 20 for name in ("person", "car", "traffic light"))
        report["comparison"] = {
            "baseline": names[0], "candidate": names[1], "sufficient_key_class_samples": sufficient,
            "map50_95_gain": (candidate["map50_95"] - baseline["map50_95"]) if baseline["map50_95"] is not None and candidate["map50_95"] is not None else None,
            "pipeline_p95_ratio": candidate["pipeline_p95_ms"] / baseline["pipeline_p95_ms"] if candidate["pipeline_p95_ms"] and baseline["pipeline_p95_ms"] else None,
            "key_class_recall_change": {name: candidate["per_class"][name]["recall_at_025_iou50"] - baseline["per_class"][name]["recall_at_025_iou50"]
                                        if baseline["per_class"][name]["gt_count"] else None for name in ("person", "car", "traffic light")},
        }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    evaluate(json.loads(args.config.read_text(encoding="utf-8-sig")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
