"""Summarize bounded detector failures from existing local evaluation artifacts.

This tool reads only the supplied predictions JSONL and fixed image manifest;
it does not run inference or traverse the dataset. Keep its output local.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image, ImageDraw

from traffic_risk.evaluate_detection import category, iou


def _bucket(gt: dict, image: dict) -> str:
    box = gt["bbox"]
    ratio = max(0, box[2] - box[0]) * max(0, box[3] - box[1]) / (image["width"] * image["height"])
    return "small" if ratio < .001 else "medium" if ratio < .01 else "large"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def analyze(prediction_rows: list[dict], manifest_rows: list[dict], *, iou_threshold: float = .5,
            confidence_threshold: float = .25, max_cases: int = 100) -> tuple[dict, list[dict]]:
    manifest = {row["sample_id"]: row for row in manifest_rows}
    by_role: dict[str, dict] = {}
    cases: list[dict] = []
    for role in ("development", "holdout"):
        gt_support: Counter = Counter()
        gt_detected: Counter = Counter()
        gt_failure: Counter = Counter()
        pred_counts: Counter = Counter()
        pred_true_positive: Counter = Counter()
        pred_failure: Counter = Counter()
        size_prediction_counts: Counter = Counter()
        size_prediction_true_positive: Counter = Counter()
        records = [r for r in prediction_rows if r.get("split") == role]
        for row in records:
            source = manifest.get(row.get("sample_id"))
            if source is None:
                raise ValueError(f"sample missing from manifest: {row.get('sample_id')}")
            truths = row.get("ground_truth", [])
            preds = [p for p in row.get("objects", []) if p.get("confidence", 0) >= confidence_threshold]
            object_meta = {str(o.get("object_id")): o for o in source.get("objects", [])}
            buckets = [_bucket(gt, source) for gt in truths]
            categories = [category(gt["category"]) for gt in truths]
            attrs = [object_meta.get(str(gt.get("object_id")), {}).get("attributes", {}) for gt in truths]
            matched_gt: set[int] = set()
            matched_pred: set[int] = set()
            matched_pair: dict[int, int] = {}
            if row.get("error"):
                for i, gt in enumerate(truths):
                    key = (categories[i], buckets[i], str(attrs[i].get("occluded", "unknown")),
                           str(attrs[i].get("truncated", "unknown")))
                    gt_support[key] += 1
                    gt_failure[key + ("processing_failure",)] += 1
                    if len(cases) < max_cases:
                        cases.append({"sample_id": row["sample_id"], "split": role,
                                      "object_id": gt.get("object_id"), "category": categories[i],
                                      "bbox_size": buckets[i], "failure": "processing_failure",
                                      "bbox": gt["bbox"], "details": row["error"]})
                continue
            # Confidence-ordered matching among same-category boxes.
            for pi in sorted(range(len(preds)), key=lambda n: -preds[n].get("confidence", 0)):
                choices = [(iou(preds[pi]["bbox"], gt["bbox"]), gi) for gi, gt in enumerate(truths)
                           if gi not in matched_gt and categories[gi] == category(preds[pi]["category"])]
                overlap, gi = max(choices, default=(0.0, -1))
                if gi >= 0 and overlap >= iou_threshold:
                    matched_gt.add(gi)
                    matched_pred.add(pi)
                    matched_pair[pi] = gi
            for pi, prediction in enumerate(preds):
                name = category(prediction["category"])
                pred_counts[name] += 1
                if pi in matched_pred:
                    gi = matched_pair[pi]
                    size_prediction_counts[(categories[gi], buckets[gi])] += 1
                    size_prediction_true_positive[(categories[gi], buckets[gi])] += 1
                else:
                    overlap, gi = max(((iou(prediction["bbox"], gt["bbox"]), index)
                                       for index, gt in enumerate(truths)), default=(0.0, -1))
                    if gi >= 0 and overlap > 0:
                        size_prediction_counts[(categories[gi], buckets[gi])] += 1
                if pi in matched_pred:
                    pred_true_positive[name] += 1
                else:
                    others = [(iou(prediction["bbox"], gt["bbox"]), gi) for gi, gt in enumerate(truths)]
                    overlap, gi = max(others, default=(0.0, -1))
                    reason = "wrong_class" if overlap >= iou_threshold and gi >= 0 and categories[gi] != name else "false_positive"
                    pred_failure[name] += 1
                    if len(cases) < max_cases:
                        cases.append({"sample_id": row["sample_id"], "split": role,
                                      "category": name, "failure": reason,
                                      "confidence": prediction.get("confidence"), "best_iou": overlap,
                                      "bbox": prediction["bbox"]})
            for gi, gt in enumerate(truths):
                key = (categories[gi], buckets[gi], str(attrs[gi].get("occluded", "unknown")),
                       str(attrs[gi].get("truncated", "unknown")))
                gt_support[key] += 1
                if gi in matched_gt:
                    gt_detected[key] += 1
                else:
                    overlap = max((iou(p["bbox"], gt["bbox"]) for p in preds), default=0.0)
                    wrong_class_overlap = max((iou(p["bbox"], gt["bbox"]) for p in preds
                                               if category(p["category"]) != categories[gi]), default=0.0)
                    reason = ("wrong_class" if wrong_class_overlap >= iou_threshold else
                              "low_iou" if overlap > 0 else "missed")
                    gt_failure[key + (reason,)] += 1
                    if len(cases) < max_cases:
                        cases.append({"sample_id": row["sample_id"], "split": role,
                                      "object_id": gt.get("object_id"), "category": categories[gi],
                                      "bbox_size": buckets[gi], "failure": reason, "best_iou": overlap,
                                      "bbox": gt["bbox"]})
        groups = {}
        for key, support in sorted(gt_support.items()):
            name, size, occluded, truncated = key
            group = groups.setdefault(name, {}).setdefault(size, {"gt_support": 0, "detected": 0,
                "missed": 0, "low_iou": 0, "wrong_class": 0, "processing_failure": 0,
                "occluded": {}, "truncated": {}})
            group["gt_support"] += support
            group["detected"] += gt_detected[key]
            group["missed"] += gt_failure[key + ("missed",)]
            group["low_iou"] += gt_failure[key + ("low_iou",)]
            group["wrong_class"] += gt_failure[key + ("wrong_class",)]
            group["processing_failure"] += gt_failure[key + ("processing_failure",)]
            group["occluded"][occluded] = group["occluded"].get(occluded, 0) + support
            group["truncated"][truncated] = group["truncated"].get(truncated, 0) + support
            group["recall"] = group["detected"] / group["gt_support"] if group["gt_support"] else None
        for name, sizes in groups.items():
            for size, group in sizes.items():
                total_predictions = size_prediction_counts[(name, size)]
                group["prediction_count_assigned_to_gt_size"] = total_predictions
                group["precision"] = (size_prediction_true_positive[(name, size)] / total_predictions
                                      if total_predictions else None)
        by_role[role] = {"images": len(records), "ground_truth_by_class_and_size": groups,
            "prediction_precision_by_class": {name: pred_true_positive[name] / count if count else None
                for name, count in sorted(pred_counts.items())},
            "predictions_by_class": dict(pred_counts), "unmatched_predictions_by_reason": dict(pred_failure)}
    return {"schema": "detection-error-analysis/1", "iou_threshold": iou_threshold,
            "confidence_threshold": confidence_threshold, "max_cases": max_cases,
            "local_only": True, "roles": by_role}, cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--iou", type=float, default=.5)
    parser.add_argument("--max-cases", type=int, default=100)
    parser.add_argument("--image-root", type=Path,
                        help="Optional dataset image root; with --atlas, only manifest-listed images under this root are read")
    parser.add_argument("--atlas", action="store_true", help="Write local annotated PNGs for selected cases (requires --image-root)")
    args = parser.parse_args(argv)
    if not 0 <= args.confidence <= 1 or not 0 < args.iou <= 1 or not 0 <= args.max_cases <= 100:
        parser.error("confidence must be [0,1], IoU (0,1], and max-cases 0..100")
    if args.atlas and args.image_root is None:
        parser.error("--atlas requires --image-root")
    manifest_rows = _read_jsonl(args.manifest)
    summary, cases = analyze(_read_jsonl(args.predictions), manifest_rows,
                             confidence_threshold=args.confidence, iou_threshold=args.iou,
                             max_cases=args.max_cases)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cases), encoding="utf-8")
    if args.atlas:
        atlas_dir = args.output / "atlas"
        atlas_dir.mkdir()
        root = args.image_root.resolve(strict=True)
        images = {row["sample_id"]: row.get("image_path") for row in manifest_rows}
        index = []
        for number, case in enumerate(cases, 1):
            relative = images.get(case["sample_id"])
            if not relative:
                continue
            source = (root / relative).resolve(strict=True)
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError(f"manifest image is outside --image-root: {relative}")
            with Image.open(source) as image:
                image = image.convert("RGB")
                draw = ImageDraw.Draw(image)
                box = tuple(float(v) for v in case["bbox"])
                draw.rectangle(box, outline="#ff2d2d", width=max(2, round(min(image.size) / 400)))
                title = f"{case['failure']} | {case.get('category', 'unknown')}"
                draw.rectangle((0, 0, min(image.width, 720), 24), fill="black")
                draw.text((6, 5), title, fill="white")
                filename = f"{number:03d}-{case['failure']}.png"
                image.save(atlas_dir / filename)
            index.append({"file": f"atlas/{filename}", "sample_id": case["sample_id"],
                          "split": case["split"], "failure": case["failure"],
                          "category": case.get("category"), "bbox": case["bbox"]})
        (args.output / "atlas_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(args.output / "summary.json"), "cases": len(cases)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
