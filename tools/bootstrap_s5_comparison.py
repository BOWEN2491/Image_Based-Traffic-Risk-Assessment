"""Paired ride-group bootstrap intervals for saved S5 development predictions.

Reads only JSONL prediction artifacts and performs no detector inference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

try:
    from .run_s5_cpu_experiment import CONFIGS
except ImportError:  # Direct `python tools/bootstrap_s5_comparison.py` invocation.
    from run_s5_cpu_experiment import CONFIGS
from traffic_risk.evaluate_detection import category, match


def bootstrap(input_dir: Path, *, resamples: int = 200, seed: int = 42,
              max_seconds: float = 60) -> dict:
    if resamples < 1:
        raise ValueError("resamples must be positive")
    summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    statuses = {row["name"]: row["status"] for row in summary["configs"]}
    configs = [row for row in CONFIGS if statuses.get(row["name"]) == "complete"]
    baseline_config = next((row for row in configs if row["name"] == "B0"), None)
    if baseline_config is None:
        raise ValueError("B0 must be complete for paired comparisons")
    predictions = {}
    for config in configs:
        path = input_dir / f"{config['name']}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows or any(row.get("split") != "development" for row in rows):
            raise ValueError(f"{config['name']} predictions must contain development records only")
        predictions[config["name"]] = rows
    base_groups: dict[str, list[dict]] = defaultdict(list)
    for row in predictions["B0"]:
        base_groups[row["group_id"]].append(row)
    group_ids = sorted(base_groups)
    for name, rows in predictions.items():
        if {row["group_id"] for row in rows} != set(group_ids):
            raise ValueError(f"{name} group set differs from B0")
        if len(rows) != len(group_ids):
            raise ValueError(f"{name} must contain one image per ride group")
    thresholds = {config["name"]: config["confidence"] for config in configs}
    grouped_recall = {}
    point_values = {}
    for name, rows in predictions.items():
        tl_hits, tl_total = {}, {}
        for row in rows:
            truths = [gt for gt in row["ground_truth"] if category(gt["category"]) == "traffic light"]
            candidates = [p for p in row["objects"] if category(p["category"]) == "traffic light"
                          and p["confidence"] >= thresholds[name]]
            tl_hits[row["group_id"]] = len(match(candidates, truths, .5))
            tl_total[row["group_id"]] = len(truths)
        grouped_recall[name] = (tl_hits, tl_total)
        report_config = next(config for config in summary["configs"] if config["name"] == name)
        point_values[name] = {
            "map50_95": report_config.get("metrics", {}).get("map50_95"),
            "traffic_light_recall": (sum(tl_hits.values()) / sum(tl_total.values())
                                      if sum(tl_total.values()) else None),
        }
    rng = random.Random(seed)
    deltas = {config["name"]: {"map50_95": [], "traffic_light_recall": []}
              for config in configs if config["name"] != "B0"}
    started = time.monotonic()
    completed_resamples = 0
    for iteration in range(resamples):
        if time.monotonic() - started >= max_seconds:
            break
        picked = [rng.choice(group_ids) for _ in group_ids]
        base_hits, base_total = grouped_recall["B0"]
        base_recall = sum(base_hits[group] for group in picked) / max(1, sum(base_total[group] for group in picked))
        for name in deltas:
            hits, total = grouped_recall[name]
            recall = sum(hits[group] for group in picked) / max(1, sum(total[group] for group in picked))
            deltas[name]["traffic_light_recall"].append(recall - base_recall)
        completed_resamples += 1
        if (iteration + 1) % 25 == 0:
            print(f"bootstrap: {iteration + 1}/{resamples}", flush=True)
    output = {}
    for name, metric_deltas in deltas.items():
        output[name] = {}
        for metric_name, values in metric_deltas.items():
            base_value = point_values["B0"][metric_name]
            candidate_value = point_values[name][metric_name]
            interval = ([float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
                        if metric_name == "traffic_light_recall" and values else None)
            output[name][metric_name] = {
                "baseline": base_value, "candidate": candidate_value,
                "delta": candidate_value - base_value if base_value is not None and candidate_value is not None else None,
                "paired_group_bootstrap_95_ci": interval,
                "valid_resamples": len(values) if metric_name == "traffic_light_recall" else 0,
                "interval_status": "estimated" if interval else "not_computed_within_bounded_method",
            }
    return {"schema": "s5-ride-group-bootstrap/1", "seed": seed,
            "requested_resamples": resamples, "completed_resamples": completed_resamples,
            "runtime_seconds": time.monotonic() - started, "runtime_cap_seconds": max_seconds,
            "analysis_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "unit": "ride group; one manifest image per group", "development_only": True,
            "source_status": summary.get("source_status"), "promotion_allowed": False,
            "promotion_blockers": summary.get("promotion_blockers", []),
            "limitations": ["Only traffic-light recall receives a bootstrap interval; mAP50-95 point deltas are reported without interval because per-resample AP recomputation exceeded the bounded runtime budget."],
            "included_configs": [row["name"] for row in configs], "comparisons": output}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    if args.resamples < 1 or args.max_seconds <= 0:
        parser.error("resamples and max-seconds must be positive")
    if args.output.exists():
        parser.error("output file already exists")
    result = bootstrap(args.input, resamples=args.resamples, seed=args.seed, max_seconds=args.max_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "comparisons": len(result["comparisons"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
