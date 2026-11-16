"""Repeat pipeline timing in reverse candidate order, without detector AP work."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from ultralytics import YOLO

from traffic_risk.config import Settings
from traffic_risk.evaluate_detection import pipeline_failed, sha256, validate_sample_splits
from traffic_risk.model_runtime import ModelRuntime
from traffic_risk.pipeline import run_single_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    torch.set_num_threads(2)
    manifest = Path(config["images_manifest"])
    samples = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    samples = [s for s in samples if s["source_split"] == "val" and s["split"] == "holdout"][:50]
    validate_sample_splits(samples)
    root = Path(config["dataset_root"])
    for sample in samples:
        if sha256(root / sample["image_path"]) != sample["image_sha256"]:
            raise ValueError("Changed input image")
    models = list(reversed(list(config["models"].items())))
    report = {"images": len(samples), "repeats": 3, "warmups_per_round": 5, "threads": 2,
              "order": [name for name, _ in models], "manifest_sha256": sha256(manifest), "models": {}}
    for name, path in models:
        runtime = ModelRuntime(Settings(mode="rules", device="cpu"))
        runtime.yolo = YOLO(path)
        rounds = []
        for repeat in range(3):
            for sample in samples[:5]:
                run_single_image(root / sample["image_path"], runtime)
            times, failures = [], 0
            for sample in samples:
                started = time.perf_counter()
                response = run_single_image(root / sample["image_path"], runtime)
                times.append((time.perf_counter() - started) * 1000)
                failures += int(pipeline_failed(response))
            rounds.append({"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
                           "failures": failures, "times_ms": times})
            print(name, repeat + 1, rounds[-1]["p95_ms"], flush=True)
        report["models"][name] = {"rounds": rounds, "median_p95_ms": float(np.median([r["p95_ms"] for r in rounds]))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
