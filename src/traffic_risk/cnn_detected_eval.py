"""Score saved CNNs on detector crops, including every missed ground-truth light."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .cnn_contract import make_model, validate_contract
from .cnn_experiment import evaluate


def run(matches: Path, checkpoints: Path, output: Path):
    torch.set_num_threads(2)
    rows = [json.loads(line) for line in matches.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    seen = set()
    for row in rows:
        if row["split"] != "holdout":
            continue
        key = (row["sample_id"], row["object_id"])
        if key in seen:
            raise ValueError("Duplicate ground-truth light in detector matches")
        seen.add(key)
        relative = row["crop_path"]
        if relative is not None and matches.parent.resolve() not in (matches.parent / relative).resolve().parents:
            raise ValueError("Detector crop path escapes the results directory")
        records.append({"path": relative, "label": row["color"], "detector_missed": relative is None,
                        "runtime_size_gate": True})
    if not records:
        raise ValueError("No holdout traffic lights in detector matches")
    report = {"promotion_allowed": False, "provenance": "local-unverified",
              "matches_sha256": hashlib.sha256(matches.read_bytes()).hexdigest(),
              "gt_lights": len(records), "detector_misses": sum(r["detector_missed"] for r in records),
              "models": {}}
    for architecture in ("resnet18", "mobilenetv3"):
        directory = checkpoints / architecture
        metadata = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
        cnn_contract = validate_contract(metadata["cnn_contract"])
        if cnn_contract["architecture"] != architecture:
            raise ValueError("Checkpoint architecture mismatch")
        model = make_model(architecture)
        model.load_state_dict(torch.load(directory / "best_model.pth", map_location="cpu", weights_only=True), strict=True)
        report["models"][architecture] = evaluate(model, records, matches.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.matches, args.checkpoints, args.output)


if __name__ == "__main__":
    main()
