"""Bounded, source-separated CPU comparison of traffic-light classifiers."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .cnn_contract import CLASS_MAPPING, contract, inference_transform, make_model


class Crops(Dataset):
    def __init__(self, records, root):
        self.records = records
        self.root = root
        self.transform = inference_transform()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, CLASS_MAPPING[record["label"]]


def summarize(targets, predictions, requested_support=None):
    confusion = np.zeros((4, 4), dtype=int)
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[target, prediction] += 1
    tp = np.diag(confusion)
    support = confusion.sum(axis=1) if requested_support is None else np.asarray(requested_support)
    recall = np.divide(tp, support, out=np.zeros(4), where=support > 0)
    denom = support + confusion.sum(axis=0)
    f1 = np.divide(2 * tp, denom, out=np.zeros(4), where=denom > 0)
    return {"macro_f1": float(f1.mean()), "confusion_matrix": confusion.tolist(),
            "recall": dict(zip(CLASS_MAPPING, recall.tolist(), strict=True)),
            "support": dict(zip(CLASS_MAPPING, support.tolist(), strict=True)),
            "sufficient_support": bool((support >= 20).all())}


def evaluate(model, records, root):
    model.eval()
    targets, predictions, policy_predictions, latency, rows = [], [], [], [], []
    requested_support = [sum(r["label"] == label for r in records) for label in CLASS_MAPPING]
    failed = 0
    dataset = Crops(records, root)
    with torch.inference_mode():
        model(torch.zeros(1, 3, 128, 128))
        for index, record in enumerate(records):
            started = time.perf_counter()
            try:
                if record.get("detector_missed"):
                    raise ValueError("Detector missed this labelled traffic light")
                tensor, target = dataset[index]
                tiny = False
                if record.get("runtime_size_gate"):
                    with Image.open(root / record["path"]) as original:
                        tiny = min(original.size) < 6
                if tiny:
                    prediction, policy, confidence = CLASS_MAPPING["unknown"], CLASS_MAPPING["unknown"], 0.0
                else:
                    probabilities = torch.softmax(model(tensor.unsqueeze(0)), dim=1)[0]
                    confidence, index = probabilities.max(0)
                    prediction = int(index)
                    policy = prediction if float(confidence) >= 0.5 else CLASS_MAPPING["unknown"]
                latency.append((time.perf_counter() - started) * 1000)
                targets.append(target)
                predictions.append(prediction)
                policy_predictions.append(policy)
                rows.append({"path": record["path"], "target": target, "prediction": prediction,
                             "runtime_prediction": policy, "confidence": float(confidence)})
            except (OSError, ValueError, RuntimeError) as exc:
                failed += 1
                rows.append({"path": record["path"], "error": type(exc).__name__})
    return {**summarize(targets, policy_predictions, requested_support),
            "raw_argmax": summarize(targets, predictions, requested_support),
            "policy": "confidence below 0.5 becomes unknown; failures count as false negatives",
            "count": len(records), "failures": failed,
            "failure_rate": failed / max(1, len(records)),
            "p95_ms": float(np.percentile(latency, 95)) if latency else None,
            "rss_bytes": psutil.Process().memory_info().rss, "predictions": rows}


def run(manifest: Path, output: Path, epochs=5, max_seconds=1800):
    torch.set_num_threads(2)
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_split = {split: [r for r in records if r["split"] == split]
                for split in ("train", "train-validation", "development", "holdout")}
    if not all(by_split.values()) or len(by_split["train"]) + len(by_split["train-validation"]) > 2000:
        raise ValueError("Expected four nonempty splits and at most 2000 training/development-of-training crops")
    groups = {}
    hashes = {}
    for record in records:
        path = (manifest.parent / record["path"]).resolve()
        if manifest.parent.resolve() not in path.parents:
            raise ValueError("Crop path must remain inside the manifest directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Crop integrity mismatch")
        for mapping, key in ((groups, record["group_id"]), (hashes, record["pixel_sha256"])):
            previous = mapping.setdefault(key, record["split"])
            if previous != record["split"]:
                raise ValueError("Cross-split group or pixel duplicate")
    output.mkdir(parents=True, exist_ok=True)
    report = {"seed": 42, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "provenance": "local-unverified", "promotion_allowed": False, "models": {}}
    for architecture in ("resnet18", "mobilenetv3"):
        torch.manual_seed(42)
        np.random.seed(42)
        started = time.monotonic()
        model = make_model(architecture, pretrained=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        head = model.fc if architecture == "resnet18" else model.classifier[-1]
        for parameter in head.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
        loader = DataLoader(Crops(by_split["train"], manifest.parent), batch_size=32, shuffle=True,
                            generator=torch.Generator().manual_seed(42), num_workers=0)
        directory = output / architecture
        directory.mkdir(exist_ok=True)
        best_loss, best_epoch, timed_out = float("inf"), 0, False
        history = []
        for epoch in range(1, epochs + 1):
            model.eval()  # Freeze backbone batch-normalization statistics as well as parameters.
            head.train()
            for images, labels in loader:
                if time.monotonic() - started >= max_seconds:
                    timed_out = True
                    break
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(images), labels)
                loss.backward()
                optimizer.step()
            if timed_out:
                break
            model.eval()
            losses = []
            with torch.inference_mode():
                for images, labels in DataLoader(Crops(by_split["train-validation"], manifest.parent), batch_size=32):
                    if time.monotonic() - started >= max_seconds:
                        timed_out = True
                        break
                    losses.extend(torch.nn.functional.cross_entropy(model(images), labels, reduction="none").tolist())
            if timed_out:
                break
            validation_loss = float(np.mean(losses))
            history.append({"epoch": epoch, "validation_loss": validation_loss})
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                torch.save(model.state_dict(), directory / "best_model.pth")
            print(architecture, epoch, validation_loss, flush=True)
        entry = {"history": history, "best_epoch": best_epoch, "timed_out": timed_out,
                 "training_seconds": time.monotonic() - started}
        if best_epoch:
            model.load_state_dict(torch.load(directory / "best_model.pth", weights_only=True), strict=True)
            for split in ("development", "holdout"):
                entry[split] = evaluate(model, by_split[split], manifest.parent)
            (directory / "model_metadata.json").write_text(json.dumps({"cnn_contract": contract(architecture),
                "best_epoch": best_epoch, "seed": 42, "manifest_sha256": report["manifest_sha256"]}, indent=2))
            (directory / "class_indices.json").write_text(json.dumps(CLASS_MAPPING))
        report["models"][architecture] = entry
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    baseline = report["models"]["resnet18"].get("holdout")
    candidate = report["models"]["mobilenetv3"].get("holdout")
    if baseline and candidate:
        report["comparison"] = {
            "macro_f1_gain": candidate["macro_f1"] - baseline["macro_f1"],
            "recall_delta": {k: candidate["recall"][k] - baseline["recall"][k] for k in CLASS_MAPPING},
            "p95_ratio": candidate["p95_ms"] / baseline["p95_ms"] if baseline["p95_ms"] and candidate["p95_ms"] else None,
            "promotion_allowed": False,
            "reason": "Local annotation provenance unverified; crop timing is not end-to-end pipeline timing",
        }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--max-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    if not 0 < args.max_seconds <= 1800:
        parser.error("max-seconds must be between 1 and 1800")
    run(args.manifest, args.output, args.epochs, args.max_seconds)


if __name__ == "__main__":
    main()
