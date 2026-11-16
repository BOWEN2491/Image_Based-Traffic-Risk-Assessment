import json
import hashlib

import pytest
import torch
from PIL import Image

from traffic_risk import cnn_experiment as experiment


def records(tmp_path):
    rows = []
    for split in ("train", "train-validation", "development", "holdout"):
        for label in experiment.CLASS_MAPPING:
            key = f"{split}-{label}"
            Image.new("RGB", (12, 12), "red").save(tmp_path / f"{key}.png")
            rows.append({"path": f"{key}.png", "label": label, "split": split,
                         "group_id": key, "pixel_sha256": key,
                         "sha256": hashlib.sha256((tmp_path / f"{key}.png").read_bytes()).hexdigest()})
    return rows


def test_run_saves_best_and_scores_both_architectures(tmp_path, monkeypatch):
    class Tiny(torch.nn.Module):
        def __init__(self, architecture):
            super().__init__()
            self.fc = torch.nn.Linear(3, 4)
            self.classifier = torch.nn.Sequential(torch.nn.Linear(3, 4))
            self.architecture = architecture

        def forward(self, images):
            head = self.fc if self.architecture == "resnet18" else self.classifier
            return head(images.mean(dim=(2, 3)))

    monkeypatch.setattr(experiment, "make_model", lambda name, pretrained: Tiny(name))
    manifest = tmp_path / "crops.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in records(tmp_path)))
    report = experiment.run(manifest, tmp_path / "output", epochs=2)
    assert report["promotion_allowed"] is False
    for entry in report["models"].values():
        assert entry["best_epoch"] in {1, 2}
        assert entry["holdout"]["count"] == 4
        assert entry["holdout"]["failures"] == 0
        assert entry["holdout"]["sufficient_support"] is False
    rows = records(tmp_path)
    rows[-1]["group_id"] = rows[0]["group_id"]
    manifest.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="Cross-split"):
        experiment.run(manifest, tmp_path / "invalid")
    rows = records(tmp_path)
    rows[0]["sha256"] = "0" * 64
    manifest.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="integrity"):
        experiment.run(manifest, tmp_path / "invalid")
    rows = records(tmp_path)
    manifest.write_text("\n".join(json.dumps(row) for row in rows))
    clock = iter(range(100))
    monkeypatch.setattr(experiment.time, "monotonic", lambda: next(clock))
    report = experiment.run(manifest, tmp_path / "timeout", max_seconds=0.1)
    assert all(entry["timed_out"] and entry["best_epoch"] == 0 for entry in report["models"].values())


def test_metrics_and_decode_failure(tmp_path):
    result = experiment.summarize([0, 1, 2, 3], [0, 1, 2, 3])
    assert result["macro_f1"] == 1
    model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4))
    result = experiment.evaluate(model, [{"path": "missing.png", "label": "red"}], tmp_path)
    assert result["failure_rate"] == 1
    assert result["p95_ms"] is None


def test_evaluate_tiny_runtime_crop_records_abstention(tmp_path):
    Image.new("RGB", (4, 4), "red").save(tmp_path / "tiny.png")
    model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4))
    result = experiment.evaluate(model, [{"path": "tiny.png", "label": "red", "runtime_size_gate": True}], tmp_path)
    assert result["failures"] == 0
    assert result["predictions"][0]["runtime_prediction"] == experiment.CLASS_MAPPING["unknown"]
    assert result["predictions"][0]["abstained"] is True


def test_invalid_time_limit(tmp_path):
    with pytest.raises(SystemExit):
        experiment.main(["--manifest", str(tmp_path), "--output", str(tmp_path), "--max-seconds", "1801"])
