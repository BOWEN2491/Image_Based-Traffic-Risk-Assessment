import json

import pytest
import torch
from PIL import Image

from traffic_risk import cnn_detected_eval
from traffic_risk.cnn_contract import contract


def test_detected_eval_keeps_misses_in_class_support(tmp_path, monkeypatch):
    def model(_name):
        return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4))

    monkeypatch.setattr(cnn_detected_eval, "make_model", model)
    for name in ("resnet18", "mobilenetv3"):
        directory = tmp_path / name
        directory.mkdir()
        torch.save(model(name).state_dict(), directory / "best_model.pth")
        (directory / "model_metadata.json").write_text(json.dumps({"cnn_contract": contract(name)}))
    Image.new("RGB", (4, 4), "red").save(tmp_path / "tiny.png")
    rows = [
        {"sample_id": "scene", "object_id": "0", "split": "holdout", "color": "red", "crop_path": None},
        {"sample_id": "scene", "object_id": "1", "split": "holdout", "color": "unknown", "crop_path": "tiny.png"},
    ]
    matches = tmp_path / "matches.jsonl"
    matches.write_text("\n".join(json.dumps(row) for row in rows))
    result = cnn_detected_eval.run(matches, tmp_path, tmp_path / "report.json")
    assert result["detector_misses"] == 1
    for value in result["models"].values():
        assert value["support"]["red"] == 1
        assert value["recall"]["red"] == 0
        assert value["recall"]["unknown"] == 1
        assert value["failure_rate"] == 0.5
    matches.write_text("\n".join(json.dumps(row) for row in rows + rows))
    with pytest.raises(ValueError, match="Duplicate"):
        cnn_detected_eval.run(matches, tmp_path, tmp_path / "report.json")
