from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.model_runtime import PerceptionError
from src.pipeline import run_single_image


class Runtime:
    versions = {"test": "1"}

    def __init__(self, objects, risk="low"):
        self.objects = objects
        self.risk = risk

    def detect(self, _path):
        if isinstance(self.objects, Exception):
            raise self.objects
        return {"frames": [{"objects": self.objects}]}

    def predict_risk(self, _features):
        return self.risk


def image(tmp_path: Path) -> Path:
    path = tmp_path / "scene.png"
    Image.new("RGB", (100, 100)).save(path)
    return path


def car():
    return {"category": "car", "box2d": {"x1": 40, "y1": 10, "x2": 60, "y2": 30}, "attributes": {}}


def test_successful_model_path(tmp_path):
    result = run_single_image(image(tmp_path), Runtime([car()], "medium"))
    assert result["status"] == "ok"
    assert result["risk"] == "medium"


def test_hard_rule_overrides_model(tmp_path):
    pedestrian = {"category": "person", "box2d": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}, "attributes": {}}
    result = run_single_image(image(tmp_path), Runtime([pedestrian], "low"))
    assert result["risk"] == "high"


def test_empty_detection_is_unknown(tmp_path):
    result = run_single_image(image(tmp_path), Runtime([]))
    assert result["status"] == "uncertain"
    assert result["risk"] == "unknown"


def test_perception_failure_is_unknown(tmp_path):
    result = run_single_image(image(tmp_path), Runtime(PerceptionError("secret C:/model/path")))
    assert result["risk"] == "unknown"
    assert "secret" not in result["reason"] and "C:/" not in result["reason"]


def test_value_error_is_unknown_and_sanitized(tmp_path):
    class InvalidRuntime(Runtime):
        def detect(self, _path):
            raise ValueError("secret implementation detail")

    result = run_single_image(image(tmp_path), InvalidRuntime([]))
    assert result["status"] == "uncertain"
    assert result["risk"] == "unknown"
    assert "secret" not in result["reason"]


def test_invalid_model_class_is_unknown(tmp_path):
    result = run_single_image(image(tmp_path), Runtime([car()], "unexpected"))
    assert result["status"] == "uncertain"
