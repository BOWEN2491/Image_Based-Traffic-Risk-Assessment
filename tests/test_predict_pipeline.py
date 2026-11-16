from __future__ import annotations

from pathlib import Path

from PIL import Image

from traffic_risk.model_runtime import PerceptionError
from traffic_risk.pipeline import run_single_image


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
    assert result["reason_code"] == "model_prediction"


def test_hard_rule_overrides_model(tmp_path):
    pedestrian = {"category": "person", "box2d": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}, "attributes": {}}
    result = run_single_image(image(tmp_path), Runtime([pedestrian], "low"))
    assert result["risk"] == "high"
    assert result["reason_code"] == "hard_rule_triggered"


def test_empty_detection_is_unknown(tmp_path):
    result = run_single_image(image(tmp_path), Runtime([]))
    assert result["status"] == "uncertain"
    assert result["risk"] == "unknown"
    assert result["reason_code"] == "no_supported_detections"


def test_perception_failure_is_unknown(tmp_path):
    result = run_single_image(image(tmp_path), Runtime(PerceptionError("secret C:/model/path")))
    assert result["risk"] == "unknown"
    assert result["reason_code"] == "perception_error"
    assert "secret" not in result["reason"] and "C:/" not in result["reason"]


def test_value_error_is_unknown_and_sanitized(tmp_path):
    class InvalidRuntime(Runtime):
        def detect(self, _path):
            raise ValueError("secret implementation detail")

    result = run_single_image(image(tmp_path), InvalidRuntime([]))
    assert result["status"] == "uncertain"
    assert result["risk"] == "unknown"
    assert "secret" not in result["reason"]
    assert result["reason_code"] == "invalid_features"


def test_invalid_model_class_is_unknown(tmp_path):
    result = run_single_image(image(tmp_path), Runtime([car()], "unexpected"))
    assert result["status"] == "uncertain"
    assert result["reason_code"] == "invalid_model_output"


def test_feature_schema_failure_has_explicit_code(tmp_path, monkeypatch):
    from traffic_risk.pipeline import FeatureSchemaError

    monkeypatch.setattr("traffic_risk.pipeline.validate_model_features", lambda _value: (_ for _ in ()).throw(FeatureSchemaError("bad schema")))
    result = run_single_image(image(tmp_path), Runtime([car()]))
    assert result["reason_code"] == "feature_schema_error"


def test_rules_abstention_and_failures_are_distinguishable(tmp_path):
    from types import SimpleNamespace

    runtime = Runtime([car()])
    runtime.settings = SimpleNamespace(mode="rules")
    result = run_single_image(image(tmp_path), runtime)
    assert result["risk"] == "unknown"
    assert result["reason_code"] == "no_rule_triggered"
