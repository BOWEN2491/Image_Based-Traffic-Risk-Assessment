from __future__ import annotations

from pathlib import Path

import pytest

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
    return {"id": 0, "category": "car", "confidence": 0.7,
            "box2d": {"x1": 40, "y1": 10, "x2": 60, "y2": 30}, "attributes": {}}


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


def test_optional_detection_details_use_exif_oriented_dimensions_once(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6  # Display orientation is 90 degrees clockwise.
    Image.new("RGB", (120, 80)).save(path, exif=exif)
    obj = {"id": 2, "category": "car", "confidence": 0.8,
           "box2d": {"x1": 0, "y1": 0, "x2": 40, "y2": 60}, "attributes": {}}

    result = run_single_image(path, Runtime([obj]), include_detections=True)

    assert result["image_size"] == {"width": 80, "height": 120}
    import cv2

    assert cv2.imread(str(path)).shape[:2] == (result["image_size"]["height"], result["image_size"]["width"])
    assert result["detections"] == [{
        "id": 2, "category": "car", "bbox": [0.0, 0.0, 0.5, 0.5],
        "confidence": 0.8, "traffic_light_color": None,
    }]


def test_detection_details_are_omitted_by_default(tmp_path):
    assert "detections" not in run_single_image(image(tmp_path), Runtime([car()]))


@pytest.mark.parametrize("box", [
    {"x1": float("nan"), "y1": 0, "x2": 60, "y2": 30},
    {"x1": 0, "y1": 0, "x2": float("inf"), "y2": 30},
])
def test_optional_detection_details_reject_nonfinite_boxes(tmp_path, box):
    obj = {**car(), "box2d": box}
    result = run_single_image(image(tmp_path), Runtime([obj]), include_detections=True)
    assert result["reason_code"] == "invalid_features"
    assert "detections" not in result


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), 1.1])
def test_optional_detection_details_reject_invalid_confidence(tmp_path, confidence):
    obj = {**car(), "confidence": confidence}
    result = run_single_image(image(tmp_path), Runtime([obj]), include_detections=True)
    assert result["reason_code"] == "invalid_features"
    assert "detections" not in result


def test_model_runtime_exports_detector_confidence_for_optional_details(tmp_path):
    from types import SimpleNamespace

    import numpy as np

    from traffic_risk.model_runtime import ModelRuntime

    class Scalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class BoxTensor:
        def cpu(self):
            return self

        def numpy(self):
            return np.asarray([[1.0, 2.0, 30.0, 40.0]])

        def reshape(self, *_args):
            return self.numpy().reshape(-1)

    class Box:
        cls = Scalar(2)
        xyxy = BoxTensor()
        conf = Scalar(0.87)

    class Detector:
        def predict(self, **kwargs):
            self.kwargs = kwargs
            return [SimpleNamespace(names={2: "car"}, boxes=[Box()])]

    detector = Detector()
    runtime = ModelRuntime.__new__(ModelRuntime)
    runtime.settings = SimpleNamespace(mode="rules", device="cpu")
    runtime.yolo = detector
    runtime.cnn = None
    detected = runtime.detect(image(tmp_path))
    obj = detected["frames"][0]["objects"][0]
    assert obj["confidence"] == 0.87
    assert detector.kwargs["max_det"] == 300

    class DetectedRuntime(Runtime):
        def detect(self, _path):
            return detected

    details = run_single_image(image(tmp_path), DetectedRuntime([]), include_detections=True)
    assert details["detections"][0]["confidence"] == 0.87
