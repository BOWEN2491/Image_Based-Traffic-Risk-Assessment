from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from traffic_risk.config import Settings
from traffic_risk.model_runtime import ModelRuntime
from traffic_risk.pipeline import run_single_image


def box(category, coordinates):
    values = np.asarray([coordinates], dtype=float)
    return SimpleNamespace(
        cls=SimpleNamespace(item=lambda: category),
        conf=SimpleNamespace(item=lambda: 0.9),
        xyxy=SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: values)),
    )


@pytest.mark.parametrize("boxes,expected", [
    ([box(0, [10, 10, 30, 40]), box(1, [35, 30, 65, 100])], "high"),
    ([box(0, [10, 10, 30, 40])], "unknown"),
    ([], "unknown"),
])
def test_rules_preserves_detections_without_cnn(tmp_path, monkeypatch, boxes, expected):
    path = tmp_path / "scene.png"
    Image.new("RGB", (100, 100)).save(path)
    runtime = ModelRuntime(Settings(mode="rules"))
    result = SimpleNamespace(names={0: "traffic light", 1: "person"}, boxes=boxes)
    calls = []
    runtime.yolo = SimpleNamespace(predict=lambda **kwargs: calls.append(kwargs) or [result])

    def forbidden(*args, **kwargs):
        pytest.fail("Public rules must not call the unavailable CNN")

    monkeypatch.setattr(runtime, "_traffic_light_color", forbidden)
    detected = runtime.detect(path)["frames"][0]["objects"]
    assert len(detected) == len(boxes)
    assert calls[0]["device"] == "cpu"
    assert runtime.cnn is None
    if detected:
        assert detected[0]["attributes"]["trafficLightColor"] == "unknown"
    prediction = run_single_image(path, runtime)
    assert prediction["risk"] == expected
    assert prediction["status"] == ("ok" if expected == "high" else "uncertain")


def test_models_still_classifies_traffic_light(tmp_path, monkeypatch):
    path = tmp_path / "scene.png"
    Image.new("RGB", (100, 100)).save(path)
    runtime = ModelRuntime(Settings(mode="models", model_manifest=tmp_path / "manifest.json"))
    runtime.cnn = object()
    result = SimpleNamespace(names={0: "traffic light"}, boxes=[box(0, [10, 10, 30, 40])])
    runtime.yolo = SimpleNamespace(predict=lambda **kwargs: [result])
    calls = []
    monkeypatch.setattr(runtime, "_traffic_light_color", lambda image, coords: calls.append(coords) or "red")
    objects = runtime.detect(path)["frames"][0]["objects"]
    assert objects[0]["attributes"]["trafficLightColor"] == "red"
    assert calls == [(10, 10, 30, 40)]
