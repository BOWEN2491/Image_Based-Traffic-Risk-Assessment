from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import cv2
from PIL import Image

from traffic_risk.cnn_contract import CLASS_MAPPING, classify_logits, classify_probabilities, contract, inference_transform, make_model
from traffic_risk.model_runtime import ModelRuntime
from traffic_risk.predict_tl_color import load_model, predict_image


@pytest.mark.parametrize(("probability", "expected", "abstained"), [
    (0.49, "unknown", True),
    (0.50, "green", False),
    (0.51, "green", False),
])
def test_shared_probability_policy_threshold(probability, expected, abstained):
    remainder = (1 - probability) / 3
    label, confidence, raw_label, did_abstain = classify_probabilities(
        torch.tensor([probability, remainder, remainder, remainder], dtype=torch.float64)
    )
    assert label == expected
    assert raw_label == "green"
    assert confidence == pytest.approx(probability)
    assert did_abstain is abstained


def test_shared_probability_policy_preserves_raw_unknown_class():
    logits = torch.log(torch.tensor([0.1, 0.1, 0.7, 0.1], dtype=torch.float64))
    label, confidence, raw_label, abstained = classify_logits(logits)
    assert (label, raw_label, abstained) == ("unknown", "unknown", False)
    assert confidence == pytest.approx(0.7)


@pytest.mark.parametrize("logits", [[1.0, 2.0], [1.0, float("nan"), 3.0, 4.0]])
def test_shared_probability_policy_rejects_invalid_logits(logits):
    with pytest.raises(ValueError):
        classify_logits(logits)


@pytest.mark.parametrize("architecture", ["resnet18", "mobilenetv3"])
def test_standalone_and_runtime_contract_match(tmp_path, architecture):
    torch.set_num_threads(1)
    model_dir = tmp_path / "cnn"
    model_dir.mkdir()
    checkpoint = model_dir / "best_model.pth"
    mapping_path = model_dir / "class_indices.json"
    metadata_path = model_dir / "model_metadata.json"
    torch.manual_seed(17)
    runtime_model = make_model(architecture)
    torch.save(runtime_model.state_dict(), checkpoint)
    mapping_path.write_text(json.dumps(CLASS_MAPPING), encoding="utf-8")
    metadata_path.write_text(json.dumps({"cnn_contract": contract(architecture)}), encoding="utf-8")

    standalone, names = load_model(checkpoint, mapping_path)
    assert names == {index: name for name, index in CLASS_MAPPING.items()}
    assert standalone._traffic_risk_cnn_contract == contract(architecture)

    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    image_path = tmp_path / "scene.png"
    Image.fromarray(rgb).save(image_path)
    standalone_label, standalone_probability = predict_image(standalone, names, image_path)

    runtime = object.__new__(ModelRuntime)
    runtime.cnn = runtime_model.eval()
    runtime.cnn_transform = inference_transform()
    runtime.settings = SimpleNamespace(device="cpu")
    runtime.class_names = {index: name for name, index in CLASS_MAPPING.items()}
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    runtime_label = runtime._traffic_light_color(bgr, (0, 0, 128, 128))
    with Image.open(image_path) as image:
        runtime_tensor = runtime.cnn_transform(image.convert("RGB")).unsqueeze(0)
        standalone_tensor = standalone._traffic_risk_inference_transform(image.convert("RGB")).unsqueeze(0)
    torch.testing.assert_close(standalone_tensor, runtime_tensor)
    with torch.inference_mode():
        runtime_logits = runtime_model(runtime_tensor)
        _, runtime_probability, _, _ = classify_logits(runtime_logits)
        standalone_logits = standalone(standalone_tensor)
        loaded_label, loaded_probability, _, _ = classify_logits(standalone_logits)
    assert standalone_label == runtime_label == loaded_label
    assert standalone_probability == pytest.approx(runtime_probability)
    assert standalone_probability == pytest.approx(loaded_probability)
    assert bgr.shape == rgb.shape


def test_legacy_resnet_loads_and_mobilenet_without_metadata_fails(tmp_path):
    torch.set_num_threads(1)
    mapping_path = tmp_path / "class_indices.json"
    mapping_path.write_text(json.dumps(CLASS_MAPPING), encoding="utf-8")

    legacy_path = tmp_path / "legacy.pth"
    torch.save(make_model("resnet18").state_dict(), legacy_path)
    with pytest.warns(RuntimeWarning, match="legacy ResNet18"):
        legacy, _ = load_model(legacy_path, mapping_path)
    assert legacy._traffic_risk_cnn_contract == contract("resnet18")

    mobile_path = tmp_path / "mobile.pth"
    torch.save(make_model("mobilenetv3").state_dict(), mobile_path)
    with pytest.warns(RuntimeWarning, match="legacy ResNet18"):
        with pytest.raises(RuntimeError):
            load_model(mobile_path, mapping_path)


def test_invalid_or_missing_explicit_metadata_does_not_fall_back(tmp_path):
    mapping_path = tmp_path / "class_indices.json"
    mapping_path.write_text(json.dumps(CLASS_MAPPING), encoding="utf-8")
    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_bytes(b"not-loaded")
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"cnn_contract": {**contract(), "image_size": 32}}), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        load_model(checkpoint, mapping_path, metadata_path=invalid)
    sidecar = tmp_path / "model_metadata.json"
    sidecar.write_text(invalid.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        load_model(checkpoint, mapping_path)
    with pytest.raises(FileNotFoundError):
        load_model(checkpoint, mapping_path, metadata_path=tmp_path / "missing.json")


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_predict_image_converts_rgb_modes(tmp_path, mode):
    class Predictor:
        _traffic_risk_inference_transform = inference_transform()

        def __call__(self, tensor):
            return torch.tensor([[4.0, 0.0, 0.0, 0.0]], dtype=tensor.dtype)

    image_path = tmp_path / f"{mode}.png"
    Image.new(mode, (20, 20), 128).save(image_path)
    assert predict_image(Predictor(), {index: name for name, index in CLASS_MAPPING.items()}, image_path)[0] == "green"


def test_predict_image_rejects_bad_image(tmp_path):
    class Predictor:
        _traffic_risk_inference_transform = inference_transform()

        def __call__(self, tensor):
            return torch.zeros((1, 4), dtype=tensor.dtype)

    bad_image = tmp_path / "bad.png"
    bad_image.write_bytes(b"not an image")
    with pytest.raises(OSError):
        predict_image(Predictor(), {index: name for name, index in CLASS_MAPPING.items()}, bad_image)


def test_prediction_details_reject_wrong_mapping(tmp_path):
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (8, 8)).save(image_path)
    with pytest.raises(ValueError, match="class names"):
        predict_image(object(), {0: "green", 1: "red"}, image_path)
