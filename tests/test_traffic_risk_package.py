from __future__ import annotations

import io
import json
import hashlib
import sys
import types
import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.traffic_risk.build_features import compute_features
from src.traffic_risk.config import Settings
from src.traffic_risk.contracts import AssetRecord, ModelManifest, ModelMetadata
from src.traffic_risk.hard_rules import apply_hard_rules, apply_signal_policy
from src.traffic_risk.pipeline import run_single_image
from src.traffic_risk.schema import MODEL_FEATURES, FeatureSchemaError, public_features, validate_model_features
from src.traffic_risk.model_runtime import ModelRuntime, ModelUnavailableError, PerceptionError


def _image() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(stream, format="PNG")
    return stream.getvalue()


def _settings(tmp_path: Path, **changes) -> Settings:
    values = {"model_dir": tmp_path / "models", "upload_dir": tmp_path / "uploads", "max_upload_bytes": 5000}
    values.update(changes)
    return Settings(**values)


def test_contracts_schema_and_settings(tmp_path, monkeypatch):
    asset = AssetRecord(path="x", size=0, sha256="a" * 64)
    assert asset.size == 0
    manifest = ModelManifest(assets={"x": asset}, distribution_status="withdrawn")
    assert manifest.is_withdrawn()
    assert ModelMetadata(feature_order=list(MODEL_FEATURES)).schema_version == "2.0.0"
    assert Settings(model_dir=tmp_path).yolo_path.name == "yolo11n.pt"
    with pytest.raises(ValueError):
        Settings(mode="invalid")
    with pytest.raises(ValueError):
        Settings(mode="models")
    monkeypatch.setenv("RISK_MODE", "models")
    monkeypatch.setenv("RISK_MODEL_MANIFEST", str(tmp_path / "manifest.json"))
    assert Settings().mode == "models"


def test_feature_extraction_and_validation(tmp_path):
    objects = [
        {"category": "person", "box2d": {"x1": 0, "y1": 0, "x2": 20, "y2": 30}},
        {"category": "car", "box2d": {"x1": 10, "y1": 10, "x2": 35, "y2": 35}},
        {"category": "traffic light", "attributes": {"trafficLightColor": "red"}, "box2d": {"x1": 18, "y1": 1, "x2": 23, "y2": 8}},
        {"category": "traffic sign", "box2d": {"x1": 1, "y1": 1, "x2": 3, "y2": 3}},
        {"category": "bicycle", "box2d": {"x1": 0, "y1": 0, "x2": 2, "y2": 2}},
        {"category": "car", "box2d": {"x1": "bad"}},
    ]
    features = compute_features(objects, 40, 40)
    assert features["n_human"] == 2 and features["n_vehicle"] == 1
    assert features["n_tl_red"] == 1 and features["central_tl_is_red"] == 1
    assert len(validate_model_features(features)) == 11
    assert public_features({"x": float("nan"), "ok": 1, "flag": True}) == {"ok": 1, "flag": True}
    with pytest.raises(ValueError):
        compute_features([], 0, 1)
    with pytest.raises(ValueError):
        compute_features([], 1, 1, center_width_pct=0)
    with pytest.raises(FeatureSchemaError):
        validate_model_features({})


def test_rules_and_pipeline_rules_mode(tmp_path):
    features = dict.fromkeys(MODEL_FEATURES, 0.0)
    features.update({"central_tl_is_red": 1, "n_human": 1})
    assert apply_hard_rules(features)[0] == "high"
    assert apply_signal_policy("low", {"central_tl_is_yellow": 1, "near_vehicle_count": 1})[0] == "medium"

    class Runtime:
        versions = {"mode": "rules"}
        settings = type("S", (), {"mode": "rules"})()
        def detect(self, _path):
            return {"frames": [{"objects": [{"category": "car", "box2d": {"x1": 1, "y1": 1, "x2": 5, "y2": 5}}]}]}
        def predict_risk(self, _features):
            return "low"

    path = tmp_path / "x.png"
    Image.new("RGB", (40, 40)).save(path)
    result = run_single_image(path, Runtime())
    assert result["status"] == "uncertain" and result["risk"] == "unknown"


def test_new_api_health_and_not_ready(tmp_path):
    from src.traffic_risk.app import create_app
    class Runtime:
        def __init__(self, _settings): self.loaded = False
        def load(self): self.loaded = True
    with TestClient(create_app(_settings(tmp_path), Runtime)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ready", "mode": "rules"}
    class Broken:
        def __init__(self, _settings): pass
        def load(self): raise RuntimeError("no model")
    with TestClient(create_app(_settings(tmp_path), Broken)) as client:
        assert client.get("/ready").status_code == 503


def test_model_runtime_manifest_rules_and_failures(tmp_path):
    asset = tmp_path / "models" / "yolo" / "yolo11n.pt"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": "2.0.0", "model_versions": {"yolo": "test"}, "assets": [{"destination": "yolo/yolo11n.pt", "size": 7, "sha256": digest}]}))
    runtime = ModelRuntime(_settings(tmp_path, model_manifest=manifest))
    loaded = runtime._verify_manifest()
    assert loaded["schema_version"] == "2.0.0" and runtime.versions["mode"] == "rules"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"distribution_status": "withdrawn"}))
    with pytest.raises(ModelUnavailableError, match="withdrawn"):
        ModelRuntime(_settings(tmp_path, model_manifest=bad))._verify_manifest()
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"assets": []}))
    with pytest.raises(ModelUnavailableError, match="YOLO"):
        ModelRuntime(_settings(tmp_path, model_manifest=missing))._verify_manifest()


def test_model_runtime_prediction_and_detection_errors(tmp_path, monkeypatch):
    settings = _settings(tmp_path, mode="models", model_manifest=tmp_path / "manifest.json")
    runtime = ModelRuntime(settings)
    with pytest.raises(ModelUnavailableError):
        runtime.predict_risk({})
    runtime.booster = type("Booster", (), {"predict": lambda self, matrix: [2],})()
    fake_xgb = types.SimpleNamespace(DMatrix=lambda values, feature_names: (values, feature_names))
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgb)
    assert runtime.predict_risk(dict.fromkeys(MODEL_FEATURES, 0.0)) == "high"
    runtime.yolo = object()
    with pytest.raises(ModelUnavailableError):
        runtime.detect(tmp_path / "missing.png")
    runtime.settings = _settings(tmp_path)
    with pytest.raises(PerceptionError):
        runtime.detect(tmp_path / "missing.png")


def test_pipeline_failure_paths(tmp_path):
    class Runtime:
        versions = {"mode": "models"}
        settings = type("S", (), {"mode": "models"})()
        def __init__(self, failure=None, result=None): self.failure, self.result = failure, result
        def detect(self, _path):
            if self.failure: raise self.failure
            return self.result or {"frames": []}
        def predict_risk(self, _features): return "not-a-risk"
    path = tmp_path / "x.png"
    Image.new("RGB", (20, 20)).save(path)
    assert run_single_image(path, Runtime())['risk'] == 'unknown'
    assert run_single_image(path, Runtime(PerceptionError("bad")))['risk'] == 'unknown'
    assert run_single_image(path, Runtime(ValueError("bad")))['risk'] == 'unknown'


def test_cli_parser_and_workflow_errors(tmp_path):
    from src.traffic_risk.cli import build_parser, main
    assert set(build_parser()._subparsers._group_actions[0].choices) == {
        "predict", "download-models", "build-features", "generate-weak-labels",
        "train-risk", "train-traffic-light", "build-local-manifest", "review-roi", "merge-review",
    }
    with pytest.raises(FileNotFoundError):
        main(["build-local-manifest", "--model-dir", str(tmp_path / "missing"), "--output", str(tmp_path / "m.json")])
    with pytest.raises(FileNotFoundError):
        main(["predict", str(tmp_path / "missing.png")])
    model_dir = tmp_path / "models"
    for relative in ("yolo/yolo11n.pt", "cnn/best_model.pth", "cnn/class_indices.json", "risk/risk_xgb.ubj", "risk/feature_order.json", "risk/model_metadata.json"):
        path = model_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    output = tmp_path / "manifest.json"
    assert main(["build-local-manifest", "--model-dir", str(model_dir), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["schema_version"] == "2.0.0"


def test_new_api_validation_helpers(tmp_path):
    import src.traffic_risk.app as api
    from starlette.datastructures import UploadFile
    with pytest.raises(Exception):
        api._validate_extension("x.gif")
    assert api._magic_format(_image()) == "PNG"
    with pytest.raises(Exception):
        api._validate_image(b"bad", "image/png", 100)
    with pytest.raises(Exception):
        api._validate_image(_image(), "image/jpeg", 100)
    with pytest.raises(Exception):
        api._validate_image(_image(), "image/png", 1)
    upload = UploadFile(filename="x.png", file=io.BytesIO(_image()))
    assert asyncio.run(api._read_upload(upload, 5000))
    empty = UploadFile(filename="x.png", file=io.BytesIO(b""))
    with pytest.raises(Exception):
        asyncio.run(api._read_upload(empty, 10))
    oversized = UploadFile(filename="x.png", file=io.BytesIO(_image()))
    with pytest.raises(Exception):
        asyncio.run(api._read_upload(oversized, 1))


def test_new_api_predict_success_and_upload_rejections(tmp_path):
    from src.traffic_risk.app import create_app
    class Runtime:
        versions = {"mode": "rules", "feature_schema": "2.0.0", "yolo": "test"}
        def __init__(self, _settings): pass
        def load(self): pass
        def detect(self, _path):
            return {"frames": [{"objects": [{"category": "car", "box2d": {"x1": 1, "y1": 1, "x2": 5, "y2": 5}}]}]}
        def predict_risk(self, _features): return "low"
    with TestClient(create_app(_settings(tmp_path), Runtime)) as client:
        response = client.post("/api/predict", files={"file": ("scene.png", _image(), "image/png")})
        assert response.status_code == 200 and response.json()["risk"] == "low"
        assert client.post("/api/predict", files={"file": ("scene.png", _image(), "image/jpeg")}).status_code == 415


def test_models_manifest_asset_set_validation(tmp_path):
    root = tmp_path / "models"
    destinations = ["yolo/yolo11n.pt", "cnn/best_model.pth", "cnn/class_indices.json", "risk/risk_xgb.ubj", "risk/feature_order.json", "risk/model_metadata.json"]
    assets = []
    for destination in destinations:
        path = root / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        data = destination.encode()
        path.write_bytes(data)
        assets.append({"destination": destination, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": "2.0.0", "model_versions": {"yolo": "a", "traffic_light_cnn": "b", "risk_xgb": "c"}, "assets": assets}))
    runtime = ModelRuntime(_settings(tmp_path, mode="models", model_manifest=manifest_path))
    assert runtime._verify_manifest()["schema_version"] == "2.0.0"


def test_runtime_rules_load_and_invalid_prediction(tmp_path, monkeypatch):
    asset = tmp_path / "models" / "yolo" / "yolo11n.pt"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"weights")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": "2.0.0", "model_versions": {"yolo": "test"}, "assets": [{"destination": "yolo/yolo11n.pt", "size": 7, "sha256": hashlib.sha256(b"weights").hexdigest()}]}))
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=lambda path: {"path": path}))
    runtime = ModelRuntime(_settings(tmp_path, model_manifest=manifest))
    runtime.load()
    assert runtime.yolo["path"].endswith("yolo11n.pt")
    class BadBooster:
        def predict(self, _matrix): return [0.2, 0.3]
    runtime.settings = _settings(tmp_path, mode="models", model_manifest=manifest)
    runtime.booster = BadBooster()
    with pytest.raises(PerceptionError):
        runtime.predict_risk(dict.fromkeys(MODEL_FEATURES, 0.0))


def test_runtime_detects_supported_objects(tmp_path):
    import numpy as np
    image = tmp_path / "scene.png"
    Image.new("RGB", (30, 30), "white").save(image)
    class Scalar:
        def __init__(self, value): self.value = value
        def item(self): return self.value
    class Box:
        cls = Scalar(0)
        xyxy = type("Coords", (), {"cpu": lambda self: self, "numpy": lambda self: np.array([[1, 1, 10, 10]])})()
    class Result:
        names = {0: "car", 1: "unknown"}
        boxes = [Box()]
    class Yolo:
        def predict(self, **_kwargs): return [Result()]
    runtime = ModelRuntime(_settings(tmp_path))
    runtime.yolo = Yolo()
    result = runtime.detect(image)
    assert result["frames"][0]["objects"][0]["category"] == "car"


@pytest.mark.parametrize("changes", [
    {"near_person_count": 1}, {"max_box_area_person": 0.01, "bottom_half_person_ratio": 0.7},
    {"max_box_area_vehicle": 0.17}, {"near_vehicle_count": 2},
    {"center_region_vehicle_ratio": 0.9, "sum_box_area_vehicle": 0.3},
    {"object_density_per_mp": 31, "sum_box_area_vehicle": 0.4},
    {"mean_overlap_iou": 0.02, "max_box_area_vehicle": 0.13},
])
def test_every_conservative_rule(changes):
    row = dict.fromkeys(MODEL_FEATURES, 0.0)
    row.update(changes)
    assert apply_hard_rules(row)[0] == "high"


def test_identity_helpers_and_pipeline_validation(tmp_path):
    from src.sample_identity import IdentityConflict, SampleRecord, assert_destinations_free, build_records, sample_id_for, write_jsonl
    root = tmp_path / "data"
    root.mkdir()
    first = root / "a.png"
    first.write_bytes(b"a")
    assert sample_id_for(root, first) == "a.png"
    with pytest.raises(IdentityConflict):
        sample_id_for(root, tmp_path / "outside.png")
    records = build_records(root, [first], {"a.png": "x"})
    target = tmp_path / "manifest.jsonl"
    write_jsonl(records, target)
    assert target.read_text().count("a.png") == 2
    assert_destinations_free([tmp_path / "new.txt"])
    with pytest.raises(IdentityConflict):
        assert_destinations_free([first])
    with pytest.raises(FileNotFoundError):
        build_records(root, [root / "missing.png"])
    with pytest.raises(IdentityConflict):
        write_jsonl([SampleRecord("a", "a.png", "1", label="x"), SampleRecord("b", "b.png", "1", label="y")], target)


def test_pipeline_invalid_detector_shapes(tmp_path):
    path = tmp_path / "x.png"
    Image.new("RGB", (10, 10)).save(path)
    class Runtime:
        versions = {}
        def detect(self, _path): return {"frames": [{"objects": []}]}
    assert run_single_image(path, Runtime())["risk"] == "unknown"
    class Bad:
        versions = {}
        def detect(self, _path): return {"frames": [{"objects": [{"category": "car"}]}]}
        def predict_risk(self, _features): return "low"
    assert run_single_image(path, Bad())["risk"] == "unknown"


def test_xgb_training_rejects_incomplete_classes(tmp_path):
    import pandas as pd
    from src.build_XGBoost import train
    frame = pd.DataFrame([{**dict.fromkeys(MODEL_FEATURES, 0.0), "risk_weak": 0} for _ in range(4)])
    csv_path = tmp_path / "features.csv"
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="exactly classes"):
        train(csv_path, tmp_path / "out")

