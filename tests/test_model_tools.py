from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools import download_models
from traffic_risk.config import Settings
from traffic_risk import model_runtime
from traffic_risk.model_runtime import ModelRuntime, ModelUnavailableError, PerceptionError
from traffic_risk.schema import MODEL_FEATURES, SCHEMA_VERSION


def _write_runtime_assets(root: Path, *, mapping=None, feature_order=None) -> Path:
    contents = {
        "yolo/yolo11n.pt": b"yolo",
        "cnn/best_model.pth": b"cnn",
        "cnn/class_indices.json": json.dumps(
            mapping or {"green": 0, "red": 1, "unknown": 2, "yellow": 3}
        ).encode(),
        "risk/risk_xgb.ubj": b"xgb",
        "risk/feature_order.json": json.dumps(
            feature_order if feature_order is not None else list(MODEL_FEATURES)
        ).encode(),
        "risk/model_metadata.json": json.dumps(
            {"train_cols_nosignal": list(MODEL_FEATURES)}
        ).encode(),
    }
    assets = []
    for destination, payload in contents.items():
        path = root / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        asset = {
            "name": path.name,
            "destination": destination,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if path.name == "best_model.pth":
            asset["class_mapping"] = {"green": 0, "red": 1, "unknown": 2, "yellow": 3}
        if path.name == "risk_xgb.ubj":
            asset["feature_schema"] = {
                "version": SCHEMA_VERSION,
                "features": list(MODEL_FEATURES),
            }
        assets.append(asset)
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "release": "v0.1.0",
                "feature_schema": SCHEMA_VERSION,
                "model_versions": {
                    "yolo": "yolo11n",
                    "traffic_light_cnn": "v0.1.0",
                    "risk_xgb": "v0.1.0",
                },
                "assets": assets,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_download_and_verify(monkeypatch, tmp_path):
    payload = b"trusted model"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"assets": [{"name": "model", "destination": "risk/model.ubj", "url": "https://example.test/model", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]}), encoding="utf-8")
    monkeypatch.setattr(download_models, "MANIFEST", manifest)
    monkeypatch.setattr(download_models.urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(payload))
    destination = tmp_path / "models"
    download_models.download(destination)
    download_models.download(destination)
    assert (destination / "risk/model.ubj").read_bytes() == payload


def test_bad_digest_cleans_partial(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"assets": [{"name": "model", "destination": "model.ubj", "url": "https://example.test/model", "size": 3, "sha256": "0" * 64}]}), encoding="utf-8")
    monkeypatch.setattr(download_models, "MANIFEST", manifest)
    monkeypatch.setattr(download_models.urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(b"bad"))
    with pytest.raises(RuntimeError, match="verification"):
        download_models.download(tmp_path / "models")
    assert not (tmp_path / "models/model.ubj.part").exists()


def test_release_manifest_has_provenance_and_loader_contracts():
    manifest = json.loads(download_models.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["release"] == "v0.1.0"
    assert manifest["feature_schema"] == "1.0.0"
    assert manifest["model_versions"] == {
        "yolo": "yolo11n",
        "traffic_light_cnn": "v0.1.0",
        "risk_xgb": "v0.1.0",
    }
    assert len(manifest["assets"]) == 6
    for asset in manifest["assets"]:
        assert all(asset[field] for field in ("origin", "license", "format", "loader_constraints"))
        assert len(asset["sha256"]) == 64
        assert asset["size"] > 0

    assets = {asset["name"]: asset for asset in manifest["assets"]}
    cnn = assets["best_model.pth"]
    assert cnn["architecture"]["name"] == "torchvision.models.resnet18"
    assert cnn["class_mapping"] == {"green": 0, "red": 1, "unknown": 2, "yellow": 3}

    risk = assets["risk_xgb.ubj"]
    assert risk["loader_constraints"]["entrypoint"] == "xgboost.Booster.load_model"
    assert risk["loader_constraints"]["num_features"] == len(MODEL_FEATURES)
    assert risk["feature_schema"]["features"] == list(MODEL_FEATURES)
    assert risk["versions"]["training"]["xgboost"].startswith("unknown")
    assert risk["versions"]["conversion_runtime"]["xgboost"] == "3.2.0"


def test_packaged_manifest_is_available():
    assert download_models.MANIFEST.is_file()
    assert json.loads(download_models.MANIFEST.read_text(encoding="utf-8"))["release"] == "v0.1.0"


def test_model_runtime_predicts_with_native_booster_contract(monkeypatch, tmp_path):
    captured = {}

    class Matrix:
        def __init__(self, data, feature_names):
            captured["data"] = data
            captured["feature_names"] = feature_names

    class Booster:
        def predict(self, matrix):
            assert isinstance(matrix, Matrix)
            return [[0.05, 0.9, 0.05]]

    monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(DMatrix=Matrix))
    runtime = ModelRuntime(Settings(model_dir=tmp_path, mode="models", model_manifest=tmp_path / "manifest.json"))
    runtime.booster = Booster()
    features = dict.fromkeys(MODEL_FEATURES, 0.0)

    assert runtime.predict_risk(features) == "medium"
    assert captured["feature_names"] == list(MODEL_FEATURES)
    assert captured["data"] == [[0.0] * len(MODEL_FEATURES)]


def test_model_runtime_rejects_malformed_booster_output(monkeypatch, tmp_path):
    class Booster:
        def predict(self, _matrix):
            return [0.25, 0.75]

    monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(DMatrix=lambda *_args, **_kwargs: object()))
    runtime = ModelRuntime(Settings(model_dir=tmp_path, mode="models", model_manifest=tmp_path / "manifest.json"))
    runtime.booster = Booster()

    with pytest.raises(PerceptionError, match="invalid prediction"):
        runtime.predict_risk(dict.fromkeys(MODEL_FEATURES, 0.0))


def test_runtime_rejects_tampered_asset_before_loading(monkeypatch, tmp_path):
    manifest = _write_runtime_assets(tmp_path)
    monkeypatch.setattr(model_runtime, "MANIFEST", manifest)
    (tmp_path / "yolo/yolo11n.pt").write_bytes(b"tampered")
    with pytest.raises(ModelUnavailableError, match="integrity"):
        ModelRuntime(Settings(model_dir=tmp_path, mode="models", model_manifest=manifest)).load()


def test_runtime_rejects_wrong_feature_order_before_loading(monkeypatch, tmp_path):
    manifest = _write_runtime_assets(tmp_path, feature_order=list(reversed(MODEL_FEATURES)))
    monkeypatch.setattr(model_runtime, "MANIFEST", manifest)
    with pytest.raises(ModelUnavailableError, match="feature order"):
        ModelRuntime(Settings(model_dir=tmp_path, mode="models", model_manifest=manifest)).load()


def test_runtime_rejects_duplicate_class_index_before_loading(monkeypatch, tmp_path):
    manifest = _write_runtime_assets(
        tmp_path,
        mapping={"green": 0, "red": 0, "unknown": 2, "yellow": 3},
    )
    monkeypatch.setattr(model_runtime, "MANIFEST", manifest)
    with pytest.raises(ModelUnavailableError, match="class mapping"):
        ModelRuntime(Settings(model_dir=tmp_path, mode="models", model_manifest=manifest)).load()


def test_runtime_loads_verified_contract_with_fake_libraries(monkeypatch, tmp_path):
    manifest = _write_runtime_assets(tmp_path)
    monkeypatch.setattr(model_runtime, "MANIFEST", manifest)

    class FakeCNN:
        fc = SimpleNamespace(in_features=8)

        def load_state_dict(self, state, strict):
            assert state == {"weight": 1} and strict is True

        def eval(self):
            return self

        def to(self, device):
            assert device == "cpu"
            return self

    class FakeBooster:
        feature_names = list(MODEL_FEATURES)

        def load_model(self, path):
            assert Path(path).name == "risk_xgb.ubj"

        def num_features(self):
            return len(MODEL_FEATURES)

    fake_torch = ModuleType("torch")
    fake_torch.__path__ = []
    fake_torch.load = lambda *_args, **_kwargs: {"weight": 1}
    fake_nn = ModuleType("torch.nn")
    fake_nn.Linear = lambda *_args: object()
    fake_torch.nn = fake_nn
    fake_models = SimpleNamespace(resnet18=lambda weights=None: FakeCNN())
    fake_transforms = SimpleNamespace(
        Compose=lambda values: values,
        Resize=lambda value: ("resize", value),
        ToTensor=lambda: "tensor",
        Normalize=lambda mean, std: (mean, std),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_nn)
    monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(Booster=FakeBooster))
    monkeypatch.setitem(sys.modules, "torchvision", SimpleNamespace(models=fake_models, transforms=fake_transforms))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path: ("yolo", path)))

    runtime = ModelRuntime(Settings(model_dir=tmp_path, mode="models", model_manifest=manifest))
    runtime.load()
    assert runtime.versions == {
        "mode": "models",
        "release": "v0.1.0",
        "feature_schema": SCHEMA_VERSION,
        "yolo": "yolo11n",
        "traffic_light_cnn": "v0.1.0",
        "risk_xgb": "v0.1.0",
    }
    assert runtime.class_names == {0: "green", 1: "red", 2: "unknown", 3: "yellow"}
