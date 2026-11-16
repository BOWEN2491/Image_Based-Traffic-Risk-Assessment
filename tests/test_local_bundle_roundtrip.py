"""Exercise generated manifests with real native CNN and XGBoost checkpoints."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from traffic_risk.cli import main
from traffic_risk.cnn_contract import CLASS_MAPPING, contract, make_model, validate_contract
from traffic_risk.config import Settings
from traffic_risk.model_runtime import ModelRuntime
from traffic_risk.schema import MODEL_FEATURES


@pytest.mark.parametrize("architecture", ["resnet18", "mobilenetv3"])
def test_native_training_manifest_runtime_roundtrip(tmp_path, monkeypatch, architecture):
    import torch
    import xgboost as xgb
    torch.set_num_threads(1)
    torch.manual_seed(42)
    for name in ("cnn", "risk", "yolo"):
        (tmp_path / name).mkdir()
    model = make_model(architecture)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss = torch.nn.functional.cross_entropy(model(torch.randn(2, 3, 128, 128)), torch.tensor([0, 1]))
    loss.backward()
    optimizer.step()
    torch.save(model.state_dict(), tmp_path / "cnn/best_model.pth")
    matrix = xgb.DMatrix(np.zeros((3, len(MODEL_FEATURES))), label=[0, 1, 2], feature_names=list(MODEL_FEATURES))
    booster = xgb.train({"objective": "multi:softprob", "num_class": 3, "nthread": 1}, matrix, num_boost_round=1)
    booster.save_model(tmp_path / "risk/risk_xgb.ubj")
    for relative, value in {
        "cnn/class_indices.json": CLASS_MAPPING,
        "cnn/model_metadata.json": {"cnn_contract": contract(architecture)},
        "risk/feature_order.json": list(MODEL_FEATURES),
        "risk/model_metadata.json": {"train_cols_nosignal": list(MODEL_FEATURES)},
    }.items():
        (tmp_path / relative).write_text(json.dumps(value))
    (tmp_path / "yolo/yolo11n.pt").write_bytes(b"detector stub; separately covered by real public smoke")
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path: object()))
    manifest = tmp_path / "manifest.json"
    assert main(["build-local-manifest", "--model-dir", str(tmp_path), "--output", str(manifest)]) == 0
    runtime = ModelRuntime(Settings(mode="models", model_dir=tmp_path, model_manifest=manifest))
    runtime.load()
    assert runtime.cnn_contract == contract(architecture)
    with torch.inference_mode():
        assert runtime.cnn(torch.zeros(1, 3, 128, 128)).shape == (1, 4)
    assert runtime.predict_risk(dict.fromkeys(MODEL_FEATURES, 0)) in {"low", "medium", "high"}


@pytest.mark.parametrize("update", [{"image_size": 32}, {"architecture": "simple"}, {"mean": [0, 0, 0]}, {"class_mapping": {"red": 0}}])
def test_incompatible_cnn_contract_is_rejected(update):
    with pytest.raises(ValueError):
        validate_contract({**contract(), **update})
