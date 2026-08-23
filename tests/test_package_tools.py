from __future__ import annotations

import csv
import json
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from traffic_risk.final_predict import predict_with_combined
from traffic_risk.merge_review_back import copy_to, index_target, iter_review_items, move_to
from traffic_risk.review_roi_labeler import imread_resize, load_items, move_to as review_move, save_manifest
from traffic_risk.schema import MODEL_FEATURES


def _features(**updates):
    values = dict.fromkeys(MODEL_FEATURES, 0.0)
    values.update(updates)
    return values


def test_final_predict_applies_hard_rule_before_model():
    class NeverCalled:
        def predict_risk(self, _features):
            raise AssertionError("model must not run after hard rule")

    risk, reason = predict_with_combined(_features(n_human=3, near_person_count=3), NeverCalled())
    assert risk == "high"
    assert "pedestrian" in reason


def test_final_predict_uses_model_when_no_hard_rule():
    class Predictor:
        def predict_risk(self, features):
            assert features["n_human"] == 0.0
            return "low"

    risk, reason = predict_with_combined(_features(), Predictor())
    assert risk == "low"
    assert "model" in reason.lower()


def test_review_inventory_and_atomic_manifest(tmp_path):
    root = tmp_path / "roi"
    (root / "red").mkdir(parents=True)
    (root / "unknown").mkdir()
    (root / "red" / "a.png").write_bytes(b"x")
    (root / "unknown" / "b.jpg").write_bytes(b"y")
    assert [p.name for p, _ in load_items(root, None)] == ["a.png", "b.jpg"]
    assert [p.name for p, _ in load_items(root, "unknown")] == ["b.jpg"]
    destination = root / "green"
    moved = review_move(destination, root / "red" / "a.png")
    assert moved == destination / "a.png"
    (destination / "b.jpg").write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        review_move(destination, root / "unknown" / "b.jpg")
    manifest = root / "review.csv"
    save_manifest(root, [(moved, "green"), (root / "unknown" / "b.jpg", "unknown")], manifest)
    with manifest.open(newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle))[0] == ["path", "label"]
    assert not manifest.with_suffix(".csv.tmp").exists()


def test_review_bad_image_returns_placeholder(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    image = imread_resize(bad)
    assert image.shape == (320, 320, 3)


def test_merge_inventory_and_collision_safe_copy(tmp_path):
    target = tmp_path / "target"
    review = tmp_path / "review"
    (target / "red").mkdir(parents=True)
    (target / "green").mkdir()
    (review / "green").mkdir(parents=True)
    existing = target / "red" / "same.png"
    existing.write_bytes(b"same")
    reviewed = review / "green" / "same.png"
    reviewed.write_bytes(b"new")
    assert index_target(target)["same.png"][1] == "red"
    assert list(iter_review_items(review)) == [(reviewed, "green")]
    with pytest.raises(FileExistsError):
        copy_to(target / "red", reviewed)
    moved = move_to(target / "green", existing)
    assert moved.exists() and not existing.exists()


def test_merge_main_dry_run_and_copy_missing(tmp_path, monkeypatch):
    from traffic_risk import merge_review_back

    target = tmp_path / "target"
    review = tmp_path / "review"
    (target / "red").mkdir(parents=True)
    (review / "green").mkdir(parents=True)
    (target / "red" / "move.png").write_bytes(b"x")
    (review / "green" / "move.png").write_bytes(b"x")
    (review / "green" / "new.png").write_bytes(b"y")
    monkeypatch.setattr(sys, "argv", ["merge-review", "--review-root", str(review), "--target-root", str(target), "--copy-missing"])
    merge_review_back.main()
    assert (target / "green" / "move.png").exists()
    assert (target / "green" / "new.png").exists()
    assert (target / "merge_log.csv").exists()


def test_yolo_category_mapping_and_invalid_roi(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=object))
    from traffic_risk import yolo_to_bddjson as converter

    assert converter.map_to_bdd_category("traffic_light") == "traffic light"
    assert converter.map_to_bdd_category("unknown-object") == ""
    assert converter.map_to_bdd_category("person") == "person"
    assert converter.map_to_bdd_category("bicycle") == "bicycle"
    assert converter.map_to_bdd_category("motorcycle") == "motorcycle"
    assert converter.map_to_bdd_category("cyclist") == "rider"
    assert converter.map_to_bdd_category("car") == "car"
    assert converter.map_to_bdd_category("traffic sign") == "traffic sign"
    converter._TL_CNN_CTX = {"unused": True}
    assert converter.classify_tl_color(np.zeros((4, 4, 3), dtype=np.uint8), (0, 0, 2, 2), min_side=6) == "unknown"


def test_yolo_cnn_loader_reports_missing_model(tmp_path):
    from traffic_risk.yolo_to_bddjson import _init_tl_cnn

    with pytest.raises(FileNotFoundError, match="Model not found"):
        _init_tl_cnn(tmp_path)


def test_feature_count_command(tmp_path, monkeypatch, capsys):
    source = tmp_path / "features.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["temp_feature_count", str(source)])
    runpy.run_module("traffic_risk.temp_feature_count", run_name="__main__")
    assert capsys.readouterr().out.strip() == "1"


def test_build_local_manifest_command_hashes_all_assets(tmp_path):
    from traffic_risk.cli import main

    model_dir = tmp_path / "models"
    for relative in ("yolo/yolo11n.pt", "cnn/best_model.pth", "cnn/class_indices.json", "risk/risk_xgb.ubj", "risk/feature_order.json", "risk/model_metadata.json"):
        path = model_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    output = tmp_path / "manifest.json"
    assert main(["build-local-manifest", "--model-dir", str(model_dir), "--output", str(output)]) == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "2.0.0"
    assert len(manifest["assets"]) == 6


def test_standalone_traffic_light_loader_contract(monkeypatch, tmp_path):
    from traffic_risk import predict_tl_color

    class Model:
        def __init__(self):
            self.fc = SimpleNamespace(in_features=4)

        def load_state_dict(self, state, strict):
            assert state == {"weight": 1} and strict is True

        def eval(self):
            return self

        def to(self, device):
            assert device == "cpu"
            return self

    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(Linear=lambda *_args: object()),
        load=lambda *_args, **_kwargs: {"weight": 1},
    )
    fake_torchvision = SimpleNamespace(models=SimpleNamespace(resnet18=lambda weights=None: Model()))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)
    model_path = tmp_path / "model.pth"
    mapping_path = tmp_path / "mapping.json"
    model_path.write_bytes(b"weights")
    mapping_path.write_text(json.dumps({"red": 0, "green": 1}), encoding="utf-8")
    model, names = predict_tl_color.load_model(model_path, mapping_path)
    assert names == {0: "red", 1: "green"}
    assert model is not None


def test_weak_labels_command_writes_output(tmp_path, monkeypatch):
    from traffic_risk import weak_labels_run

    source = tmp_path / "features.csv"
    output = tmp_path / "out.csv"
    frame = {
        "n_person": [0, 1, 2],
        "n_vehicle": [0, 1, 2],
        "n_rider": [0, 0, 1],
        "n_ts": [0, 1, 2],
        "max_box_area_person": [0.0, 0.1, 0.2],
        "max_box_area_vehicle": [0.0, 0.1, 0.2],
        "sum_box_area_person": [0.0, 0.1, 0.2],
        "sum_box_area_vehicle": [0.0, 0.1, 0.2],
        "near_person_count": [0, 1, 2],
        "near_vehicle_count": [0, 1, 2],
        "object_density_per_mp": [0.0, 1.0, 2.0],
        "mean_overlap_iou": [0.0, 0.1, 0.2],
        "bottom_half_person_ratio": [0.0, 0.1, 0.2],
        "center_region_vehicle_ratio": [0.0, 0.1, 0.2],
    }
    import pandas as pd

    pd.DataFrame(frame).to_csv(source, index=False)
    monkeypatch.setattr(sys, "argv", ["generate-weak-labels", "--features", str(source), "--output", str(output)])
    weak_labels_run.main()
    assert output.exists()
