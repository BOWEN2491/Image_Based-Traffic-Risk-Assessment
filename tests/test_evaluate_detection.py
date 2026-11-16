import pytest
import json
from traffic_risk.evaluate_detection import average_precision, iou, match, metrics, validate_sample_splits, pipeline_failed


def test_caught_pipeline_errors_count_as_failures_but_abstentions_do_not():
    assert pipeline_failed({"status": "uncertain", "reason": "Perception could not produce a reliable result"})
    assert not pipeline_failed({"status": "uncertain", "reason": "No supported road users or signals were detected"})
    assert not pipeline_failed({"status": "ok", "reason": "rule"})


def obj(name="car", score=.9, box=(0, 0, 10, 10)):
    return {"category": name, "bbox": box, "confidence": score}


def test_iou_and_class_matching_are_one_to_one():
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0
    assert match([obj(score=.5), obj(score=.9), obj("person")], [obj()]) == {1: 0}


def test_ap_penalizes_ranked_false_positives_and_empty_predictions():
    assert average_precision([(.9, 0), (.8, 1)], 1) == pytest.approx(.5)
    assert average_precision([], 2) == 0
    assert average_precision([], 0) is None


def test_missed_ground_truth_and_confidence_cutoff():
    report = metrics([{"ground_truth": [obj(), obj("person")], "objects": [obj(score=.1)]}])
    assert report["per_class"]["car"]["ap50"] == 1
    assert report["per_class"]["car"]["recall_at_025_iou50"] == 0
    assert report["per_class"]["person"]["ap50"] == 0
    assert report["per_class"]["train"]["ap50"] is None
    assert report["map50"] == .5


def test_evaluation_writes_predictions_metrics_and_matched_crops(tmp_path, monkeypatch):
    import torch
    from PIL import Image
    from traffic_risk.evaluate_detection import evaluate, sha256

    class Box:
        cls = torch.tensor(0)
        conf = torch.tensor(.9)
        xyxy = torch.tensor([[1., 1., 20., 20.]])

    class Result:
        names = {0: "traffic light"}
        boxes = [Box()]

    class Detector:
        def __init__(self, path):
            self.fail = str(path).endswith("fail.pt")

        def predict(self, **kwargs):
            if self.fail and "conf" in kwargs:
                raise RuntimeError("injected detector failure")
            return [Result()]

    monkeypatch.setattr("ultralytics.YOLO", Detector)
    Image.new("RGB", (32, 32)).save(tmp_path / "image.png")
    (tmp_path / "label.json").write_text("{}")
    (tmp_path / "weights.pt").write_bytes(b"not loaded")
    sample = {"sample_id": "val/one", "image_id": "one", "source_split": "val", "split": "holdout",
              "group_id": "ride:one", "pixel_sha256": "identity-one",
              "image_path": "image.png", "label_path": "label.json",
              "image_sha256": sha256(tmp_path / "image.png"), "label_sha256": sha256(tmp_path / "label.json"),
              "objects": [{"object_id": 0, "category": "traffic light", "bbox": [1, 1, 20, 20], "color": "red"}]}
    (tmp_path / "images.jsonl").write_text(json.dumps(sample) + "\n")
    config = {"dataset_root": str(tmp_path), "images_manifest": str(tmp_path / "images.jsonl"),
              "output_dir": str(tmp_path / "out"), "models": {"baseline": str(tmp_path / "weights.pt"), "candidate": str(tmp_path / "weights.pt")}}
    report = evaluate(config)
    assert report["promotion_allowed"] is False
    assert report["models"]["baseline"]["splits"]["holdout"]["map50"] == 1
    assert report["comparison"]["map50_95_gain"] == 0
    crop = json.loads((tmp_path / "out/baseline/matched_traffic_lights.jsonl").read_text())
    assert (tmp_path / "out/baseline" / crop["crop_path"]).is_file()
    assert crop["color"] == "red"
    (tmp_path / "fail.pt").write_bytes(b"not loaded")
    config["models"] = {"failure": str(tmp_path / "fail.pt")}
    failed = evaluate(config)
    assert failed["models"]["failure"]["failure_rate"] == 1
    assert failed["models"]["failure"]["splits"]["holdout"]["map50"] == 0
    row = json.loads((tmp_path / "out/failure/predictions.jsonl").read_text())
    assert "injected detector failure" in row["error"]
    assert len(row["ground_truth"]) == 1
    missing = json.loads((tmp_path / "out/failure/matched_traffic_lights.jsonl").read_text())
    assert missing["crop_path"] is None
    assert missing["reason"] == "inference_failure"
    (tmp_path / "image.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        evaluate(config)


def test_evaluation_rejects_duplicates_and_split_leakage():
    sample = {"sample_id": "a", "split": "development", "group_id": "ride", "pixel_sha256": "pixel"}
    with pytest.raises(ValueError, match="Duplicate"):
        validate_sample_splits([sample, sample])
    with pytest.raises(ValueError, match="leakage"):
        validate_sample_splits([sample, {**sample, "sample_id": "b", "split": "holdout"}])
    with pytest.raises(ValueError, match="Missing"):
        validate_sample_splits([{**sample, "group_id": None}])
