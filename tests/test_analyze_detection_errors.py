import json

from PIL import Image

from tools.analyze_detection_errors import analyze, main


def test_failure_reasons_and_size_support_are_separated():
    manifest = [{"sample_id": "s1", "width": 100, "height": 100, "objects": [
        {"object_id": "a", "attributes": {"occluded": True, "truncated": False}},
        {"object_id": "b", "attributes": {}},
    ]}]
    predictions = [{"sample_id": "s1", "split": "development", "error": None,
        "ground_truth": [
            {"object_id": "a", "category": "traffic light", "bbox": [10, 10, 20, 20]},
            {"object_id": "b", "category": "car", "bbox": [40, 40, 80, 80]},
        ],
        "objects": [
            {"category": "traffic light", "bbox": [10, 10, 20, 20], "confidence": .9},
            {"category": "person", "bbox": [40, 40, 80, 80], "confidence": .8},
            {"category": "car", "bbox": [0, 0, 5, 5], "confidence": .7},
        ]}]
    report, cases = analyze(predictions, manifest)
    light = report["roles"]["development"]["ground_truth_by_class_and_size"]["traffic light"]["large"]
    assert light["gt_support"] == 1
    assert light["recall"] == 1
    assert light["occluded"] == {"True": 1}
    assert {row["failure"] for row in cases} == {"wrong_class", "false_positive"}


def test_processing_failure_remains_distinct_from_abstention():
    manifest = [{"sample_id": "s", "width": 100, "height": 100, "objects": []}]
    predictions = [{"sample_id": "s", "split": "holdout", "error": "RuntimeError: bad",
                    "ground_truth": [{"object_id": "x", "category": "car", "bbox": [0, 0, 50, 50]}],
                    "objects": []}]
    report, cases = analyze(predictions, manifest)
    large = report["roles"]["holdout"]["ground_truth_by_class_and_size"]["car"]["large"]
    assert large["processing_failure"] == 1
    assert large["recall"] == 0
    assert cases[0]["failure"] == "processing_failure"


def test_local_atlas_is_bounded_and_annotates_manifest_images(tmp_path, monkeypatch):
    image_root = tmp_path / "dataset"
    image_root.mkdir()
    Image.new("RGB", (64, 48), "white").save(image_root / "frame.png")
    manifest_path = tmp_path / "images.jsonl"
    manifest_path.write_text(json.dumps({"sample_id": "s", "image_path": "frame.png",
        "width": 64, "height": 48, "objects": []}) + "\n", encoding="utf-8")
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(json.dumps({"sample_id": "s", "split": "development", "error": None,
        "ground_truth": [{"object_id": "a", "category": "car", "bbox": [5, 5, 30, 30]}],
        "objects": []}) + "\n", encoding="utf-8")
    output = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["analyze", "--predictions", str(predictions_path),
        "--manifest", str(manifest_path), "--output", str(output), "--image-root", str(image_root), "--atlas"])
    assert main() == 0
    index = json.loads((output / "atlas_index.json").read_text(encoding="utf-8"))
    assert len(index) == 1
    assert (output / index[0]["file"]).is_file()
