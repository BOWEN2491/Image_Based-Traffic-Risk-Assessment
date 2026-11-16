import json
import random
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from traffic_risk.prepare_evaluation import COLORS, group_split, prepare


def sample(root, split, name, ride, *, color="red", duplicate=None, bad_box=False):
    image_dir = root / "bdd100_images_100k/100k" / split
    label_dir = root / "bdd100k_labels/100k" / split
    info_dir = root / "bdd100k_info/100k" / split
    for directory in (image_dir, label_dir, info_dir):
        directory.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{name}.jpg"
    if duplicate:
        image_path.write_bytes(Path(duplicate).read_bytes())
    else:
        values = random.Random(name).randbytes(24 * 24 * 3)
        Image.frombytes("RGB", (24, 24), values).save(image_path)
    (label_dir / f"{name}.json").write_text(json.dumps({"name": name, "frames": [{"objects": [
        {"id": 0, "category": "traffic light", "attributes": {"trafficLightColor": color},
         "box2d": {"x1": 1, "y1": 1, "x2": 99 if bad_box else 20, "y2": 20}}
    ]}]}))
    (info_dir / f"{name}.json").write_text(json.dumps({"rideID": ride}))
    return image_path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_deterministic_grouped_manifests_and_crops(tmp_path):
    root = tmp_path / "source"
    for i in range(8):
        sample(root, "val", f"v{i}", f"vride{i // 2}", color=COLORS[i % 4])
    for i in range(24):
        sample(root, "train", f"t{i}", f"tride{i // 2}", color=COLORS[i % 4])
    first, second = tmp_path / "first", tmp_path / "second"
    result = prepare(root, first, max_images=8, max_crops=16)
    prepare(root, second, max_images=8, max_crops=16)
    assert result["promotion_allowed"] is False
    assert result["training_class_counts"] == dict.fromkeys(COLORS, 4)
    for filename in ("images.jsonl", "crops.jsonl", "summary.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    images, crops = rows(first / "images.jsonl"), rows(first / "crops.jsonl")
    by_group = {}
    for row in images + crops:
        by_group.setdefault(row["group_id"], set()).add(row["split"])
        assert row["provenance"] == "local-unverified"
        assert not row["promotion_allowed"]
    assert all(len(splits) == 1 for splits in by_group.values())
    assert len({r["pixel_sha256"] for r in images}) == len(images)
    assert len({r["pixel_sha256"] for r in crops}) == len(crops)
    assert all((first / row["path"]).is_file() for row in crops)
    assert all(row["source_split"] == "val" for row in crops if row["split"] == "holdout")


def test_cross_split_group_and_content_excluded(tmp_path):
    root = tmp_path / "source"
    original = sample(root, "val", "original", "shared")
    sample(root, "train", "same-group", "shared")
    sample(root, "train", "same-pixels", "different", duplicate=original)
    sample(root, "train", "valid", "unique", color="none")
    output = tmp_path / "out"
    prepare(root, output, max_images=1, max_crops=4)
    exclusions = rows(output / "excluded.jsonl")
    assert {r["reason"] for r in exclusions} >= {"group reserved for evaluation", "duplicate image pixels"}
    crop = next(r for r in rows(output / "crops.jsonl") if r["source_split"] == "train")
    assert crop["label"] == "unknown"


def test_invalid_boxes_and_names_are_not_ground_truth(tmp_path):
    root = tmp_path / "source"
    sample(root, "val", "bad", "a", bad_box=True)
    sample(root, "val", "wrong-name", "b")
    path = root / "bdd100k_labels/100k/val/wrong-name.json"
    payload = json.loads(path.read_text())
    payload["name"] = "different-image"
    path.write_text(json.dumps(payload))
    sample(root, "train", "train", "c")
    output = tmp_path / "out"
    result = prepare(root, output, max_images=2, max_crops=4)
    assert result["image_counts"].get("holdout", 0) == 0
    assert result["image_counts"].get("development", 0) == 0
    assert len(rows(output / "excluded.jsonl")) == 2


def test_refuse_source_mutation_overwrite_and_unbounded_runs(tmp_path):
    root = tmp_path / "source"
    sample(root, "val", "v", "v")
    sample(root, "train", "t", "t")
    with pytest.raises(ValueError, match="outside"):
        prepare(root, root / "out")
    with pytest.raises(ValueError, match="1..1000"):
        prepare(root, tmp_path / "out", max_images=1001)
    output = tmp_path / "out"
    prepare(root, output, max_images=1, max_crops=4)
    with pytest.raises(FileExistsError):
        prepare(root, output)


def test_group_assignment_is_stable_and_has_both_roles():
    assert {group_split(f"ride:{i}", "val", 42) for i in range(100)} == {"development", "holdout"}
    assert {group_split(f"ride:{i}", "train", 42) for i in range(100)} == {"train", "train-validation"}


def test_scan_limit_saves_shortfalls_without_fabricating_balance(tmp_path):
    root = tmp_path / "source"
    sample(root, "val", "v", "v")
    for index in range(8):
        sample(root, "train", f"t{index}", f"t{index}", color="red")
    output = tmp_path / "out"
    result = prepare(root, output, max_images=1, max_crops=20, max_train_images=2)
    assert result["scanned_counts"]["train"] == 2
    assert result["stop_reason"] == "training scan limit reached"
    assert result["training_class_shortfalls"]["yellow"] == 5
    assert len([row for row in rows(output / "crops.jsonl") if row["source_split"] == "train"]) == 2


def test_time_budget_saves_partial_manifests(tmp_path, monkeypatch):
    root = tmp_path / "source"
    sample(root, "val", "v", "v")
    sample(root, "train", "t", "t")
    ticks = iter([0, 2, 3])
    monkeypatch.setattr("traffic_risk.prepare_evaluation.time.monotonic", lambda: next(ticks))
    output = tmp_path / "out"
    result = prepare(root, output, max_seconds=1)
    assert result["stop_reason"] == "preparation time budget reached"
    assert result["image_counts"] == {}
    assert (output / "summary.json").is_file()


@pytest.mark.parametrize("command,module", [
    ("prepare-evaluation", "prepare_evaluation"),
    ("evaluate-detection", "evaluate_detection"),
    ("cnn-experiment", "cnn_experiment"),
])
def test_cli_forwards_arguments_and_failure_codes(monkeypatch, command, module):
    from traffic_risk.cli import main
    original = sys.argv
    received = []

    def workflow():
        received.extend(sys.argv)
        return 3

    monkeypatch.setitem(sys.modules, f"traffic_risk.{module}", SimpleNamespace(main=workflow))
    assert main([command, "--output", "result"]) == 3
    assert received == [f"traffic-risk {command}", "--output", "result"]
    assert sys.argv is original
