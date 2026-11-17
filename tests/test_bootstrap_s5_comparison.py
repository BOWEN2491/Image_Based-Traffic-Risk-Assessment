import json

from tools.bootstrap_s5_comparison import bootstrap


def test_bootstrap_is_group_paired_deterministic_and_never_promotes(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"source_status": "local-unverified",
        "promotion_allowed": False, "promotion_blockers": ["unverified source"],
        "configs": [{"name": "B0", "status": "complete", "metrics": {"map50_95": .5}},
                    {"name": "C1", "status": "complete", "metrics": {"map50_95": .5}}]}),
        encoding="utf-8")
    rows = []
    for index in range(4):
        truth = [{"category": "car", "bbox": [0, 0, 20, 20]},
                 {"category": "traffic light", "bbox": [30, 30, 40, 40]}]
        predictions = [{"category": obj["category"], "bbox": obj["bbox"], "confidence": .9}
                       for obj in truth]
        rows.append({"group_id": f"g{index}", "split": "development", "ground_truth": truth,
                     "objects": predictions, "error": None})
    for name in ("B0", "C1"):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    first = bootstrap(tmp_path, resamples=20, seed=42)
    second = bootstrap(tmp_path, resamples=20, seed=42)
    assert {key: value for key, value in first.items() if key != "runtime_seconds"} == {
        key: value for key, value in second.items() if key != "runtime_seconds"}
    assert first["comparisons"]["C1"]["map50_95"]["paired_group_bootstrap_95_ci"] is None
    assert first["comparisons"]["C1"]["map50_95"]["delta"] == 0
    assert first["comparisons"]["C1"]["traffic_light_recall"]["delta"] == 0
    assert first["comparisons"]["C1"]["traffic_light_recall"]["paired_group_bootstrap_95_ci"] == [0, 0]
    assert first["promotion_allowed"] is False
