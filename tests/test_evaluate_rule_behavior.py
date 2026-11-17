from __future__ import annotations

import json
import time

import pytest

from tools.evaluate_rule_behavior import _summarize, _transition_summary, evaluate, sha256_file
from traffic_risk.hard_rules import RULE_REASONS, _trace_hard_rules, apply_hard_rules, trace_hard_rules
from traffic_risk.schema import MODEL_FEATURES


def _features(**changes):
    row = {"central_tl_is_red": 0, "near_person_count": 0,
           "max_box_area_person": 0, "bottom_half_person_ratio": 0,
           "max_box_area_vehicle": 0, "near_vehicle_count": 0,
           "center_region_vehicle_ratio": 0, "sum_box_area_vehicle": 0,
           "object_density_per_mp": 0, "mean_overlap_iou": 0}
    row.update(changes)
    return row


def _prediction(sample_id="sample-1", *, features=None, risk="low", status="ok",
                reason_code="model_prediction", group_id="group-1", role="development",
                source_namespace="bdd100k"):
    return {"sample_id": sample_id, "source_namespace": source_namespace,
            "group_id": group_id, "role": role, "status": status, "risk": risk,
            "reason_code": reason_code, "features": features or _features()}


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _config(tmp_path, baseline, candidate, **changes):
    value = {"baseline_manifest": str(baseline), "candidate_manifest": str(candidate),
             "seed": 42, "max_rows": 100, "max_seconds": 30, "max_review_queue": 200}
    value.update(changes)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_trace_reports_all_overlaps_in_original_priority_and_apply_api_is_unchanged():
    features = _features(central_tl_is_red=1, near_person_count=1,
                         max_box_area_vehicle=0.2, near_vehicle_count=2)

    assert _trace_hard_rules(features) == [
        "central_red_signal", "near_person", "vehicle_extremely_close", "multiple_near_vehicles",
    ]
    assert trace_hard_rules(features) == {
        "matches": ["central_red_signal", "near_person", "vehicle_extremely_close", "multiple_near_vehicles"],
        "first_match": "central_red_signal",
    }
    assert set(_trace_hard_rules(_features())) == set()
    assert apply_hard_rules(features) == ("high", RULE_REASONS["central_red_signal"])
    assert apply_hard_rules(_features()) == (None, None)


def test_trace_preserves_inclusive_threshold_boundaries():
    assert _trace_hard_rules(_features(max_box_area_person=0.008,
                                       bottom_half_person_ratio=0.60)) == ["person_forward_zone"]
    assert _trace_hard_rules(_features(max_box_area_vehicle=0.16043790055171955)) == ["vehicle_extremely_close"]
    assert "center_blockage" in _trace_hard_rules(_features(center_region_vehicle_ratio=0.8,
                                                             sum_box_area_vehicle=0.2))


def test_summary_separates_normal_unknown_abstention_and_processing_failure():
    rows = [
        _prediction(risk="unknown", status="uncertain", reason_code="no_rule_triggered"),
        _prediction("sample-2", risk="unknown", status="uncertain", reason_code="perception_error"),
    ]

    summary = _summarize(rows)

    assert summary["risk_distribution"]["unknown"] == 2
    assert summary["normal_abstentions"] == {"no_rule_triggered": 1}
    assert summary["pipeline_failures"] == {"perception_error": 1}


def test_summary_does_not_trace_incomplete_feature_rows():
    complete = dict.fromkeys(MODEL_FEATURES, 0.0)
    complete["central_tl_is_red"] = 1
    rows = [
        _prediction(features=complete, risk="high", reason_code="hard_rule_triggered"),
        _prediction("partial", features={"central_tl_is_red": 1}),
    ]

    summary = _summarize(rows)

    assert summary["feature_complete_rows"] == 1
    assert summary["feature_incomplete_rows"] == 1
    assert summary["rule_hit_counts"]["central_red_signal"] == 1


def test_transition_table_is_namespaced_and_counts_missing_outputs():
    baseline = [_prediction("same"), _prediction("gone")]
    candidate = [_prediction("same", risk="high", reason_code="hard_rule_triggered"),
                 _prediction("same", source_namespace="other")]

    # Duplicate stable identities are rejected by the input reader, while the pure
    # helper still demonstrates that equal IDs from different sources do not pair.
    summary, changed = _transition_summary(baseline, candidate)

    assert summary["paired_sample_ids"] == 1
    assert summary["baseline_only"] == 1
    assert summary["candidate_only"] == 1
    assert summary["changed"] == 3
    assert {row["source_namespace"] for row in changed} == {"bdd100k", "other"}


def test_evaluator_is_deterministic_and_never_promotes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    baseline = _write_jsonl(tmp_path / "baseline.jsonl", [
        _prediction("a", features=_features(central_tl_is_red=1), risk="high",
                    reason_code="hard_rule_triggered"),
        _prediction("b", risk="unknown", status="uncertain", reason_code="perception_error"),
    ])
    candidate = _write_jsonl(tmp_path / "candidate.jsonl", [
        _prediction("a", features=_features(central_tl_is_red=1), risk="high",
                    reason_code="hard_rule_triggered"),
        _prediction("b", risk="unknown", status="uncertain", reason_code="no_rule_triggered"),
    ])
    config = _config(tmp_path, baseline, candidate)

    first = evaluate(config, tmp_path / "run-1")
    evaluate(config, tmp_path / "run-2")

    assert first["promotion_allowed"] is False
    assert first["output_transitions"]["changed"] == 1
    assert (tmp_path / "run-1/review_queue.jsonl").read_bytes() == (tmp_path / "run-2/review_queue.jsonl").read_bytes()
    assert (tmp_path / "run-1/report.json").read_bytes() == (tmp_path / "run-2/report.json").read_bytes()
    assert json.loads((tmp_path / "run-1/report.json").read_text())["review_queue"]["sha256"]


def test_evaluator_rejects_holdout_rows_missing_namespace_and_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for row, message in [
        (_prediction(role="holdout"), "reserved holdout"),
        ({**_prediction(), "source_namespace": ""}, "missing source_namespace"),
    ]:
        baseline = _write_jsonl(tmp_path / "baseline.jsonl", [row])
        candidate = _write_jsonl(tmp_path / "candidate.jsonl", [row])
        config = _config(tmp_path, baseline, candidate)
        with pytest.raises(ValueError, match=message):
            evaluate(config, tmp_path / "run")
        (tmp_path / "run").mkdir()
        with pytest.raises(FileExistsError):
            evaluate(config, tmp_path / "run")
        (tmp_path / "run").rmdir()


def test_evaluator_enforces_row_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = [_prediction("a"), _prediction("b", group_id="group-2")]
    baseline = _write_jsonl(tmp_path / "baseline.jsonl", rows)
    candidate = _write_jsonl(tmp_path / "candidate.jsonl", rows)
    config = _config(tmp_path, baseline, candidate, max_rows=1)

    with pytest.raises(ValueError, match="max_rows"):
        evaluate(config, tmp_path / "run")


def test_evaluator_enforces_manifest_byte_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    large = _prediction()
    large["reason"] = "x" * 256
    baseline = _write_jsonl(tmp_path / "baseline.jsonl", [large])
    candidate = _write_jsonl(tmp_path / "candidate.jsonl", [large])
    config = _config(tmp_path, baseline, candidate, max_input_bytes=100)

    with pytest.raises(ValueError, match="max_input_bytes"):
        evaluate(config, tmp_path / "run")


def test_chunked_hash_observes_expired_deadline(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"data")

    with pytest.raises(TimeoutError, match="hashing inputs"):
        sha256_file(path, time.monotonic() - 1)
