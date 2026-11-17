from __future__ import annotations

import json

from tools.profile_pipeline import (
    _ACTIVE_RECORD,
    _Probe,
    _gather_cancel_on_error,
    _profile_completeness,
    _stage,
    latency_summary,
    PROFILED_STAGES,
    select_performance_samples,
)


def _row(sample_id: str, group_id: str, *, split: str = "development", source: str = "val") -> dict:
    suffix = sample_id.rsplit("-", 1)[-1]
    category = [{"category": "person"}] if suffix.isdigit() and int(suffix) % 2 else []
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "split": split,
        "source_split": source,
        "width": 1280,
        "height": 720,
        "objects": category,
        "frame_attributes": {"timeofday": "daytime", "scene": "city street"},
    }


def test_performance_sample_selection_is_repeatable_and_development_only():
    rows = [_row(f"sample-{index}", f"group-{index}") for index in range(20)]
    rows.extend([
        _row("duplicate-group", "group-0"),
        _row("holdout", "holdout-group", split="holdout"),
        _row("train", "train-group", source="train"),
    ])

    first = select_performance_samples(rows, 12, seed=42)
    second = select_performance_samples(rows, 12, seed=42)

    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert len(first) == 12
    assert len({row["group_id"] for row in first}) == 12
    assert all(row["split"] == "development" and row["source_split"] == "val" for row in first)


def test_sample_selection_reports_shortfall_by_returning_all_unique_groups():
    rows = [_row("only", "one"), _row("same-group", "one")]

    selected = select_performance_samples(rows, 5)

    assert [row["sample_id"] for row in selected] == ["only"]


def test_latency_percentiles_interpolate_and_empty_is_explicit():
    assert latency_summary([1.0, 2.0, 3.0, 4.0]) == {
        "count": 4,
        "p50_ms": 2.5,
        "p95_ms": 3.8499999999999996,
    }
    assert latency_summary([]) == {"count": 0, "p50_ms": None, "p95_ms": None}


def test_stage_timer_records_a_nonnegative_duration():
    record = {"stages_ms": {}}
    token = _ACTIVE_RECORD.set(record)
    try:
        with _stage("controlled_stage"):
            pass
    finally:
        _ACTIVE_RECORD.reset(token)

    assert len(record["stages_ms"]["controlled_stage"]) == 1
    assert record["stages_ms"]["controlled_stage"][0] >= 0


def _complete_profile_rows():
    rows = []
    stage_status = json.dumps({stage: "measured" for stage in PROFILED_STAGES})
    for round_index in range(2):
        for sample_index in range(2):
            rows.append({"kind": "pipeline", "round": round_index, "sample_index": sample_index,
                         "sample_id": f"sample-{sample_index}", "pipeline_failure": False,
                         "stage_status_json": stage_status})
    rows.extend({"kind": "api", "phase": "warmup", "request_index": index,
                 "sample_id": f"sample-{index % 2}", "stage_status_json": stage_status,
                 "http_status": 200, "pipeline_failure": False} for index in range(1))
    for round_index in range(2):
        for sample_index in range(2):
            rows.append({"kind": "api", "phase": "sequential", "round": round_index,
                         "sample_id": f"sample-{sample_index}", "request_index": sample_index,
                         "stage_status_json": stage_status,
                         "http_status": 200, "pipeline_failure": False})
    for clients in (1, 2, 4):
        for index in range(clients * 2):
            rows.append({"kind": "api", "phase": f"concurrency-{clients}", "request_index": index,
                         "sample_id": f"sample-{index % 2}", "stage_status_json": stage_status,
                         "http_status": 200, "pipeline_failure": False})
    return rows


def test_profiler_completeness_uses_each_phase_denominator_and_failure_reason():
    config = {"rounds": 2, "warmups": 1, "requests_per_client": 2}
    rows = _complete_profile_rows()
    assert _profile_completeness(rows, config, sample_count=2)["complete"]
    rows[0].pop("stage_status_json")
    assert _profile_completeness(rows, config, sample_count=2)["probe_coverage_complete"] is False

    # Replacing a missing task with a duplicate keeps the total row count unchanged.
    rows.remove(next(row for row in rows if row.get("sample_id") == "sample-1" and row.get("round") == 1))
    rows.append({"kind": "pipeline", "round": 1, "sample_index": 0,
                 "sample_id": "sample-0", "pipeline_failure": False})
    result = _profile_completeness(rows, config, sample_count=2)
    assert result["pipeline"]["complete"] is False
    assert result["complete"] is False

    rows = _complete_profile_rows()
    rows[0]["pipeline_failure"] = True
    assert _profile_completeness(rows, config, sample_count=2)["complete"] is False

    rows = _complete_profile_rows()
    rows.remove(next(row for row in rows if row.get("phase") == "concurrency-2"))
    assert _profile_completeness(rows, config, sample_count=2)["api"]["complete"] is False
    rows = _complete_profile_rows()
    next(row for row in rows if row.get("phase") == "sequential")["pipeline_failure"] = True
    assert _profile_completeness(rows, config, sample_count=2)["api"]["complete"] is False


def test_parallel_timeout_cancels_and_joins_outstanding_workers():
    import asyncio

    async def exercise():
        cancelled = asyncio.Event()

        async def waiting():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def timeout():
            await asyncio.sleep(0)
            raise TimeoutError("controlled timeout")

        tasks = [asyncio.create_task(waiting()), asyncio.create_task(timeout())]
        cancelled_count = await _gather_cancel_on_error(tasks)
        assert cancelled_count == 1
        assert cancelled.is_set()
        assert all(task.done() for task in tasks)

    asyncio.run(exercise())


def test_probe_finalization_marks_missing_and_not_applicable_without_fake_zeroes():
    record = {"stages_ms": {}, "stage_calls": {}, "detected_object_count": 0}
    _Probe.finalize_record(record, include_detections=False)

    assert record["stage_status"]["feature_build_ms"] == "not_applicable"
    assert record["stage_status"]["hard_rules_ms"] == "not_applicable"
    assert record["stage_status"]["overlay_details_ms"] == "not_applicable"
    assert record["stage_status"]["detector_postprocess_nms_ms"] == "missing"
    assert "detector_postprocess_nms_ms" not in record["stages_ms"]


def test_probe_wrapper_restores_original_even_after_exception_without_stacking():
    class Target:
        @staticmethod
        def call():
            raise RuntimeError("controlled")

    original = Target.call
    probe = _Probe()
    probe.patch_sync(Target, "call", "controlled_ms")
    record = {"stages_ms": {}}
    token = _ACTIVE_RECORD.set(record)
    try:
        try:
            Target.call()
        except RuntimeError:
            pass
    finally:
        _ACTIVE_RECORD.reset(token)
        probe.restore()

    assert Target.call is original
    assert len(record["stages_ms"]["controlled_ms"]) == 1
    assert record["stage_calls"]["controlled_ms"] == 1


def test_probe_measures_legacy_path_and_restores_the_call_site(tmp_path):
    from types import SimpleNamespace

    import cv2
    import numpy as np
    from traffic_risk import pipeline

    image_path = tmp_path / "legacy.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((80, 120, 3), dtype=np.uint8))
    original_run_detected = pipeline._run_detected

    class LegacyRuntime:
        versions = {}
        settings = SimpleNamespace(mode="rules")

        def detect(self, _path):
            return {"frames": [{"objects": [{
                "id": 0, "category": "car", "confidence": 0.9,
                "attributes": {}, "box2d": {"x1": 10, "y1": 20, "x2": 90, "y2": 70},
            }]}]}

    probe = _Probe()
    record = {"stages_ms": {}, "opencv_imread_calls": 0, "pillow_open_calls": 0}
    probe.install()
    token = _ACTIVE_RECORD.set(record)
    try:
        result = pipeline.run_single_image(image_path, LegacyRuntime(), include_detections=True)
    finally:
        _ACTIVE_RECORD.reset(token)
        probe.restore()
    _Probe.finalize_record(record, include_detections=True)

    assert result["status"] in {"ok", "uncertain"}
    assert pipeline._run_detected is original_run_detected
    assert record["opencv_imread_calls"] == 1
    assert record["stage_status"]["legacy_runtime_detect_ms"] == "measured"
    assert record["stage_status"]["opencv_imread_ms"] == "measured"
    assert record["stage_status"]["image_decode_ms"] == "not_applicable"
    assert record["stage_status"]["detector_detect_image_ms"] == "not_applicable"
    assert record["stage_status"]["feature_build_ms"] == "measured"
    assert record["stage_status"]["hard_rules_ms"] == "measured"
    assert record["stage_status"]["overlay_details_ms"] == "measured"
