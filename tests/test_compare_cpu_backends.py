import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

from tools.compare_cpu_backends import (
    BACKENDS,
    SIZES,
    _api_completeness,
    _round_stability,
    _stats,
    _parity,
    _performance_gates,
    adoption_completeness,
    build_concurrency_schedule,
    build_measurement_schedule,
    validate_measurement_schedule,
    compare,
)


def _write_parity(path, response):
    path.mkdir(parents=True)
    row = {"sample_id": "s1", "include_detections": True, "response": response, "objects": None}
    (path / "parity_outputs.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_response_field_difference_fails_parity(tmp_path):
    response = {"status": "ok", "risk": "low", "reason_code": "clear", "reason": "ok",
                "features": {"a": 1}, "image_size": [1, 1], "detections": None}
    outputs = {}
    for size in SIZES:
        torch_dir = tmp_path / f"torch-{size}"
        _write_parity(torch_dir, response)
        outputs[f"torch-{size}"] = torch_dir
        for backend in ("openvino", "onnx"):
            candidate_dir = tmp_path / f"{backend}-{size}"
            changed = {**response, "reason": "changed"} if backend == "onnx" else response
            _write_parity(candidate_dir, changed)
            outputs[f"{backend}-{size}"] = candidate_dir
    result = _parity(outputs)
    assert result["gates"]["onnx-640"]["status"] == "fail"
    assert result["gates"]["onnx-640"]["response_field_mismatch_count"] == 1


def test_adoption_requires_all_six_complete_and_measurement_counts(tmp_path):
    required = {task["task_id"] for task in build_measurement_schedule() + build_concurrency_schedule()}
    outputs, statuses = {}, {key: {"status": "complete"} for key in required}
    for backend in BACKENDS:
        for size in SIZES:
            key = f"{backend}-{size}"
            folder = tmp_path / key
            folder.mkdir()
            (folder / "summary.json").write_text(json.dumps({"completeness": {
                "pipeline_600_per_mode": True, "sequential_api_600_complete_per_mode": True,
                "sequential_api_600_success_per_mode": True, "concurrency_complete": True,
                "api_complete": True, "round_outputs_complete": True, "all_tasks_complete": True},
                "api_completeness": {"api_acceptable": True}}), encoding="utf-8")
            outputs[key] = folder
    assert adoption_completeness(outputs, statuses)
    statuses.pop("concurrency-onnx-960")
    assert not adoption_completeness(outputs, statuses)
    statuses["concurrency-onnx-960"] = {"status": "complete"}
    (outputs["onnx-960"] / "summary.json").write_text(json.dumps({"completeness": {
        "pipeline_600_per_mode": True, "sequential_api_600_complete_per_mode": True,
        "sequential_api_600_success_per_mode": False, "concurrency_complete": True,
        "api_complete": True, "round_outputs_complete": True, "all_tasks_complete": True},
        "api_completeness": {"api_acceptable": False}}), encoding="utf-8")
    assert not adoption_completeness(outputs, statuses)


def test_api_busy_503_increase_blocks_failure_gate(tmp_path):
    outputs = {}
    for size in SIZES:
        for backend in BACKENDS:
            folder = tmp_path / f"{backend}-{size}"
            folder.mkdir()
            api = {"p50_ms": 1, "p95_ms": 1, "median_round_p95_ms": 1,
                   "failures": 0, "busy_503": 0, "other_failure": 0}
            summary = {"pipeline_failures": 0, "peak_process_rss_bytes": 100,
                       "pipeline_false": dict(api), "pipeline_true": dict(api),
                       "api_false": dict(api), "api_true": dict(api)}
            if backend == "torch":
                summary["api_false"]["busy_503"] = 1
            elif backend == "openvino":
                summary["api_false"]["busy_503"] = 2
            (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            outputs[f"{backend}-{size}"] = folder
    gate = _performance_gates(outputs)["openvino-640"]
    assert gate["no_failure_increase"] is False


def _complete_api_rows_for_one_config(config, samples):
    rows = []
    for task in build_measurement_schedule():
        if task["config_id"] != "torch-640":
            continue
        for include in task["mode_order"]:
            rows.extend({"task_id": task["task_id"], "task_kind": task["task_kind"],
                         "global_round": task["global_round"], "config_id": task["config_id"],
                         "phase": "warmup", "request_index": index,
                         "include_detections": include, "http_status": 200}
                        for index in range(config["warmups"]))
            rows.extend({"task_id": task["task_id"], "task_kind": task["task_kind"],
                         "global_round": task["global_round"], "config_id": task["config_id"],
                         "phase": "sequential",
                         "sample_id": sample_id, "include_detections": include, "http_status": 200}
                        for sample_id in samples)
    for task in build_concurrency_schedule():
        if task["config_id"] != "torch-640":
            continue
        for include in task["mode_order"]:
            rows.extend({"task_id": task["task_id"], "task_kind": task["task_kind"],
                         "global_round": task["global_round"], "config_id": task["config_id"],
                         "phase": "warmup", "request_index": index,
                         "include_detections": include, "http_status": 200}
                        for index in range(config["warmups"]))
            for clients in (1, 2, 4):
                rows.extend({"task_id": task["task_id"], "task_kind": task["task_kind"],
                             "global_round": task["global_round"], "config_id": task["config_id"],
                             "phase": f"concurrency-{clients}",
                             "request_index": index, "include_detections": include, "http_status": 503}
                            for index in range(clients * config["requests_per_client"]))

    return rows


def test_api_completeness_counts_warmups_sequential_and_concurrency_per_mode():
    config = {"warmups": 1, "rounds": 3, "requests_per_client": 2}
    samples = ["s0", "s1"]
    rows = _complete_api_rows_for_one_config(config, samples)
    complete = _api_completeness(rows, config, sample_count=len(samples), sample_ids=samples)
    assert complete["api_expected_rows"] == len(rows)
    assert complete["api_expected_rows"] == 48
    assert complete["api_expected_warmup_rows"] == 8
    assert complete["api_expected_sequential_rows"] == 12
    assert complete["api_expected_concurrency_rows"] == 28
    assert complete["sequential_api_600_success_per_mode"] is True
    assert complete["api_complete"] is True
    assert _api_completeness(rows[:-1], config, len(samples), samples)["api_complete"] is False
    sequential_row = next(row for row in rows if row["phase"] == "sequential")
    sequential_row["sample_id"] = "unknown-sample"
    assert _api_completeness(rows, config, len(samples), samples)["api_complete"] is False
    rows = _complete_api_rows_for_one_config(config, samples)
    next(row for row in rows if row["phase"] == "sequential")["pipeline_failure"] = True
    assert _api_completeness(rows, config, len(samples), samples)["api_complete"] is True
    assert _api_completeness(rows, config, len(samples), samples)["api_acceptable"] is False
    rows[0]["task_id"] = "different-task"
    assert _api_completeness(rows, config, len(samples), samples)["task_ids_valid"] is False


def test_measurement_schedule_is_fixed_round_robin_and_position_balanced():
    first = build_measurement_schedule()
    second = build_measurement_schedule()
    assert first == second
    assert len(first) == 18
    assert validate_measurement_schedule(first)
    assert first[0]["task_id"] == "r0-640-torch"
    assert first[3]["task_id"] == "r0-960-torch"
    assert first[6]["task_id"] == "r1-960-openvino"
    assert first[12]["task_id"] == "r2-640-onnx"
    assert not validate_measurement_schedule(first[:-1])


def test_formal_statistics_exclude_warmups_and_concurrency():
    rows = [
        {"kind": "pipeline", "include_detections": False, "phase": "warmup", "elapsed_ms": 10000},
        {"kind": "pipeline", "include_detections": False, "phase": "measurement", "global_round": 0, "elapsed_ms": 10},
        {"kind": "pipeline", "include_detections": False, "phase": "measurement", "global_round": 0, "elapsed_ms": 20},
        {"kind": "api", "include_detections": False, "phase": "warmup", "elapsed_ms": 10000, "http_status": 200},
        {"kind": "api", "include_detections": False, "phase": "sequential", "global_round": 0, "elapsed_ms": 10, "http_status": 200},
        {"kind": "api", "include_detections": False, "phase": "concurrency-4", "elapsed_ms": 50000, "http_status": 503},
    ]
    pipeline = _stats(rows, False, "pipeline")
    api = _stats(rows, False, "api")
    assert pipeline["count"] == 2
    assert pipeline["p95_ms"] < 20
    assert api["count"] == 1
    assert api["p95_ms"] == 10
    assert api["busy_503"] == 1


def test_round_stability_requires_exact_response_and_detector_consistency(tmp_path):
    config_id = "torch-640"
    folder = tmp_path / config_id
    folder.mkdir()
    rows = []
    response = {"status": "uncertain", "risk": "unknown", "reason_code": "no_supported_detections",
                "reason": "No supported road users or signals were detected", "features": None, "image_size": None}
    for round_index in range(3):
        for sample_index in range(100):
            for include in (False, True):
                rows.append({"global_round": round_index, "sample_id": f"s{sample_index}",
                             "include_detections": include, "response": response, "objects": []})
    (folder / "round_outputs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = _round_stability({config_id: folder})
    assert result["status"] == "fail"  # one configuration cannot certify the six-config matrix
    assert result["configurations"][config_id]["status"] == "pass"
    rows[-1]["response"] = {**response, "features": {"changed": 1}}
    (folder / "round_outputs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert _round_stability({config_id: folder})["configurations"][config_id]["status"] == "fail"


def test_compare_launches_frozen_round_jobs_before_concurrency_jobs(tmp_path, monkeypatch):
    import tools.compare_cpu_backends as module

    input_root = tmp_path / "input"
    data_root = tmp_path / "dataset"
    input_root.mkdir()
    data_root.mkdir()
    model_path = input_root / "model.pt"
    model_path.write_bytes(b"checkpoint")
    performance_manifest = input_root / "manifest.jsonl"
    performance_manifest.write_text("frozen", encoding="utf-8")
    artifact_paths = {}
    artifact_hashes = {}
    for backend in ("openvino", "onnx"):
        for size in SIZES:
            key = f"{backend}-{size}"
            path = input_root / f"{key}.json"
            path.write_text(key, encoding="utf-8")
            artifact_paths[key] = str(path)
            artifact_hashes[key] = hashlib.sha256(key.encode()).hexdigest()
    config = {
        "dataset_root": str(data_root), "performance_manifest": str(performance_manifest),
        "model_path": str(model_path), "model_sha256": "source-sha",
        "backend_artifacts": artifact_paths, "seed": 42, "threads": 2, "warmups": 5,
        "rounds": 3, "requests_per_client": 20, "budget_per_config_seconds": 900,
        "time_budget_seconds": 5400, "busy_timeout_seconds": 0.1,
    }
    sample_rows = [{"sample_id": f"s{index}", "image_path": f"{index}.jpg"} for index in range(100)]
    monkeypatch.setattr(module, "_read_config", lambda _path, _root: (dict(config), "config-sha"))
    monkeypatch.setattr(module, "_read_samples", lambda _config: (sample_rows, "manifest-sha"))
    monkeypatch.setattr(module, "verify_artifact_manifest",
                        lambda path, **_kwargs: {"manifest_sha256": artifact_hashes[
                            f"{path.stem.rsplit('-', 1)[0]}-{path.stem.rsplit('-', 1)[1]}"]})
    launched = []

    def fake_run(command, **_kwargs):
        worker_data = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        task = worker_data["task"]
        launched.append(task["task_id"])
        folder = Path(worker_data["output"])
        folder.mkdir(parents=True)
        backend = task["backend"]
        provider = ({"backend": "PyTorch", "device": "cpu", "dtype": "torch.float32"}
                   if backend == "torch" else
                   {"backend": "OpenVINO", "execution_devices": ["CPU"], "input_type": "f32"}
                   if backend == "openvino" else
                   {"backend": "ONNX Runtime", "providers": ["CPUExecutionProvider"],
                    "input_type": "tensor(float)"})
        rows = []
        response = {"status": "uncertain", "risk": "unknown", "reason_code": "no_supported_detections",
                    "reason": "No supported road users or signals were detected", "features": None,
                    "image_size": None, "detections": None}
        if task["task_kind"] == "measurement":
            for include in task["mode_order"]:
                for index in range(5):
                    rows.append({"kind": "pipeline", "task_kind": "measurement", "task_id": task["task_id"],
                                 "config_id": task["config_id"], "global_round": task["global_round"],
                                 "config_sha256": "config-sha", "performance_manifest_sha256": "manifest-sha",
                                 "source_checkpoint_sha256": "source-sha",
                                 "artifact_manifest_sha256": artifact_hashes.get(task["config_id"].replace("-640", "-640").replace("-960", "-960")),
                                 "phase": "warmup", "request_index": index, "sample_id": f"s{index}",
                                 "include_detections": include, "elapsed_ms": 10000})
                for index, sample in enumerate(sample_rows):
                    rows.append({"kind": "pipeline", "task_kind": "measurement", "task_id": task["task_id"],
                                 "config_id": task["config_id"], "global_round": task["global_round"],
                                 "config_sha256": "config-sha", "performance_manifest_sha256": "manifest-sha",
                                 "source_checkpoint_sha256": "source-sha",
                                 "artifact_manifest_sha256": artifact_hashes.get(task["config_id"]),
                                 "phase": "measurement", "request_index": index, "sample_id": sample["sample_id"],
                                 "include_detections": include, "elapsed_ms": 10, "pipeline_failure": False})
        for include in task["mode_order"]:
            for index in range(5):
                rows.append({"kind": "api", "task_kind": task["task_kind"], "task_id": task["task_id"],
                             "config_id": task["config_id"], "global_round": task["global_round"],
                             "config_sha256": "config-sha", "performance_manifest_sha256": "manifest-sha",
                             "source_checkpoint_sha256": "source-sha",
                             "artifact_manifest_sha256": artifact_hashes.get(task["config_id"]),
                             "phase": "warmup", "request_index": index, "sample_id": f"s{index}",
                             "include_detections": include, "http_status": 200, "elapsed_ms": 10000,
                             "pipeline_failure": False})
            if task["task_kind"] == "measurement":
                for index, sample in enumerate(sample_rows):
                    rows.append({"kind": "api", "task_kind": task["task_kind"], "task_id": task["task_id"],
                                 "config_id": task["config_id"], "global_round": task["global_round"],
                                 "config_sha256": "config-sha", "performance_manifest_sha256": "manifest-sha",
                                 "source_checkpoint_sha256": "source-sha",
                                 "artifact_manifest_sha256": artifact_hashes.get(task["config_id"]),
                                 "phase": "sequential", "request_index": index,
                                 "sample_id": sample["sample_id"], "include_detections": include,
                                 "http_status": 200, "elapsed_ms": 10, "pipeline_failure": False})
            else:
                for clients in (1, 2, 4):
                    for index in range(clients * 20):
                        rows.append({"kind": "api", "task_kind": task["task_kind"], "task_id": task["task_id"],
                                     "config_id": task["config_id"], "global_round": None,
                                     "config_sha256": "config-sha", "performance_manifest_sha256": "manifest-sha",
                                     "source_checkpoint_sha256": "source-sha",
                                     "artifact_manifest_sha256": artifact_hashes.get(task["config_id"]),
                                     "phase": f"concurrency-{clients}", "request_index": index,
                                     "sample_id": f"s{index % 100}", "include_detections": include,
                                     "http_status": 503, "elapsed_ms": 50000, "pipeline_failure": False})
        (folder / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        if task["task_kind"] == "measurement":
            round_rows = [{"global_round": task["global_round"], "sample_id": sample["sample_id"],
                           "include_detections": include, "response": response, "objects": []}
                          for sample in sample_rows for include in task["mode_order"]]
            (folder / "round_outputs.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in round_rows), encoding="utf-8")
            if task["global_round"] == 0:
                (folder / "parity_outputs.jsonl").write_text(
                    "".join(json.dumps({"sample_id": sample["sample_id"], "include_detections": include,
                                        "response": response, "objects": []}) + "\n"
                            for sample in sample_rows for include in task["mode_order"]), encoding="utf-8")
        summary = {"schema_version": 2, "status": "complete", "task_id": task["task_id"],
                   "task_kind": task["task_kind"], "config_id": task["config_id"],
                   "config_sha256": "config-sha", "performance_manifest_sha256": "manifest-sha",
                   "source_checkpoint_sha256": "source-sha",
                   "artifact_manifest_sha256": artifact_hashes.get(task["config_id"]),
                   "actual_provider": provider, "peak_process_rss_bytes": 100}
        (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = compare(input_root / "config.json", tmp_path / "run", root=Path.cwd())
    expected = [task["task_id"] for task in build_measurement_schedule() + build_concurrency_schedule()]
    assert launched == expected
    assert launched[:18] == [task["task_id"] for task in build_measurement_schedule()]
    assert result["measurement_complete"] is True
    assert result["status"] == "complete"
    assert result["round_stability"]["status"] == "pass"
