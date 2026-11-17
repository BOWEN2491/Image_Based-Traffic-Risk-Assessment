"""Bounded, reproducible CPU profiling for pipeline and local API requests.

Paths in config are resolved relative to the repository root (current working
directory), unless they are absolute. Output directories are never overwritten.
The generated sample manifest contains local image identities and stays ignored.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import csv
import hashlib
import importlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


_ACTIVE_RECORD: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "cpu_profile_active_record", default=None
)
PROFILED_STAGES = frozenset({
    "opencv_imread_ms", "image_decode_ms", "legacy_runtime_detect_ms", "detector_detect_image_ms",
    "detector_preprocess_ms", "detector_forward_ms", "detector_postprocess_nms_ms",
    "feature_build_ms", "hard_rules_ms", "signal_policy_ms", "overlay_details_ms",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_ms(name: str, elapsed_ns: int) -> None:
    record = _ACTIVE_RECORD.get()
    if record is not None:
        record.setdefault("stages_ms", {}).setdefault(name, []).append(elapsed_ns / 1_000_000)


@contextmanager
def _stage(name: str) -> Iterator[None]:
    started = time.perf_counter_ns()
    try:
        yield
    finally:
        _record_ms(name, time.perf_counter_ns() - started)


def _wrap_sync(module: Any, attribute: str, stage: str, *, count: str | None = None,
               observer: Callable[[Any, dict[str, Any]], None] | None = None) -> tuple[Any, Callable[..., Any]]:
    original = getattr(module, attribute)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        record = _ACTIVE_RECORD.get()
        if record is not None and count:
            record[count] = record.get(count, 0) + 1
        if record is not None:
            calls = record.setdefault("stage_calls", {})
            calls[stage] = calls.get(stage, 0) + 1
        with _stage(stage):
            result = original(*args, **kwargs)
        if record is not None:
            if observer is not None:
                observer(result, record)
        return result

    setattr(module, attribute, wrapped)
    return original, wrapped


def _wrap_async(module: Any, attribute: str, stage: str) -> tuple[Any, Callable[..., Any]]:
    original = getattr(module, attribute)

    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        with _stage(stage):
            return await original(*args, **kwargs)

    setattr(module, attribute, wrapped)
    return original, wrapped


def _sample_tokens(row: dict[str, Any]) -> set[str]:
    objects = row.get("objects") or []
    categories = {str(obj.get("category", "")).lower().replace("_", " ") for obj in objects}
    count = len(objects)
    area = int(row["width"]) * int(row["height"])
    density = "none" if count == 0 else "few" if count <= 4 else "many"
    size = "small" if area <= 1280 * 720 else "medium" if area <= 1920 * 1080 else "large"
    tokens = {f"density:{density}", f"resolution:{size}"}
    if "traffic light" in categories:
        tokens.add("target:traffic-light")
    if "person" in categories:
        tokens.add("target:person")
    if categories & {"car", "bus", "truck", "train", "motorcycle", "bicycle"}:
        tokens.add("target:vehicle-or-rider")
    attributes = row.get("frame_attributes") or {}
    if attributes.get("timeofday"):
        tokens.add(f"timeofday:{str(attributes['timeofday']).lower()}")
    if attributes.get("scene"):
        tokens.add(f"scene:{str(attributes['scene']).lower()}")
    return tokens


def select_performance_samples(rows: list[dict[str, Any]], count: int, seed: int = 42) -> list[dict[str, Any]]:
    """Greedily balance development-only size, density and object categories."""
    if count <= 0:
        raise ValueError("sample_count must be positive")
    candidates = [row for row in rows if row.get("source_split") == "val" and row.get("split") == "development"]
    by_group: dict[str, dict[str, Any]] = {}
    for row in candidates:
        group = row.get("group_id")
        if not group:
            raise ValueError("development performance samples require group_id")
        by_group.setdefault(str(group), row)
    candidates = list(by_group.values())
    if not candidates:
        raise ValueError("no development rows are available for performance profiling")
    rng = random.Random(seed)
    tie_break = {str(row["sample_id"]): rng.random() for row in candidates}
    frequencies = Counter(token for row in candidates for token in _sample_tokens(row))
    selected: list[dict[str, Any]] = []
    selected_groups: set[str] = set()
    covered: Counter[str] = Counter()
    remaining = list(candidates)
    target = min(count, len(remaining))
    while remaining and len(selected) < target:
        def score(row: dict[str, Any]) -> tuple[float, float, str]:
            tokens = sorted(_sample_tokens(row))
            rarity = sum(1.0 / frequencies[token] for token in tokens if covered[token] == 0)
            balance = sum(1.0 / (1 + covered[token]) for token in tokens)
            return (rarity, balance, f"{tie_break[str(row['sample_id'])]:.16f}")

        chosen = max(remaining, key=score)
        remaining.remove(chosen)
        group = str(chosen["group_id"])
        if group in selected_groups:
            continue
        selected_groups.add(group)
        selected.append(chosen)
        covered.update(_sample_tokens(chosen))
    return selected


def latency_summary(values_ms: list[float]) -> dict[str, float | int | None]:
    if not values_ms:
        return {"count": 0, "p50_ms": None, "p95_ms": None}
    ordered = sorted(values_ms)

    def percentile(p: float) -> float:
        index = (len(ordered) - 1) * p
        low = math.floor(index)
        high = math.ceil(index)
        if low == high:
            return float(ordered[low])
        return float(ordered[low] * (high - index) + ordered[high] * (index - low))

    return {"count": len(ordered), "p50_ms": percentile(0.50), "p95_ms": percentile(0.95)}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    required = {
        "dataset_root", "source_manifest", "model_dir", "seed", "sample_count", "threads",
        "warmups", "rounds", "include_detections", "requests_per_client", "time_budget_seconds",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"missing config fields: {', '.join(missing)}")
    if config["threads"] != 2 or config["seed"] != 42:
        raise ValueError("this plan fixes seed=42 and CPU threads=2")
    if not isinstance(config["include_detections"], bool):
        raise ValueError("include_detections must be a boolean; profile each mode in a separate process")
    if int(config["sample_count"]) <= 0 or int(config["rounds"]) <= 0 or int(config["warmups"]) < 0:
        raise ValueError("sample_count/rounds must be positive and warmups non-negative")
    if float(config["time_budget_seconds"]) <= 0 or float(config["time_budget_seconds"]) > 1800:
        raise ValueError("time_budget_seconds must be in (0, 1800]")
    if int(config["requests_per_client"]) < 20:
        raise ValueError("API concurrency probes require at least 20 requests per client")
    for key in ("dataset_root", "source_manifest", "model_dir"):
        config[key] = str(_resolve(root, str(config[key])))
    return config


class _RSSSampler:
    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _RSSSampler:
        import psutil

        process = psutil.Process(os.getpid())

        def sample() -> None:
            while not self._stop.is_set():
                try:
                    self.peak_bytes = max(self.peak_bytes, process.memory_info().rss)
                except psutil.Error:
                    pass
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=sample, name="cpu-profile-rss", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)


class _Probe:
    """Process-local, opt-in timing wrappers used only by this isolated CLI."""

    def __init__(self) -> None:
        self.restorers: list[tuple[Any, str, Any]] = []
        self.records: dict[str, dict[str, Any]] = {}

    def patch_sync(self, module: Any, attribute: str, stage: str, *, count: str | None = None,
                   observer: Callable[[Any, dict[str, Any]], None] | None = None) -> None:
        original, wrapped = _wrap_sync(module, attribute, stage, count=count, observer=observer)
        self.restorers.append((module, attribute, original))

    def patch_async(self, module: Any, attribute: str, stage: str) -> None:
        original, wrapped = _wrap_async(module, attribute, stage)
        self.restorers.append((module, attribute, original))

    def install(self) -> None:
        import cv2
        from PIL import Image
        from traffic_risk import app as app_module
        from traffic_risk import model_runtime, pipeline
        from ultralytics.models.yolo.detect.predict import DetectionPredictor

        try:
            from traffic_risk import image_io
        except ImportError:
            image_io = None

        self.patch_sync(cv2, "imread", "opencv_imread_ms", count="opencv_imread_calls")
        self.patch_sync(Image, "open", "pillow_open_ms", count="pillow_open_calls")
        if image_io is not None:
            self.patch_sync(image_io, "decode_image", "image_decode_ms")
        # The decoded path imports decode_image at its call site. The v0.3.0
        # path has no shared decoder module and uses ModelRuntime.detect(path).
        if hasattr(pipeline, "decode_image"):
            self.patch_sync(pipeline, "decode_image", "image_decode_ms")
        def count_detections(result: dict[str, Any], record: dict[str, Any]) -> None:
            record["detected_object_count"] = sum(
                len(frame.get("objects", [])) for frame in result.get("frames", [])
            )
        if hasattr(model_runtime.ModelRuntime, "detect_image"):
            self.patch_sync(model_runtime.ModelRuntime, "detect_image", "detector_detect_image_ms",
                            observer=count_detections)
        elif not hasattr(pipeline, "_run_detected"):
            self.patch_sync(model_runtime.ModelRuntime, "detect", "legacy_runtime_detect_ms",
                            observer=count_detections)
        self.patch_sync(DetectionPredictor, "preprocess", "detector_preprocess_ms")
        self.patch_sync(DetectionPredictor, "inference", "detector_forward_ms")
        # Ultralytics 8.4.126 places detection NMS in this task-specific override.
        self.patch_sync(DetectionPredictor, "postprocess", "detector_postprocess_nms_ms")
        self.patch_sync(pipeline, "build_features_from_bdd", "feature_build_ms")
        for attribute in ("compute_features", "_detection_details", "_detection_details_from_size",
                          "apply_hard_rules", "apply_signal_policy"):
            if hasattr(pipeline, attribute):
                stage = "feature_build_ms" if attribute == "compute_features" else (
                    "overlay_details_ms" if attribute.startswith("_detection_details") else (
                        "hard_rules_ms" if attribute == "apply_hard_rules" else "signal_policy_ms"))
                self.patch_sync(pipeline, attribute, stage)
        if hasattr(pipeline, "_run_detected"):
            original_run_detected = pipeline._run_detected

            def run_detected_with_legacy_probe(path: Any, runtime: Any, include_detections: bool) -> Any:
                try:
                    decoded_runtime = isinstance(runtime, pipeline.DecodedRuntime)
                except (AttributeError, TypeError):
                    decoded_runtime = hasattr(runtime, "detect_image")
                if decoded_runtime or not callable(getattr(runtime, "detect", None)):
                    return original_run_detected(path, runtime, include_detections)
                original_detect = runtime.detect
                instance_values = getattr(runtime, "__dict__", {})
                had_instance_override = "detect" in instance_values
                previous_value = instance_values.get("detect")

                def timed_legacy_detect(*args: Any, **kwargs: Any) -> Any:
                    record = _ACTIVE_RECORD.get()
                    if record is not None:
                        calls = record.setdefault("stage_calls", {})
                        calls["legacy_runtime_detect_ms"] = calls.get("legacy_runtime_detect_ms", 0) + 1
                    with _stage("legacy_runtime_detect_ms"):
                        return original_detect(*args, **kwargs)

                try:
                    runtime.detect = timed_legacy_detect
                except (AttributeError, TypeError):
                    return original_run_detected(path, runtime, include_detections)
                try:
                    return original_run_detected(path, runtime, include_detections)
                finally:
                    if had_instance_override:
                        runtime.detect = previous_value
                    else:
                        delattr(runtime, "detect")

            pipeline._run_detected = run_detected_with_legacy_probe
            self.restorers.append((pipeline, "_run_detected", original_run_detected))
        self.patch_sync(app_module, "_validate_image", "api_image_validation_ms")
        self.patch_async(app_module, "_read_upload", "api_upload_read_ms")
        import anyio

        self.patch_async(anyio.Path, "write_bytes", "api_temp_write_ms")

    @staticmethod
    def finalize_record(record: dict[str, Any], include_detections: bool) -> None:
        calls = record.get("stage_calls", {})
        empty_detection = record.get("detected_object_count") == 0
        stages = sorted(PROFILED_STAGES)
        states = {}
        for stage in stages:
            if record.get("request_not_processed"):
                states[stage] = "not_applicable"
            elif calls.get(stage, 0):
                states[stage] = "measured"
            elif (stage == "legacy_runtime_detect_ms" and calls.get("detector_detect_image_ms", 0)) or (
                stage == "image_decode_ms" and calls.get("legacy_runtime_detect_ms", 0)
                and not calls.get("image_decode_ms", 0)
            ) or (
                stage == "detector_detect_image_ms" and calls.get("legacy_runtime_detect_ms", 0)
                and not calls.get("detector_detect_image_ms", 0)
            ):
                states[stage] = "not_applicable"
            elif (stage in {"detector_preprocess_ms", "detector_forward_ms", "detector_postprocess_nms_ms"}
                  and calls.get("legacy_runtime_detect_ms", 0)
                  and not calls.get("detector_detect_image_ms", 0)):
                states[stage] = "not_applicable"
            elif stage in {"feature_build_ms", "hard_rules_ms"} and empty_detection:
                states[stage] = "not_applicable"
            elif stage == "signal_policy_ms" and (
                empty_detection or record.get("risk_mode") == "rules"
            ):
                states[stage] = "not_applicable"
            elif stage == "overlay_details_ms" and not include_detections:
                states[stage] = "not_applicable"
            else:
                states[stage] = "missing"
        record["stage_calls"] = {stage: calls.get(stage, 0) for stage in stages}
        record["stage_status"] = states

    def restore(self) -> None:
        for module, attribute, original in reversed(self.restorers):
            setattr(module, attribute, original)
        self.restorers.clear()


class _TimedASGI:
    def __init__(self, app: Any, probe: _Probe) -> None:
        self.app = app
        self.probe = probe

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") != "/api/predict":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        request_id = headers.get(b"x-profile-id", b"unknown").decode("ascii", errors="replace")
        record: dict[str, Any] = {"stages_ms": {}, "opencv_imread_calls": 0, "pillow_open_calls": 0}
        self.probe.records[request_id] = record
        token = _ACTIVE_RECORD.set(record)
        started = time.perf_counter_ns()
        try:
            async def send_timed(message: dict[str, Any]) -> None:
                if message.get("type") == "http.response.start":
                    record["server_status_code"] = int(message["status"])
                await send(message)
            await self.app(scope, receive, send_timed)
        finally:
            record["api_server_total_ms"] = (time.perf_counter_ns() - started) / 1_000_000
            _ACTIVE_RECORD.reset(token)


def _flat_row(record: dict[str, Any], *, elapsed_ms: float, sample_id: str, kind: str,
              round_index: int | None, include_detections: bool, **extra: Any) -> dict[str, Any]:
    stages = record.get("stages_ms", {})
    result = {
        "kind": kind, "round": round_index, "sample_id": sample_id,
        "include_detections": include_detections, "elapsed_ms": elapsed_ms,
        "server_status_code": record.get("server_status_code"),
        "reason_code": record.get("reason_code"), "status": record.get("status"), "risk": record.get("risk"),
        "opencv_imread_calls": record.get("opencv_imread_calls", 0),
        "pillow_open_calls": record.get("pillow_open_calls", 0),
        "stages_ms_json": json.dumps(stages, sort_keys=True),
        "stage_calls_json": json.dumps(record.get("stage_calls", {}), sort_keys=True),
        "stage_status_json": json.dumps(record.get("stage_status", {}), sort_keys=True),
        "detected_object_count": record.get("detected_object_count"),
    }
    result.update(extra)
    return result


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


async def _gather_cancel_on_error(tasks: list[asyncio.Task[Any]]) -> int:
    """Wait for every task; cancel and join peers before propagating a failure."""
    try:
        await asyncio.gather(*tasks)
    except BaseException as exc:
        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            return len(pending)
        raise
    return 0


async def _profile_api(app: Any, probe: _Probe, config: dict[str, Any], samples: list[dict[str, Any]],
                       output: Path, deadline: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import httpx

    from traffic_risk.model_runtime import Settings

    settings = Settings(mode="rules", model_dir=Path(config["model_dir"]),
                        upload_dir=output / "uploads", device="cpu", inference_concurrency=1,
                        busy_timeout_seconds=float(config.get("busy_timeout_seconds", 0.1)))
    application = app(settings, runtime_factory=lambda s: __import__("traffic_risk.model_runtime", fromlist=["ModelRuntime"]).ModelRuntime(s))
    rows: list[dict[str, Any]] = []
    api_summary: dict[str, Any] = {"include_detections": config["include_detections"], "rounds": [], "concurrency": {}}
    async with application.router.lifespan_context(application):
        gate = application.state.inference_gate
        original_acquire = gate.acquire

        async def acquire_timed() -> bool:
            started = time.perf_counter_ns()
            try:
                return await original_acquire()
            finally:
                _record_ms("api_gate_wait_ms", time.perf_counter_ns() - started)
        gate.acquire = acquire_timed
        transport = httpx.ASGITransport(app=_TimedASGI(application, probe))
        async with httpx.AsyncClient(transport=transport, base_url="http://profile.local") as client:
            ready = await client.get("/ready")
            if ready.status_code != 200:
                raise RuntimeError(f"API readiness failed with HTTP {ready.status_code}")

            async def request(sample: dict[str, Any], request_id: str, round_index: int | None,
                              phase: str, request_index: int | None = None) -> dict[str, Any]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("profiling time budget reached")
                path = Path(config["dataset_root"]) / sample["image_path"]
                image_bytes = path.read_bytes()
                start = time.perf_counter_ns()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("profiling time budget reached")
                try:
                    response = await asyncio.wait_for(client.post(
                        "/api/predict", params={"include_detections": str(config["include_detections"]).lower()},
                        headers={"x-profile-id": request_id},
                        files={"file": ("profile" + path.suffix.lower(), image_bytes,
                                        {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                                         ".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "image/jpeg"))},
                    ), timeout=remaining)
                except TimeoutError as exc:
                    probe.records.pop(request_id, None)
                    raise TimeoutError("profiling time budget reached during API request") from exc
                except BaseException:
                    probe.records.pop(request_id, None)
                    raise
                payload: dict[str, Any] = {}
                try:
                    payload = response.json()
                except ValueError:
                    pass
                elapsed = (time.perf_counter_ns() - start) / 1_000_000
                if response.status_code == 200 and isinstance(payload, dict):
                    payload = payload
                else:
                    payload = {}
                record = probe.records.pop(request_id, {})
                record["risk_mode"] = settings.mode
                record["request_not_processed"] = response.status_code != 200
                record.update({"status": payload.get("status"), "risk": payload.get("risk"),
                               "reason_code": payload.get("reason_code"), "server_status_code": response.status_code})
                from traffic_risk.inference_outcome import is_processing_failure
                record["pipeline_failure"] = response.status_code == 200 and is_processing_failure(
                    payload.get("reason_code")
                )
                _Probe.finalize_record(record, bool(config["include_detections"]))
                rows.append(_flat_row(record, elapsed_ms=elapsed, sample_id=sample["sample_id"], kind="api",
                                      round_index=round_index, include_detections=config["include_detections"],
                                      phase=phase, request_index=request_index, http_status=response.status_code,
                                      pipeline_failure=record.get("pipeline_failure"),
                                      api_server_total_ms=record.get("api_server_total_ms")))
                _append_jsonl(output / "requests.partial.jsonl", rows[-1])
                return {"status_code": response.status_code, "reason_code": payload.get("reason_code"),
                        "elapsed_ms": elapsed,
                        "pipeline_failure": response.status_code == 200 and is_processing_failure(payload.get("reason_code"))}

            try:
                for index in range(int(config["warmups"])):
                    await request(samples[index % len(samples)], f"warmup-{index}", None, "warmup", index)
                for round_index in range(int(config["rounds"])):
                    round_results = []
                    for index, sample in enumerate(samples):
                        response = await request(sample, f"seq-{round_index}-{index}", round_index, "sequential", index)
                        round_results.append(response)
                    api_summary["rounds"].append(_summarize_responses(round_results))

                async def run_concurrency(clients: int, concurrency_results: list[dict[str, Any]]) -> tuple[int, int]:
                    cursor = 0
                    cursor_lock = asyncio.Lock()

                    async def worker(worker_id: int) -> None:
                        nonlocal cursor
                        for request_index in range(int(config["requests_per_client"])):
                            async with cursor_lock:
                                sample = samples[cursor % len(samples)]
                                serial = cursor
                                cursor += 1
                            result = await request(sample, f"c{clients}-w{worker_id}-{request_index}", None,
                                                   f"concurrency-{clients}", serial)
                            concurrency_results.append(result)

                    tasks = [asyncio.create_task(worker(worker_id)) for worker_id in range(clients)]
                    try:
                        await asyncio.gather(*tasks)
                    except BaseException as exc:
                        pending = [task for task in tasks if not task.done()]
                        for task in pending:
                            task.cancel()
                        if pending:
                            await asyncio.gather(*pending, return_exceptions=True)
                        if isinstance(exc, TimeoutError):
                            return len(pending), 1
                        raise
                    return 0, 0

                for clients in (1, 2, 4):
                    concurrency_results: list[dict[str, Any]] = []
                    try:
                        cancelled, failed = await run_concurrency(clients, concurrency_results)
                        api_summary["concurrency"][str(clients)] = _summarize_responses(concurrency_results)
                        api_summary["concurrency"][str(clients)]["cancelled_workers"] = cancelled
                        api_summary["concurrency"][str(clients)]["failed_or_timed_out_workers"] = failed
                        if failed:
                            api_summary["partial"] = True
                            api_summary["partial_reason"] = "concurrency_timeout"
                            break
                    except TimeoutError:
                        api_summary["concurrency"][str(clients)] = _summarize_responses(concurrency_results)
                        api_summary["concurrency"][str(clients)]["partial"] = True
                        break
            except TimeoutError:
                api_summary["partial"] = True
                api_summary["partial_reason"] = "time_budget"
                if "clients" in locals() and "concurrency_results" in locals():
                    api_summary["concurrency"][str(clients)] = _summarize_responses(concurrency_results)
                    api_summary["concurrency"][str(clients)]["partial"] = True
                if "round_results" in locals() and round_results:
                    api_summary["rounds"].append({**_summarize_responses(round_results), "partial": True})
    api_summary["completeness"] = _profile_completeness(
        rows, config, len(samples), [str(row["sample_id"]) for row in samples]
    )["api"]
    return rows, api_summary


def _profile_completeness(rows: list[dict[str, Any]], config: dict[str, Any], sample_count: int,
                          sample_ids: list[str] | None = None) -> dict[str, Any]:
    """Check the frozen profiler denominator by phase, round and sample identity."""
    expected_sample_ids = sample_ids or [f"sample-{index}" for index in range(sample_count)]
    if len(expected_sample_ids) != sample_count or len(set(expected_sample_ids)) != sample_count:
        raise ValueError("expected sample IDs must be unique and match sample_count")
    expected_pipeline = int(config["rounds"]) * sample_count
    pipeline = [row for row in rows if row.get("kind") == "pipeline"]
    expected_pipeline_tasks = {(round_index, sample_index) for round_index in range(int(config["rounds"]))
                               for sample_index in range(sample_count)}
    expected_pipeline_keys = {(round_index, sample_index, expected_sample_ids[sample_index])
                              for round_index, sample_index in expected_pipeline_tasks}
    actual_pipeline_keys = {(row.get("round"), row.get("sample_index"), row.get("sample_id"))
                            for row in pipeline}
    pipeline_by_round = {index: [row for row in pipeline if row.get("round") == index]
                         for index in range(int(config["rounds"]))}
    pipeline_complete = (len(pipeline) == expected_pipeline and actual_pipeline_keys == expected_pipeline_keys
                         and all(len(items) == sample_count for items in pipeline_by_round.values())
                         and not any(row.get("pipeline_failure") is True for row in pipeline))
    api = [row for row in rows if row.get("kind") == "api"]
    expected_warmups = int(config["warmups"])
    expected_sequential = int(config["rounds"]) * sample_count
    expected_concurrency = (1 + 2 + 4) * int(config["requests_per_client"])
    counts = {phase: sum(row.get("phase") == phase for row in api)
              for phase in ("warmup", "sequential", "concurrency-1", "concurrency-2", "concurrency-4")}
    expected_seq_tasks = {(round_index, sample_index) for round_index in range(int(config["rounds"]))
                          for sample_index in range(sample_count)}
    expected_seq_keys = {(round_index, sample_index, expected_sample_ids[sample_index])
                         for round_index, sample_index in expected_seq_tasks}
    actual_seq_keys = {(row.get("round"), row.get("request_index"), row.get("sample_id")) for row in api
                       if row.get("phase") == "sequential"}
    expected_warmup_keys = {(index, expected_sample_ids[index % sample_count])
                            for index in range(expected_warmups)}
    actual_warmup_keys = {(row.get("request_index"), row.get("sample_id")) for row in api
                          if row.get("phase") == "warmup"}
    concurrency_indices_complete = all(
        {(row.get("request_index"), row.get("sample_id")) for row in api
         if row.get("phase") == f"concurrency-{clients}"}
        == {(index, expected_sample_ids[index % sample_count])
            for index in range(clients * int(config["requests_per_client"]))}
        for clients in (1, 2, 4)
    )
    concurrency_unique = all(
        len([row.get("request_index") for row in api if row.get("phase") == f"concurrency-{clients}"])
        == len({row.get("request_index") for row in api if row.get("phase") == f"concurrency-{clients}"})
        for clients in (1, 2, 4)
    )
    api_complete = (counts["warmup"] == expected_warmups and counts["sequential"] == expected_sequential
                    and actual_seq_keys == expected_seq_keys
                    and actual_warmup_keys == expected_warmup_keys and concurrency_unique
                    and concurrency_indices_complete
                    and all(counts[f"concurrency-{n}"] == n * int(config["requests_per_client"])
                            for n in (1, 2, 4))
                    and all(row.get("http_status") in ({200, 503} if str(row.get("phase", "")).startswith("concurrency-") else {200})
                            for row in api)
                    and not any(row.get("pipeline_failure", False) for row in api))
    profiled_rows = [row for row in rows if row.get("kind") in {"pipeline", "api"}]
    coverage_payloads = []
    for row in profiled_rows:
        try:
            coverage_payloads.append(json.loads(row.get("stage_status_json", "{}")))
        except (TypeError, json.JSONDecodeError):
            coverage_payloads.append({})
    coverage_complete = all(
        set(payload) == PROFILED_STAGES
        and all(state in {"measured", "not_applicable"} for state in payload.values())
        for payload in coverage_payloads
    )
    coverage_complete = coverage_complete and len(coverage_payloads) == len(profiled_rows)
    return {"pipeline": {"expected_rows": expected_pipeline, "actual_rows": len(pipeline),
                         "complete": pipeline_complete},
            "api": {"expected_warmup_rows": expected_warmups, "expected_sequential_rows": expected_sequential,
                    "expected_concurrency_rows": expected_concurrency, "phase_rows": counts,
                    "actual_rows": len(api), "complete": api_complete and coverage_complete},
            "probe_coverage_complete": coverage_complete,
            "complete": pipeline_complete and api_complete and coverage_complete}


def _summarize_responses(responses: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["elapsed_ms"]) for row in responses if row.get("status_code") == 200]
    return {**latency_summary(latencies), "responses": len(responses),
            "success": sum(row.get("status_code") == 200 for row in responses),
            "busy_503": sum(row.get("status_code") == 503 for row in responses),
            "other_failure": sum(row.get("status_code") not in {200, 503} for row in responses),
            "pipeline_failures": sum(row.get("pipeline_failure", row.get("status_code") == 200
                                          and row.get("reason_code") not in {
                                              "hard_rule_triggered", "model_prediction", "no_supported_detections",
                                              "no_rule_triggered",
                                          }) for row in responses)}


def _round_summary(rows: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    rounds = sorted({row["round"] for row in rows if row["kind"] == kind and row["round"] is not None})
    summaries = []
    for round_index in rounds:
        subset = [row for row in rows if row["kind"] == kind and row["round"] == round_index]
        successful = [row for row in subset if row.get("reason_code") not in {
            "feature_schema_error", "perception_error", "invalid_features", "invalid_model_output", "legacy_unspecified"}]
        elapsed = [float(row["elapsed_ms"]) for row in successful]
        p = latency_summary(elapsed)
        summaries.append({"round": round_index, **p, "failures": len(subset) - len(successful),
                          "sample_count": len(subset), "median_stage_p50_ms": _stage_percentiles(subset)})
    if summaries:
        p95s = [row["p95_ms"] for row in summaries if row["p95_ms"] is not None]
        for item in summaries:
            item["round_p95_median_ms"] = statistics.median(p95s) if p95s else None
    return summaries


def _stage_percentiles(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    collected: dict[str, list[float]] = {}
    for row in rows:
        try:
            stages = json.loads(row.get("stages_ms_json", "{}"))
        except (TypeError, json.JSONDecodeError):
            continue
        for name, values in stages.items():
            collected.setdefault(name, []).extend(float(value) for value in values)
    return {name: latency_summary(values) for name, values in sorted(collected.items())}


def _stage_coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    coverage: dict[str, Counter[str]] = {}
    for row in rows:
        try:
            states = json.loads(row.get("stage_status_json", "{}"))
        except (TypeError, json.JSONDecodeError):
            continue
        for stage, state in states.items():
            coverage.setdefault(stage, Counter())[state] += 1
    return {stage: dict(sorted(counts.items())) for stage, counts in sorted(coverage.items())}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _prepare_manifest(config: dict[str, Any], output: Path) -> tuple[list[dict[str, Any]], str]:
    source = Path(config["source_manifest"])
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = select_performance_samples(rows, int(config["sample_count"]), int(config["seed"]))
    root = Path(config["dataset_root"])
    if len(samples) < int(config["sample_count"]):
        config["sample_shortfall"] = int(config["sample_count"]) - len(samples)
    for row in samples:
        image_path = (root / row["image_path"]).resolve()
        if not image_path.is_relative_to(root.resolve()):
            raise ValueError("A selected performance image escapes dataset_root")
        if not image_path.is_file() or sha256_file(image_path) != row["image_sha256"]:
            raise ValueError("A selected performance image is missing or changed")
    local_manifest = []
    for row in samples:
        local_manifest.append({
            "sample_id": row["sample_id"], "group_id": row["group_id"], "split": row["split"],
            "source_split": row["source_split"], "image_path": row["image_path"],
            "image_sha256": row["image_sha256"], "pixel_sha256": row["pixel_sha256"],
            "width": row["width"], "height": row["height"], "category_counts": dict(Counter(
                str(obj.get("category", "unknown")).lower().replace("_", " ") for obj in row.get("objects", []))),
            "selection_tokens": sorted(_sample_tokens(row)),
        })
    path = output / "performance_manifest.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in local_manifest), encoding="utf-8")
    return samples, sha256_file(path)


def run(config_path: Path, output: Path, repo_root: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path.cwd()).resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError("run from the workspace root or set repo_root")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    config = _load_config(config_path, root)
    config_path = config_path.resolve()
    output.mkdir(parents=True, exist_ok=False)
    samples, sample_manifest_sha = _prepare_manifest(config, output)
    source_manifest = Path(config["source_manifest"])
    sample_root = Path(config["dataset_root"])
    os.environ["OMP_NUM_THREADS"] = str(config["threads"])
    os.environ["MKL_NUM_THREADS"] = str(config["threads"])
    os.environ["OPENBLAS_NUM_THREADS"] = str(config["threads"])
    random.seed(int(config["seed"]))
    import numpy as np
    import torch
    import cv2
    import httpx  # noqa: F401
    from traffic_risk import app as app_module
    from traffic_risk.config import Settings
    from traffic_risk.evaluate_detection import pipeline_failed
    from traffic_risk.model_runtime import ModelRuntime
    from traffic_risk.pipeline import run_single_image
    import importlib.metadata as metadata

    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.set_num_threads(int(config["threads"]))
    torch.set_num_interop_threads(int(config["threads"]))
    cv2.setNumThreads(int(config["threads"]))
    settings = Settings(mode="rules", model_dir=Path(config["model_dir"]), device="cpu")
    runtime = ModelRuntime(settings)
    cold_start = time.perf_counter_ns()
    runtime.load()
    cold_load_ms = (time.perf_counter_ns() - cold_start) / 1_000_000
    probe = _Probe()
    try:
        probe.install()
    except BaseException:
        probe.restore()
        raise
    deadline = time.monotonic() + float(config["time_budget_seconds"])
    rows: list[dict[str, Any]] = []
    partial = False
    try:
        with _RSSSampler() as rss:
            for warm_index in range(int(config["warmups"])):
                if time.monotonic() >= deadline:
                    partial = True
                    break
                sample = samples[warm_index % len(samples)]
                record: dict[str, Any] = {"stages_ms": {}, "opencv_imread_calls": 0, "pillow_open_calls": 0}
                record["risk_mode"] = settings.mode
                token = _ACTIVE_RECORD.set(record)
                try:
                    run_single_image(sample_root / sample["image_path"], runtime,
                                     include_detections=bool(config["include_detections"]))
                finally:
                    _ACTIVE_RECORD.reset(token)
            if partial:
                # Do not start the formal pipeline or API phase with no budget left.
                report_partial = True
            else:
                report_partial = False
            for round_index in range(int(config["rounds"])):
                if partial:
                    break
                for sample_index, sample in enumerate(samples):
                    if time.monotonic() >= deadline:
                        partial = True
                        break
                    record = {"stages_ms": {}, "opencv_imread_calls": 0, "pillow_open_calls": 0}
                    record["risk_mode"] = settings.mode
                    token = _ACTIVE_RECORD.set(record)
                    start = time.perf_counter_ns()
                    try:
                        response = run_single_image(sample_root / sample["image_path"], runtime,
                                                    include_detections=bool(config["include_detections"]))
                    finally:
                        elapsed = (time.perf_counter_ns() - start) / 1_000_000
                        _ACTIVE_RECORD.reset(token)
                    record.update({"status": response.get("status"), "risk": response.get("risk"),
                                   "reason_code": response.get("reason_code")})
                    _Probe.finalize_record(record, bool(config["include_detections"]))
                    pipeline_row = _flat_row(
                        record, elapsed_ms=elapsed, sample_id=sample["sample_id"], kind="pipeline",
                        round_index=round_index, include_detections=bool(config["include_detections"]),
                        sample_index=sample_index, pipeline_failure=pipeline_failed(response),
                    )
                    rows.append(pipeline_row)
                    _append_jsonl(output / "requests.partial.jsonl", pipeline_row)
                if partial:
                    break
            api_rows: list[dict[str, Any]] = []
            api_summary: dict[str, Any] = {"skipped": True}
            if not partial:
                api_rows, api_summary = asyncio.run(_profile_api(app_module.create_app, probe, config, samples,
                                                                 output, deadline))
                rows.extend(api_rows)
            report_partial = partial or bool(api_summary.get("partial"))
            source_paths = {
                "pipeline_source": root / "src/traffic_risk/pipeline.py",
                "model_runtime_source": root / "src/traffic_risk/model_runtime.py",
                "api_source": root / "src/traffic_risk/app.py",
                "image_io_source": root / "src/traffic_risk/image_io.py",
                "build_features_source": root / "src/traffic_risk/build_features.py",
                "hard_rules_source": root / "src/traffic_risk/hard_rules.py",
            }
            input_hashes = {name: sha256_file(path) for name, path in source_paths.items() if path.is_file()}
            module_names = {
                "image_io": "traffic_risk.image_io",
                "pipeline": "traffic_risk.pipeline",
                "model_runtime": "traffic_risk.model_runtime",
                "ultralytics_detection_predictor": "ultralytics.models.yolo.detect.predict",
                "build_features": "traffic_risk.build_features",
                "hard_rules": "traffic_risk.hard_rules",
            }
            probe_modules = {}
            for name, module_name in module_names.items():
                try:
                    module = importlib.import_module(module_name)
                except ImportError:
                    continue
                module_path = Path(module.__file__).resolve()
                probe_modules[name] = {"path": str(module_path), "sha256": sha256_file(module_path)}
            git_result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                        capture_output=True, text=True, check=False)
            source_revision = (git_result.stdout.strip() if git_result.returncode == 0
                               else str(config.get("source_revision", "unversioned")))
            report = {
                "schema_version": 1, "partial": report_partial,
                "partial_reason": "time_budget" if report_partial else None,
                "run_id": output.name,
                "git_commit": source_revision,
                "input_sha256": {"profile_config": sha256_file(config_path),
                                 "profile_tool": sha256_file(Path(__file__).resolve()), **input_hashes},
                "probe_modules": probe_modules,
                "source_manifest_sha256": sha256_file(source_manifest),
                "performance_manifest_sha256": sample_manifest_sha,
                "sample_count": len(samples), "sample_shortfall": config.get("sample_shortfall", 0),
                "selection": {"source_split": "val", "role": "development", "seed": config["seed"],
                              "unique_group": True, "tokens": "resolution, object density, person, vehicle/rider, traffic light, scene/time-of-day"},
                "configuration": {key: config[key] for key in (
                    "seed", "sample_count", "threads", "warmups", "rounds", "include_detections",
                    "requests_per_client", "time_budget_seconds")},
                "model": {"name": "YOLO11n", "sha256": sha256_file(settings.yolo_path),
                          "backend": "PyTorch CPU", "confidence": 0.25, "iou": 0.45, "max_det": 300,
                          "imgsz": 640, "batch": 1},
                "libraries": {name: metadata.version(name) for name in (
                    "image-based-traffic-risk-assessment", "torch", "torchvision", "ultralytics", "opencv-python", "numpy")},
                "source_version": __import__("traffic_risk").__version__,
                "cold_model_load_ms": cold_load_ms,
                "pipeline_rounds": _round_summary(rows, "pipeline"),
                "pipeline_median_round_p95_ms": statistics.median(
                    [r["p95_ms"] for r in _round_summary(rows, "pipeline") if r["p95_ms"] is not None]
                ) if _round_summary(rows, "pipeline") else None,
                "pipeline_overall": latency_summary([r["elapsed_ms"] for r in rows if r["kind"] == "pipeline"]),
                "api_rounds": api_summary.get("rounds", []), "api_concurrency": api_summary.get("concurrency", {}),
                "api_median_round_p95_ms": statistics.median(
                    [latency["p95_ms"] for latency in api_summary.get("rounds", []) if latency.get("p95_ms") is not None]
                ) if api_summary.get("rounds") else None,
                "api_overall": latency_summary([r["elapsed_ms"] for r in rows if r["kind"] == "api" and r.get("http_status") == 200]),
                "api_server_stages": _stage_percentiles([r for r in rows if r["kind"] == "api"]),
                "pipeline_stages": _stage_percentiles([r for r in rows if r["kind"] == "pipeline"]),
                "api_stage_coverage": _stage_coverage([r for r in rows if r["kind"] == "api"]),
                "pipeline_stage_coverage": _stage_coverage([r for r in rows if r["kind"] == "pipeline"]),
                "completeness": _profile_completeness(
                    rows, config, len(samples), [str(row["sample_id"]) for row in samples]
                ),
                "api_failures": {"503": sum(r.get("http_status") == 503 for r in rows if r["kind"] == "api"),
                                 "other": sum(r.get("http_status") not in {200, 503} for r in rows if r["kind"] == "api"),
                                 "processing": sum(r.get("pipeline_failure") is True for r in rows if r["kind"] == "api"),
                                 "pipeline": sum(r.get("pipeline_failure") is True for r in rows if r["kind"] == "pipeline")},
                "peak_process_rss_bytes": rss.peak_bytes, "rss_sample_interval_ms": 50,
                "timing_notes": [
                    "pipeline elapsed is end-to-end run_single_image; nested stage timings overlap and must not be summed",
                    "opencv_imread counts Python cv2.imread calls, including calls from the YOLO path loader when routed through Python",
                    "detector_detect_image measures the ModelRuntime call site; detector_preprocess/forward and task-specific postprocess include Ultralytics detector work and NMS, nested under detector_detect_image",
                    "pillow_open measures container parsing/validation or overlay image opening, not full pixel decode by itself",
                    "API client latency includes in-process ASGI transport through response JSON parsing; no network socket or browser overhead",
                    "API server total spans ASGI request entry through response send; nested stages overlap",
                    "concurrency latency summary excludes non-200 responses; 503 and other failures have separate counts",
                ],
            }
            report["partial"] = report["partial"] or not report["completeness"]["complete"]
            if report["partial"] and not report["partial_reason"]:
                report["partial_reason"] = (
                    "sample_shortfall" if config.get("sample_shortfall")
                    else "required_measurement_incomplete"
                )
            if not report["partial"] and config.get("sample_shortfall"):
                report["partial"] = True
                report["partial_reason"] = "sample_shortfall"
            (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            _write_csv(output / "requests.csv", rows)
    finally:
        probe.restore()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new run directory; existing paths are rejected")
    args = parser.parse_args(argv)
    try:
        report = run(args.config, args.output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"profile failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"run_id": report["run_id"], "partial": report["partial"],
                      "sample_count": report["sample_count"], "pipeline": report["pipeline_overall"],
                      "api": report["api_overall"]}, indent=2))
    return 0 if not report["partial"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
