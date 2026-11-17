"""Compare PyTorch, OpenVINO, and ONNX CPU detection on a frozen dev manifest.

Each backend/size pair runs in a fresh child process. The tool never downloads or
installs missing dependencies and fails if the requested provider is not active.
"""

from __future__ import annotations

import asyncio
import argparse
import contextvars
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
from pathlib import PureWindowsPath
import random
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np

if __package__:
    from .export_cpu_backend import sha256_file
else:
    from export_cpu_backend import sha256_file


BACKENDS = ("torch", "openvino", "onnx")
SIZES = (640, 960)
MODES = (False, True)
ROUND_MODE_ORDERS = ((False, True), (True, False), (False, True))
MAX_PER_CONFIG_SECONDS = 900
MAX_TOTAL_SECONDS = 5400
_ACTIVE_API_RECORD: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "backend_compare_api_record", default=None
)


def build_measurement_schedule() -> list[dict[str, Any]]:
    """Freeze the 18 serial backend/size/round jobs before any measurement starts."""
    tasks = []
    for global_round in range(3):
        size_order = SIZES if global_round % 2 == 0 else tuple(reversed(SIZES))
        backend_order = BACKENDS[global_round:] + BACKENDS[:global_round]
        for imgsz in size_order:
            for position, backend in enumerate(backend_order):
                config_id = f"{backend}-{imgsz}"
                task_id = f"r{global_round}-{imgsz}-{backend}"
                tasks.append({
                    "task_id": task_id, "task_kind": "measurement", "global_round": global_round,
                    "config_id": config_id, "backend": backend, "imgsz": imgsz,
                    "position": position, "mode_order": list(ROUND_MODE_ORDERS[global_round]),
                })
    return tasks


def build_concurrency_schedule() -> list[dict[str, Any]]:
    return [{"task_id": f"concurrency-{backend}-{imgsz}", "task_kind": "concurrency",
             "global_round": None, "config_id": f"{backend}-{imgsz}",
             "backend": backend, "imgsz": imgsz, "mode_order": list(MODES)}
            for imgsz in SIZES for backend in BACKENDS]


def validate_measurement_schedule(tasks: list[dict[str, Any]]) -> bool:
    expected = {(round_index, backend, imgsz)
                for round_index in range(3) for backend in BACKENDS for imgsz in SIZES}
    actual = {(task.get("global_round"), task.get("backend"), task.get("imgsz"))
              for task in tasks if task.get("task_kind") == "measurement"}
    if len(actual) != 18 or actual != expected or len({task.get("task_id") for task in tasks}) != len(tasks):
        return False
    for size in SIZES:
        positions = {backend: [] for backend in BACKENDS}
        for task in tasks:
            if task["imgsz"] == size and task["task_kind"] == "measurement":
                positions[task["backend"]].append(task["position"])
        if any(sorted(values) != [0, 1, 2] for values in positions.values()):
            return False
    return True


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None}
    ordered = sorted(values)

    def percentile(p: float) -> float:
        index = (len(ordered) - 1) * p
        low, high = math.floor(index), math.ceil(index)
        if low == high:
            return float(ordered[low])
        return float(ordered[low] * (high - index) + ordered[high] * (index - low))

    return {"count": len(ordered), "p50_ms": percentile(0.5), "p95_ms": percentile(0.95)}


def _validate_onnx_bundle_model(model_path: Path) -> None:
    """Parse ONNX without loading external tensors; this repair plan rejects them outright."""
    try:
        import onnx
    except ImportError as exc:
        raise ValueError("ONNX validation requires the pinned backend environment") from exc
    try:
        model = onnx.load_model(str(model_path), load_external_data=False)
    except Exception as exc:
        raise ValueError("registered ONNX model is malformed") from exc
    external_value = onnx.TensorProto.EXTERNAL

    def graph_has_external_data(graph: Any) -> bool:
        tensors = list(graph.initializer)
        for sparse in getattr(graph, "sparse_initializer", []):
            tensors.extend((sparse.values, sparse.indices))
        for node in graph.node:
            for attribute in node.attribute:
                tensors.extend(getattr(attribute, "tensors", []))
                tensor = getattr(attribute, "t", None)
                if tensor is not None:
                    tensors.append(tensor)
                sparse = getattr(attribute, "sparse_tensor", None)
                if sparse is not None:
                    tensors.extend((sparse.values, sparse.indices))
                for sparse in getattr(attribute, "sparse_tensors", []):
                    tensors.extend((sparse.values, sparse.indices))
                child_graphs = ([attribute.g] if getattr(attribute, "g", None) is not None else [])
                child_graphs.extend(getattr(attribute, "graphs", []))
                if any(graph_has_external_data(child) for child in child_graphs):
                    return True
        return any(tensor.data_location == external_value or bool(tensor.external_data) for tensor in tensors)

    if graph_has_external_data(model.graph):
        raise ValueError("external ONNX tensor data is not permitted in a CPU repair bundle")


def verify_artifact_manifest(path: Path, *, backend: str, imgsz: int, source_sha256: str) -> dict[str, Any]:
    manifest_path = path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("backend") != backend:
        raise ValueError("artifact manifest has an unknown schema or wrong backend")
    if manifest.get("source_checkpoint", {}).get("sha256", "").lower() != source_sha256.lower():
        raise ValueError("artifact source checkpoint identity mismatch")
    export = manifest.get("export", {})
    if export.get("imgsz") != imgsz or export.get("batch") != 1 or export.get("quantize") != 32:
        raise ValueError("artifact size/batch/precision configuration mismatch")
    if export.get("dynamic") is not False or export.get("nms") is not False:
        raise ValueError("artifact must use static shapes and shared NMS")
    if manifest.get("precision", {}).get("fp32_verified") is not True:
        raise ValueError("artifact does not have verified FP32 precision")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("artifact file inventory is missing")
    seen: set[str] = set()
    registered: dict[str, Path] = {}
    root = manifest_path.parent.resolve()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("artifact manifest file entry must be an object")
        relative = item.get("path")
        if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                or PureWindowsPath(relative).is_absolute() or PureWindowsPath(relative).drive):
            raise ValueError("artifact manifest path must be relative")
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("artifact manifest path escapes its bundle")
        key = os.path.normcase(resolved.relative_to(root).as_posix())
        if key in seen:
            raise ValueError("duplicate artifact manifest path")
        seen.add(key)
        registered[key] = resolved
        if not resolved.is_file() or resolved.stat().st_size != int(item.get("size_bytes", -1)):
            raise ValueError(f"missing or size-mismatched artifact file: {Path(relative).name}")
        if sha256_file(resolved).lower() != str(item.get("sha256", "")).lower():
            raise ValueError(f"artifact SHA256 mismatch: {Path(relative).name}")
    model_path = manifest.get("model_file")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("model_file must name a registered model")
    resolved_model = (root / model_path).resolve()
    if not resolved_model.is_relative_to(root):
        raise ValueError("model_file escapes its bundle")
    model_key = os.path.normcase(resolved_model.relative_to(root).as_posix())
    if model_key not in seen:
        raise ValueError("model_file is not present in the hashed inventory")
    if backend == "openvino":
        if resolved_model.suffix.lower() != ".xml":
            raise ValueError("OpenVINO model_file must be an XML model")
        if not any(key.endswith(".bin") for key in seen):
            raise ValueError("OpenVINO bundle requires a registered BIN file")
    if backend == "onnx":
        onnx_files = [registered[key] for key in registered if key.endswith(".onnx")]
        if len(onnx_files) != 1 or onnx_files[0] != resolved_model:
            raise ValueError("ONNX bundle must register exactly the declared model file")
        if any(key.endswith(".onnx.data") for key in registered):
            raise ValueError("external ONNX tensor data is not permitted in a CPU repair bundle")
        _validate_onnx_bundle_model(resolved_model)
    artifact_root = manifest.get("artifact_root")
    resolved_root = resolved_model
    if backend == "openvino":
        if (not isinstance(artifact_root, str) or Path(artifact_root).is_absolute()
                or PureWindowsPath(artifact_root).is_absolute() or PureWindowsPath(artifact_root).drive):
            raise ValueError("artifact_root must be a relative bundle directory")
        resolved_root = (root / artifact_root).resolve()
        if not resolved_root.is_relative_to(root) or not resolved_root.is_dir():
            raise ValueError("artifact_root is missing or escapes its bundle")
        if resolved_root != resolved_model.parent:
            raise ValueError("OpenVINO artifact_root must be the registered XML parent directory")
        xml_files = sorted(resolved_root.glob("*.xml"))
        if len(xml_files) != 1 or xml_files[0].resolve() != resolved_model:
            raise ValueError("OpenVINO load directory must contain exactly the registered XML model")
        bin_path = resolved_model.with_suffix(".bin")
        metadata_path = resolved_root / "metadata.yaml"
        for required_path, label in ((bin_path, "BIN"), (metadata_path, "metadata")):
            required_key = os.path.normcase(required_path.resolve().relative_to(root).as_posix())
            if not required_path.is_file() or required_key not in registered:
                raise ValueError(f"OpenVINO {label} file is missing or not registered")
        try:
            import yaml

            metadata_record = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError("OpenVINO metadata.yaml is malformed") from exc
        if not isinstance(metadata_record, dict):
            raise ValueError("OpenVINO metadata.yaml must contain a mapping")
        args = metadata_record.get("args")
        expected_names = {int(k): str(v) for k, v in manifest.get("classes", {}).items()}
        try:
            actual_names = {int(k): str(v) for k, v in metadata_record.get("names", {}).items()}
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("OpenVINO metadata names are invalid") from exc
        expected_imgsz = [imgsz, imgsz]
        if (actual_names != expected_names or metadata_record.get("task") != "detect"
                or metadata_record.get("batch") != 1 or metadata_record.get("imgsz") != expected_imgsz
                or not isinstance(args, dict) or args.get("quantize") != 32
                or args.get("dynamic") is not False or args.get("nms") is not False
                or metadata_record.get("end2end") is not False):
            raise ValueError("OpenVINO metadata conflicts with the source model or export contract")
    file_sha256 = {os.path.normcase((root / item["path"]).resolve().relative_to(root).as_posix()):
                   str(item["sha256"]).lower() for item in files}
    return {"path": manifest_path, "record": manifest, "model_path": resolved_model,
            "load_path": resolved_root if backend == "openvino" else resolved_model,
            "registered_files": registered, "file_sha256": file_sha256,
            "manifest_sha256": sha256_file(manifest_path)}


def _read_config(path: Path, root: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    config = json.loads(data.decode("utf-8-sig"))
    required = {"dataset_root", "performance_manifest", "model_path", "model_sha256", "backend_artifacts",
                "seed", "threads", "warmups", "rounds", "requests_per_client", "budget_per_config_seconds",
                "time_budget_seconds", "busy_timeout_seconds"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"missing config fields: {', '.join(missing)}")
    if config["seed"] != 42 or config["threads"] != 2:
        raise ValueError("O4 fixes seed=42 and threads=2")
    if int(config["warmups"]) != 5 or int(config["rounds"]) != 3:
        raise ValueError("O4 fixes warmups=5 and rounds=3")
    if int(config["requests_per_client"]) != 20:
        raise ValueError("CPU repair fixes API concurrency probes at exactly 20 requests per client")
    per_config = float(config["budget_per_config_seconds"])
    total = float(config["time_budget_seconds"])
    if not 0 < per_config <= MAX_PER_CONFIG_SECONDS or not 0 < total <= MAX_TOTAL_SECONDS:
        raise ValueError("O4 budgets must be <=900 sec per config and <=5400 sec total")
    if set(config["backend_artifacts"]) != {"openvino-640", "openvino-960", "onnx-640", "onnx-960"}:
        raise ValueError("backend_artifacts must name all four exported backend/size pairs")
    for name in ("dataset_root", "performance_manifest", "model_path"):
        value = Path(config[name])
        config[name] = str(value.resolve() if value.is_absolute() else (root / value).resolve())
    if not Path(config["dataset_root"]).is_dir() or not Path(config["performance_manifest"]).is_file():
        raise FileNotFoundError("dataset_root or performance_manifest is missing")
    if not Path(config["model_path"]).is_file() or sha256_file(Path(config["model_path"])) != config["model_sha256"]:
        raise ValueError("source model is missing or its SHA256 changed")
    return config, hashlib.sha256(data).hexdigest()


def _read_samples(config: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    path = Path(config["performance_manifest"])
    samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(samples) != 100:
        raise ValueError(f"O4 requires the fixed 100-row performance manifest; found {len(samples)}")
    if any(row.get("source_split") != "val" or row.get("split") != "development" for row in samples):
        raise ValueError("O4 performance manifest must contain development-only validation samples")
    if len({row.get("group_id") for row in samples}) != len(samples):
        raise ValueError("O4 samples must have unique source groups")
    dataset_root = Path(config["dataset_root"]).resolve()
    for row in samples:
        image = (dataset_root / row["image_path"]).resolve()
        if not image.is_relative_to(dataset_root) or not image.is_file():
            raise ValueError("performance sample path is missing or escapes dataset_root")
        if sha256_file(image).lower() != str(row.get("image_sha256", "")).lower():
            raise ValueError(f"performance sample file SHA changed: {row.get('sample_id')}")
    return samples, sha256_file(path)


def _disable_auto_install_and_pin_runtime_threads(backend: str, threads: int,
                                                artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    from ultralytics.engine import exporter
    from ultralytics.nn.backends import onnx as onnx_backend
    from ultralytics.nn.backends import openvino as openvino_backend
    from ultralytics.utils import checks

    original_check = checks.check_requirements

    def no_install(requirements: Any = None, exclude: Any = (), install: bool = True,
                   cmds: str = "", constrain: Any = ()) -> Any:
        return original_check(requirements, exclude=exclude, install=False, cmds=cmds, constrain=constrain)

    checks.check_requirements = no_install
    exporter.check_requirements = no_install
    onnx_backend.check_requirements = no_install
    openvino_backend.check_requirements = no_install
    capture: dict[str, Any] = {"input_shapes": [], "openvino_reads": []}
    if backend == "onnx":
        import onnxruntime as ort

        original_session = ort.InferenceSession

        def thread_limited_session(path: Any, sess_options: Any = None, providers: Any = None, **kwargs: Any) -> Any:
            options = sess_options or ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            requested = list(providers or ["CPUExecutionProvider"])
            if requested != ["CPUExecutionProvider"]:
                raise RuntimeError(f"refusing non-CPU ONNX provider request: {requested}")
            session = original_session(path, sess_options=options, providers=requested, **kwargs)
            if session.get_providers() != ["CPUExecutionProvider"]:
                raise RuntimeError(f"actual ONNX providers are not CPU-only: {session.get_providers()}")
            return session

        ort.InferenceSession = thread_limited_session
        capture["onnxruntime"] = ort
        capture["original_onnx_session"] = original_session
    elif backend == "openvino":
        import openvino as ov

        original_core = ov.Core

        class LimitedCore:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._core = original_core(*args, **kwargs)

            def __getattr__(self, name: str) -> Any:
                if name == "available_devices":
                    return ["CPU"]
                return getattr(self._core, name)

            def read_model(self, model: Any, weights: Any = None, *args: Any, **kwargs: Any) -> Any:
                model_path = Path(str(model)).resolve()
                weights_path = Path(str(weights)).resolve() if weights is not None else None
                if artifact is not None:
                    expected_model = artifact["model_path"].resolve()
                    expected_weights = expected_model.with_suffix(".bin")
                    if model_path != expected_model or weights_path != expected_weights:
                        raise RuntimeError("OpenVINO consumed model files different from the verified bundle")
                    expected_hashes = artifact["file_sha256"]
                    for consumed in (model_path, expected_weights):
                        key = os.path.normcase(consumed.relative_to(Path(artifact["path"]).parent.resolve()).as_posix())
                        if key not in expected_hashes or sha256_file(consumed).lower() != expected_hashes[key]:
                            raise RuntimeError("OpenVINO model file changed after artifact verification")
                capture["openvino_reads"].append({"model": str(model_path),
                                                   "weights": str(weights_path) if weights_path else None})
                return self._core.read_model(model, weights, *args, **kwargs)

            def compile_model(self, model: Any, device_name: Any = "CPU", config: Any = None, **kwargs: Any) -> Any:
                if str(device_name).upper() != "CPU":
                    raise RuntimeError(f"refusing non-CPU OpenVINO device: {device_name}")
                options = dict(config or {})
                options["INFERENCE_NUM_THREADS"] = threads
                return self._core.compile_model(model, device_name=device_name, config=options, **kwargs)

        ov.Core = LimitedCore
        capture["openvino"] = ov
        capture["original_core"] = original_core

    from ultralytics.engine.predictor import BasePredictor

    original_preprocess = BasePredictor.preprocess

    def capture_preprocess(self: Any, images: Any) -> Any:
        output = original_preprocess(self, images)
        info = {"shape": list(output.shape), "dtype": str(output.dtype), "device": str(output.device)}
        capture["input_shapes"].append(info)
        return output

    BasePredictor.preprocess = capture_preprocess
    capture["original_preprocess"] = original_preprocess
    capture["predictor_class"] = BasePredictor
    return capture


def _restore_patches(backend: str, capture: dict[str, Any]) -> None:
    capture["predictor_class"].preprocess = capture["original_preprocess"]
    if backend == "onnx":
        capture["onnxruntime"].InferenceSession = capture["original_onnx_session"]
    if backend == "openvino":
        capture["openvino"].Core = capture["original_core"]


class _RSSSampler:
    def __init__(self) -> None:
        import threading

        self._threading = threading
        self.stop = threading.Event()
        self.peak_bytes = 0

    def __enter__(self) -> _RSSSampler:
        import psutil

        process = psutil.Process(os.getpid())

        def sample() -> None:
            while not self.stop.is_set():
                try:
                    self.peak_bytes = max(self.peak_bytes, process.memory_info().rss)
                except psutil.Error:
                    pass
                self.stop.wait(0.05)

        self.thread = self._threading.Thread(target=sample, name="backend-rss", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=1)


class _BackendRuntime:
    """Adapt YOLO's formal predictor to the production decoded runtime protocol."""

    SUPPORTED = {"person", "bicycle", "motorcycle", "car", "bus", "truck", "train", "traffic light", "traffic sign"}

    def __init__(self, model: Any, *, backend: str, imgsz: int) -> None:
        from types import SimpleNamespace

        self.model = model
        self.backend = backend
        self.imgsz = imgsz
        self.settings = SimpleNamespace(mode="rules")
        self.versions = {"detector": f"YOLO11n-{backend}-{imgsz}"}
        self.last_objects: list[dict[str, Any]] = []
        self.last_detector_ms = 0.0

    def load(self) -> None:
        return None

    def predict_risk(self, _features: dict[str, Any]) -> str:
        return "low"

    def detect(self, _image_path: str | Path) -> dict[str, Any]:
        raise RuntimeError("O4 requires the decoded-image runtime interface")

    def detect_image(self, image: Any) -> dict[str, Any]:
        from traffic_risk.image_io import DecodedImage

        if not isinstance(image, DecodedImage):
            raise TypeError("detect_image requires DecodedImage")
        start = time.perf_counter_ns()
        result = self.model.predict(source=image.bgr, imgsz=self.imgsz, conf=0.25, iou=0.45,
                                   max_det=300, device="cpu", rect=False, verbose=False)[0]
        self.last_detector_ms = (time.perf_counter_ns() - start) / 1_000_000
        objects = []
        for identifier, box in enumerate(result.boxes):
            category = str(result.names[int(box.cls.item())]).lower().replace("_", " ")
            if category not in self.SUPPORTED:
                continue
            coordinates = box.xyxy.cpu().numpy().reshape(-1).tolist()
            objects.append({
                "id": identifier,
                "category": category,
                "confidence": float(box.conf.item()),
                "attributes": {"trafficLightColor": "unknown"},
                "box2d": dict(zip(("x1", "y1", "x2", "y2"), map(float, coordinates), strict=True)),
            })
        self.last_objects = objects
        return {"frames": [{"timestamp": 0, "objects": objects}]}


class _TimedASGI:
    def __init__(self, application: Any, records: dict[str, dict[str, Any]]) -> None:
        self.application = application
        self.records = records

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") != "/api/predict":
            await self.application(scope, receive, send)
            return
        request_id = dict(scope.get("headers", [])).get(b"x-profile-id", b"unknown").decode("ascii", "replace")
        record: dict[str, Any] = {}
        self.records[request_id] = record
        token = _ACTIVE_API_RECORD.set(record)
        started = time.perf_counter_ns()
        try:
            async def timed_send(message: dict[str, Any]) -> None:
                if message.get("type") == "http.response.start":
                    record["http_status"] = int(message["status"])
                await send(message)

            await self.application(scope, receive, timed_send)
        finally:
            record["api_server_ms"] = (time.perf_counter_ns() - started) / 1_000_000
            _ACTIVE_API_RECORD.reset(token)


def _backend_provider(runtime: _BackendRuntime, backend: str) -> dict[str, Any]:
    predictor = runtime.model.predictor
    auto = predictor.model
    inner = getattr(auto, "backend", None)
    if backend == "torch":
        parameter = next(runtime.model.model.parameters())
        return {"backend": "PyTorch", "device": str(parameter.device), "dtype": str(parameter.dtype),
                "provider": "CPU"}
    if backend == "onnx":
        session = getattr(inner, "session", None)
        if session is None:
            raise RuntimeError("ONNX Runtime session was not initialized")
        providers = session.get_providers()
        if providers != ["CPUExecutionProvider"]:
            raise RuntimeError(f"ONNX Runtime did not use CPU only: {providers}")
        input_ = session.get_inputs()[0]
        if input_.type != "tensor(float)":
            raise RuntimeError(f"ONNX runtime input is not FP32: {input_.type}")
        return {"backend": "ONNX Runtime", "providers": providers,
                "input_type": input_.type, "input_shape": input_.shape,
                "intra_op_threads": 2, "inter_op_threads": 1}
    compiled = getattr(inner, "ov_compiled_model", None)
    if compiled is None:
        raise RuntimeError("OpenVINO compiled model was not initialized")
    devices = list(compiled.get_property("EXECUTION_DEVICES"))
    if not devices or any(str(device).upper() != "CPU" for device in devices):
        raise RuntimeError(f"OpenVINO did not execute exclusively on CPU: {devices}")
    input_port = compiled.input()
    dtype = input_port.get_element_type().get_type_name()
    if dtype != "f32":
        raise RuntimeError(f"OpenVINO runtime input is not FP32: {dtype}")
    shape = list(input_port.get_shape())
    return {"backend": "OpenVINO", "execution_devices": devices,
            "input_type": dtype, "input_shape": shape, "inference_threads": 2}


def _normalize_config_output(response: dict[str, Any]) -> dict[str, Any]:
    return {key: response.get(key) for key in ("status", "risk", "reason_code", "reason", "features", "image_size", "detections")}


def _stats(rows: list[dict[str, Any]], mode: str, kind: str) -> dict[str, Any]:
    items = [row for row in rows if row.get("kind") == kind and row.get("include_detections") is mode]
    if kind == "api":
        formal = [row for row in items if row.get("phase") == "sequential"]
        success = [row for row in formal if row.get("http_status") == 200 and not row.get("pipeline_failure")]
        round_ids = sorted({row.get("global_round", row.get("round")) for row in formal
                            if row.get("global_round", row.get("round")) is not None})
        return {**latency_summary([float(row["elapsed_ms"]) for row in success]),
                "formal_requests": len(formal),
                "requests": len(items), "success": len(success),
                "busy_503": sum(row.get("http_status") == 503 for row in items),
                "other_failure": sum(row.get("http_status") not in {200, 503} for row in items),
                "pipeline_failures": sum(row.get("pipeline_failure", False) for row in items),
                "rounds": [latency_summary([float(row["elapsed_ms"]) for row in success
                                             if row.get("global_round", row.get("round")) == index])
                           for index in round_ids]}
    formal = [row for row in items if row.get("phase", "measurement") == "measurement"]
    round_ids = sorted({row.get("global_round", row.get("round")) for row in formal
                        if row.get("global_round", row.get("round")) is not None})
    return {**latency_summary([float(row["elapsed_ms"]) for row in formal]),
            "formal_requests": len(formal), "warmup_requests": sum(row.get("phase") == "warmup" for row in items),
            "failures": sum(row.get("pipeline_failure", False) for row in items),
            "rounds": [latency_summary([float(row["elapsed_ms"]) for row in formal
                                         if row.get("global_round", row.get("round")) == index]) for index in round_ids]}


def _api_completeness(rows: list[dict[str, Any]], config: dict[str, Any], sample_count: int,
                      sample_ids: list[str] | None = None) -> dict[str, Any]:
    rounds = len(ROUND_MODE_ORDERS)
    requests_per_client = int(config["requests_per_client"])
    warmups_per_mode = int(config["warmups"]) * (rounds + 1)
    sequential_per_mode = rounds * sample_count
    concurrency_per_mode = sum((1, 2, 4)) * requests_per_client
    expected_rows = len(MODES) * (warmups_per_mode + sequential_per_mode + concurrency_per_mode)
    expected_warmups = len(MODES) * warmups_per_mode
    expected_sequential = len(MODES) * sequential_per_mode
    expected_concurrency = len(MODES) * concurrency_per_mode
    all_tasks = build_measurement_schedule() + build_concurrency_schedule()
    observed_configs = {row.get("config_id") for row in rows if row.get("config_id") is not None}
    tasks = [task for task in all_tasks if not observed_configs or task["config_id"] in observed_configs]
    expected_task_ids = {task["task_id"] for task in tasks}
    task_by_id = {task["task_id"]: task for task in tasks}
    task_ids_valid = (len(rows) == sum(row.get("task_id") is not None for row in rows)
                      and all(row.get("task_id") in expected_task_ids
                              and row.get("config_id") == task_by_id[row["task_id"]]["config_id"]
                              and row.get("task_kind") == task_by_id[row["task_id"]]["task_kind"]
                              and row.get("global_round") == task_by_id[row["task_id"]]["global_round"]
                              and row.get("include_detections") in task_by_id[row["task_id"]]["mode_order"]
                              for row in rows))
    phases = ("warmup", "sequential", "concurrency-1", "concurrency-2", "concurrency-4")
    phase_counts = {phase: sum(row.get("phase") == phase for row in rows) for phase in phases}
    expected_sample_ids = sample_ids or [f"sample-{index}" for index in range(sample_count)]
    if len(expected_sample_ids) != sample_count or len(set(expected_sample_ids)) != sample_count:
        raise ValueError("expected sample IDs must be unique and match sample_count")
    expected_warmup_keys = {
        (task["task_id"], include, task["global_round"], index)
        for task in tasks for include in task["mode_order"] for index in range(int(config["warmups"]))
    }
    warmup_keys = {(row.get("task_id"), row.get("include_detections"), row.get("global_round"),
                    row.get("request_index")) for row in rows if row.get("phase") == "warmup"}
    expected_sequential_keys = {
        (task["task_id"], include, task["global_round"], sample_id)
        for task in tasks if task["task_kind"] == "measurement" for include in task["mode_order"]
        for sample_id in expected_sample_ids
    }
    sequential_keys = {(row.get("task_id"), row.get("include_detections"), row.get("global_round"), row.get("sample_id"))
                       for row in rows if row.get("phase") == "sequential"}
    expected_concurrency_keys = {
        (task["task_id"], include, task["global_round"], f"concurrency-{clients}", index)
        for task in tasks if task["task_kind"] == "concurrency" for include in task["mode_order"]
        for clients in (1, 2, 4) for index in range(clients * requests_per_client)
    }
    concurrency_keys = {
        (row.get("task_id"), row.get("include_detections"), row.get("global_round"),
         row.get("phase"), row.get("request_index"))
        for row in rows if str(row.get("phase", "")).startswith("concurrency-")
    }
    sequential_complete = phase_counts["sequential"] == expected_sequential and sequential_keys == expected_sequential_keys
    sequential_acceptable = all(
        row.get("http_status") == 200 and not row.get("pipeline_failure", False)
        for row in rows if row.get("phase") == "sequential"
    )
    warmups_complete = phase_counts["warmup"] == expected_warmups and warmup_keys == expected_warmup_keys
    concurrency_complete = (
        sum(phase_counts[f"concurrency-{n}"] for n in (1, 2, 4)) == expected_concurrency
        and concurrency_keys == expected_concurrency_keys
    )
    warmups_acceptable = all(row.get("http_status") == 200 and not row.get("pipeline_failure", False)
                             for row in rows if row.get("phase") == "warmup")
    concurrency_acceptable = all(
        row.get("http_status") in {200, 503} and not row.get("pipeline_failure", False)
        for row in rows if str(row.get("phase", "")).startswith("concurrency-")
    )
    complete = len(rows) == expected_rows and task_ids_valid and warmups_complete and sequential_complete and concurrency_complete
    return {
        "api_expected_rows": expected_rows, "api_rows": len(rows),
        "api_expected_warmup_rows": expected_warmups,
        "api_expected_sequential_rows": expected_sequential,
        "api_expected_concurrency_rows": expected_concurrency,
        "api_phase_rows": phase_counts,
        "sequential_api_expected_success_per_mode": sequential_per_mode,
        "sequential_api_600_complete_per_mode": sequential_complete,
        "sequential_api_600_success_per_mode": sequential_complete and sequential_acceptable,
        "warmups_complete": warmups_complete, "warmups_acceptable": warmups_acceptable,
        "concurrency_complete": concurrency_complete, "concurrency_acceptable": concurrency_acceptable,
        "task_ids_valid": task_ids_valid,
        "api_complete": complete,
        "api_acceptable": complete and sequential_acceptable and warmups_acceptable and concurrency_acceptable,
    }


def _task_api_completeness(rows: list[dict[str, Any]], config: dict[str, Any], task: dict[str, Any],
                           sample_ids: list[str]) -> dict[str, Any]:
    expected_warmup = {(include, task.get("global_round"), index) for include in task["mode_order"]
                       for index in range(int(config["warmups"]))}
    actual_warmup = {(row.get("include_detections"), row.get("global_round"), row.get("request_index"))
                     for row in rows if row.get("phase") == "warmup"}
    if task["task_kind"] == "measurement":
        expected_work = {(include, task["global_round"], sample_id)
                         for include in task["mode_order"] for sample_id in sample_ids}
        work_rows = [row for row in rows if row.get("phase") == "sequential"]
        actual_work = {(row.get("include_detections"), row.get("global_round"), row.get("sample_id"))
                       for row in work_rows}
        work_acceptable = all(row.get("http_status") == 200 and not row.get("pipeline_failure", False)
                              for row in work_rows)
    else:
        expected_work = {(include, f"concurrency-{clients}", index)
                         for include in task["mode_order"] for clients in (1, 2, 4)
                         for index in range(clients * int(config["requests_per_client"]))}
        work_rows = [row for row in rows if str(row.get("phase", "")).startswith("concurrency-")]
        actual_work = {(row.get("include_detections"), row.get("phase"), row.get("request_index"))
                       for row in work_rows}
        work_acceptable = all(row.get("http_status") in {200, 503} and not row.get("pipeline_failure", False)
                              for row in work_rows)
    work_complete = actual_work == expected_work and len(work_rows) == len(expected_work)
    warmup_complete = actual_warmup == expected_warmup and sum(
        row.get("phase") == "warmup" for row in rows
    ) == len(expected_warmup)
    warmup_acceptable = all(row.get("http_status") == 200 and not row.get("pipeline_failure", False)
                            for row in rows if row.get("phase") == "warmup")
    return {"warmup_complete": warmup_complete, "work_complete": work_complete,
            "warmup_acceptable": warmup_acceptable, "work_acceptable": work_acceptable,
            "complete": warmup_complete and work_complete,
            "acceptable": warmup_complete and work_complete and warmup_acceptable and work_acceptable}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _api_settings(config: dict[str, Any], output: Path) -> Any:
    from traffic_risk.config import Settings

    return Settings(mode="rules", model_dir=Path(config["model_path"]).parent.parent,
                    upload_dir=output / "uploads", device="cpu", inference_concurrency=1,
                    busy_timeout_seconds=float(config["busy_timeout_seconds"]))


async def _profile_api(runtime: _BackendRuntime, config: dict[str, Any], samples: list[dict[str, Any]],
                       output: Path, deadline: float, task: dict[str, Any],
                       on_row: Any = None) -> list[dict[str, Any]]:
    import httpx
    from traffic_risk.app import create_app

    settings = _api_settings(config, output)
    application = create_app(settings, runtime_factory=lambda _settings: runtime)
    rows: list[dict[str, Any]] = []
    trace_records: dict[str, dict[str, Any]] = {}
    async with application.router.lifespan_context(application):
        gate = application.state.inference_gate
        original_acquire = gate.acquire

        async def timed_acquire() -> bool:
            record = _ACTIVE_API_RECORD.get()
            started = time.perf_counter_ns()
            try:
                return await original_acquire()
            finally:
                if record is not None:
                    record["gate_wait_ms"] = (time.perf_counter_ns() - started) / 1_000_000

        gate.acquire = timed_acquire
        transport = httpx.ASGITransport(app=_TimedASGI(application, trace_records))
        async with httpx.AsyncClient(transport=transport, base_url="http://backend-compare.local") as client:
            ready = await client.get("/ready")
            if ready.status_code != 200:
                raise RuntimeError(f"backend API readiness failed with HTTP {ready.status_code}")

            async def request(sample: dict[str, Any], request_id: str, include: bool,
                              phase: str, round_index: int | None, request_index: int) -> dict[str, Any]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("per-configuration backend time budget reached")
                path = Path(config["dataset_root"]) / sample["image_path"]
                data = path.read_bytes()
                mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                        ".webp": "image/webp"}.get(path.suffix.lower(), "application/octet-stream")
                started = time.perf_counter_ns()
                remaining = deadline - time.monotonic()
                try:
                    response = await asyncio.wait_for(client.post(
                        "/api/predict", params={"include_detections": str(include).lower()},
                        headers={"x-profile-id": request_id},
                        files={"file": ("profile" + path.suffix.lower(), data, mime)},
                    ), timeout=remaining)
                except TimeoutError as exc:
                    trace_records.pop(request_id, None)
                    raise TimeoutError("per-configuration time budget reached during API request") from exc
                except BaseException:
                    trace_records.pop(request_id, None)
                    raise
                elapsed = (time.perf_counter_ns() - started) / 1_000_000
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}
                trace = trace_records.pop(request_id, {})
                from traffic_risk.inference_outcome import is_processing_failure
                row = {"kind": "api", "task_kind": task["task_kind"], "task_id": task["task_id"],
                       "config_id": task["config_id"], "global_round": round_index,
                       "phase": phase, "round": round_index, "request_index": request_index,
                       "sample_id": sample["sample_id"], "include_detections": include,
                       "elapsed_ms": elapsed, "http_status": response.status_code,
                       "status": payload.get("status"), "risk": payload.get("risk"),
                       "reason_code": payload.get("reason_code"),
                       "pipeline_failure": (response.status_code == 200
                                            and is_processing_failure(payload.get("reason_code"))),
                       "gate_wait_ms": trace.get("gate_wait_ms"),
                       "api_server_ms": trace.get("api_server_ms")}
                rows.append(row)
                if on_row is not None:
                    on_row(row)
                return row

            try:
                for include in task["mode_order"]:
                    for index in range(int(config["warmups"])):
                        await request(samples[index % len(samples)], f"{task['task_id']}-warm-{include}-{index}",
                                      include, "warmup", task.get("global_round"), index)
                    if task["task_kind"] == "measurement":
                        for index, sample in enumerate(samples):
                            await request(sample, f"{task['task_id']}-seq-{include}-{index}",
                                          include, "sequential", task["global_round"], index)
                    else:
                        for clients in (1, 2, 4):
                            await _concurrency_probe(
                                request, samples, include, clients, int(config["requests_per_client"]), task
                            )
            except TimeoutError:
                pass
    return rows


async def _concurrency_probe(request: Any, samples: list[dict[str, Any]], include: bool,
                             clients: int, requests_per_client: int,
                             task: dict[str, Any]) -> None:
    cursor = 0
    lock = asyncio.Lock()

    async def worker(worker_id: int) -> None:
        nonlocal cursor
        for request_index in range(requests_per_client):
            async with lock:
                sample = samples[cursor % len(samples)]
                serial = cursor
                cursor += 1
            await request(sample, f"{task['task_id']}-c{clients}-w{worker_id}-{request_index}-{serial}",
                          include, f"concurrency-{clients}", None, serial)

    tasks = [asyncio.create_task(worker(index)) for index in range(clients)]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        pending = [pending_task for pending_task in tasks if not pending_task.done()]
        for pending_task in pending:
            pending_task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        raise


def _runtime_packages(backend: str) -> dict[str, str]:
    packages = {"ultralytics": "8.4.126", "torch": "2.13.0+cpu", "numpy": "2.3.5"}
    if backend == "openvino":
        packages["openvino"] = "2025.4.1"
    elif backend == "onnx":
        packages.update({"onnx": "1.23.0", "onnxruntime": "1.32.0"})
    return packages


def _load_model(backend: str, imgsz: int, artifact: dict[str, Any] | None,
                config: dict[str, Any]) -> Any:
    from ultralytics import YOLO

    if backend == "torch":
        return YOLO(config["model_path"])
    if artifact is None:
        raise ValueError("exported backend artifacts are required")
    # Repeat the complete check at the load boundary and ensure the verified
    # directory/XML binding has not changed since the caller's preflight.
    current = verify_artifact_manifest(Path(artifact["path"]), backend=backend, imgsz=imgsz,
                                       source_sha256=config["model_sha256"])
    if (current["manifest_sha256"] != artifact.get("manifest_sha256")
            or current["model_path"] != artifact.get("model_path")
            or current["load_path"] != artifact.get("load_path")):
        raise ValueError("artifact binding changed after verification")
    if int(artifact["record"]["export"]["imgsz"]) != imgsz:
        raise ValueError("exported backend image size mismatch")
    load_path = current["load_path"]
    model = YOLO(str(load_path))
    return model


def _worker(config: dict[str, Any], backend: str, imgsz: int,
            artifact_path: str | None, samples: list[dict[str, Any]], manifest_sha: str,
            output: Path) -> dict[str, Any]:
    import cv2
    import torch
    from traffic_risk.config import Settings
    from traffic_risk.model_runtime import ModelRuntime
    from traffic_risk.pipeline import run_single_image
    import traffic_risk
    torch.manual_seed(int(config["seed"]))
    torch.set_num_threads(int(config["threads"]))
    torch.set_num_interop_threads(int(config["threads"]))
    cv2.setNumThreads(int(config["threads"]))
    np.random.seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    os.environ["OMP_NUM_THREADS"] = str(config["threads"])
    os.environ["MKL_NUM_THREADS"] = str(config["threads"])
    os.environ["OPENBLAS_NUM_THREADS"] = str(config["threads"])
    installed = {}
    for name, required in _runtime_packages(backend).items():
        actual = metadata.version(name)
        if actual != required:
            raise RuntimeError(f"O4 requires {name}=={required}; found {actual}")
        installed[name] = actual

    # Verify the public source bundle through the normal loader before using its hash-pinned checkpoint.
    formal = ModelRuntime(Settings(mode="rules", model_dir=Path(config["model_path"]).parent.parent, device="cpu"))
    formal.load()
    if sha256_file(Path(config["model_path"])) != config["model_sha256"]:
        raise RuntimeError("source checkpoint changed after formal loader validation")
    del formal

    started = time.perf_counter()
    artifact = verify_artifact_manifest(Path(artifact_path), backend=backend, imgsz=imgsz,
                                        source_sha256=config["model_sha256"]) if artifact_path else None
    capture = _disable_auto_install_and_pin_runtime_threads(backend, int(config["threads"]), artifact)
    try:
        load_started = time.perf_counter_ns()
        model = _load_model(backend, imgsz, artifact, config)
        load_ms = (time.perf_counter_ns() - load_started) / 1_000_000
        runtime = _BackendRuntime(model, backend=backend, imgsz=imgsz)
        output.mkdir(parents=True, exist_ok=False)
        deadline = time.monotonic() + float(config["budget_per_config_seconds"])
        request_rows: list[dict[str, Any]] = []
        parity_rows: dict[tuple[str, bool], dict[str, Any]] = {}
        detections: dict[str, list[dict[str, Any]]] = {}
        pipeline_partial = False

        with _RSSSampler() as rss:
            for include in MODES:
                for index in range(int(config["warmups"])):
                    sample = samples[index % len(samples)]
                    run_single_image(Path(config["dataset_root"]) / sample["image_path"], runtime,
                                     include_detections=include)
                for round_index in range(int(config["rounds"])):
                    for sample in samples:
                        if time.monotonic() >= deadline:
                            pipeline_partial = True
                            break
                        image_path = Path(config["dataset_root"]) / sample["image_path"]
                        started_ns = time.perf_counter_ns()
                        response = run_single_image(image_path, runtime, include_detections=include)
                        elapsed = (time.perf_counter_ns() - started_ns) / 1_000_000
                        failure = str(response.get("reason_code")) in {
                            "feature_schema_error", "perception_error", "invalid_features",
                            "invalid_model_output", "legacy_unspecified"}
                        row = {"kind": "pipeline", "phase": "measurement", "round": round_index,
                               "sample_id": sample["sample_id"], "include_detections": include,
                               "elapsed_ms": elapsed, "detector_ms": runtime.last_detector_ms,
                               "pipeline_failure": failure, "reason_code": response.get("reason_code"),
                               "status": response.get("status"), "risk": response.get("risk")}
                        request_rows.append(row)
                        if round_index == 0:
                            parity_rows[(sample["sample_id"], include)] = _normalize_config_output(response)
                            if include is False:
                                detections[sample["sample_id"]] = runtime.last_objects
                    if pipeline_partial:
                        break
                if pipeline_partial:
                    break
            api_rows: list[dict[str, Any]] = []
            if not pipeline_partial:
                api_rows = asyncio.run(_profile_api(runtime, config, samples, output, deadline))
                request_rows.extend(api_rows)

        provider = _backend_provider(runtime, backend)
        prep_shapes = capture.get("input_shapes", [])
        if not prep_shapes:
            raise RuntimeError("no actual preprocessed input shape was captured")
        unique_shapes = {json.dumps(item, sort_keys=True) for item in prep_shapes}
        if any(json.loads(item)["shape"] != [1, 3, imgsz, imgsz] for item in unique_shapes):
            raise RuntimeError("predictor did not use the required square input shape")
        classes = {str(key): str(value) for key, value in model.names.items()}
        if backend != "torch" and artifact["record"]["classes"] != classes:
            raise RuntimeError("backend class mapping differs from exported manifest")
        parity_output = output / "parity_outputs.jsonl"
        parity_output.write_text("".join(json.dumps({"sample_id": sid, "include_detections": include,
            "response": response, "objects": detections.get(sid, []) if include is False else None},
            ensure_ascii=False, sort_keys=True) + "\n" for (sid, include), response in sorted(parity_rows.items())),
            encoding="utf-8")
        _write_csv(output / "requests.csv", request_rows)
        api_completeness = _api_completeness(api_rows, config, len(samples))
        api_complete = api_completeness["api_complete"]
        pipeline_complete = all(_stats(request_rows, mode, "pipeline")["count"] == 300 for mode in MODES)
        worker_complete = not pipeline_partial and pipeline_complete and api_complete
        summary = {
            "schema_version": 1, "status": "complete" if worker_complete else "partial",
            "completeness": {"pipeline_300_per_mode": pipeline_complete, **api_completeness},
            "backend": backend, "imgsz": imgsz, "rect": False, "actual_input_shapes": sorted(unique_shapes),
            "requested_device": "cpu", "actual_provider": provider,
            "source_checkpoint_sha256": config["model_sha256"], "performance_manifest_sha256": manifest_sha,
            "sample_count": len(samples), "warmups_per_path": config["warmups"], "rounds": config["rounds"],
            "thread_config": {"torch_intra_op": torch.get_num_threads(), "torch_inter_op": torch.get_num_interop_threads(),
                              "opencv": cv2.getNumThreads(), "openvino_or_ort": 2},
            "export_manifest_sha256": artifact["manifest_sha256"] if artifact else None,
            "artifact_load_path": str(artifact["load_path"]) if artifact else None,
            "artifact_model_path": str(artifact["model_path"]) if artifact else None,
            "artifact_consumed_files": capture.get("openvino_reads", []),
            "libraries": installed, "source_version": traffic_risk.__version__,
            "model_load_ms": load_ms, "peak_process_rss_bytes": rss.peak_bytes, "rss_sampling_interval_ms": 50,
            "pipeline_false": {**_stats(request_rows, False, "pipeline"),
                               "median_round_p95_ms": statistics.median(
                                   [r["p95_ms"] for r in _stats(request_rows, False, "pipeline")["rounds"]
                                    if r["p95_ms"] is not None]) if _stats(request_rows, False, "pipeline")["rounds"] else None},
            "pipeline_true": {**_stats(request_rows, True, "pipeline"),
                              "median_round_p95_ms": statistics.median(
                                  [r["p95_ms"] for r in _stats(request_rows, True, "pipeline")["rounds"]
                                   if r["p95_ms"] is not None]) if _stats(request_rows, True, "pipeline")["rounds"] else None},
            "api_false": {**_stats(request_rows, False, "api"),
                          "median_round_p95_ms": statistics.median([r["p95_ms"] for r in _stats(request_rows, False, "api")["rounds"]
                                                                   if r["p95_ms"] is not None]) if _stats(request_rows, False, "api")["rounds"] else None},
            "api_true": {**_stats(request_rows, True, "api"),
                         "median_round_p95_ms": statistics.median([r["p95_ms"] for r in _stats(request_rows, True, "api")["rounds"]
                                                                  if r["p95_ms"] is not None]) if _stats(request_rows, True, "api")["rounds"] else None},
            "pipeline_failures": sum(row.get("pipeline_failure") is True for row in request_rows if row.get("kind") == "pipeline"),
            "api_failures": {"503": sum(row.get("http_status") == 503 for row in api_rows),
                             "other": sum(row.get("http_status") not in {200, 503} for row in api_rows)},
            "measurement_code_sha256": sha256_file(Path(__file__).resolve()),
            "config_sha256": config["config_sha256"], "worker_elapsed_seconds": time.perf_counter() - started,
            "preprocessing_note": "all six configurations use YOLO predictor rect=False and actual BCHW input [1,3,imgsz,imgsz]; production O2 predictor rect=True is a separate preprocessing path",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                                             encoding="utf-8")
        return summary
    finally:
        _restore_patches(backend, capture)


def _worker_task(config: dict[str, Any], task: dict[str, Any], artifact_path: str | None,
                 samples: list[dict[str, Any]], manifest_sha: str, output: Path) -> dict[str, Any]:
    """Run exactly one frozen measurement round or one concurrency task in a fresh process."""
    import cv2
    import torch
    from traffic_risk.config import Settings
    from traffic_risk.model_runtime import ModelRuntime
    from traffic_risk.pipeline import run_single_image

    backend, imgsz = task["backend"], int(task["imgsz"])
    task_kind = task["task_kind"]
    budget = min(float(config["budget_per_config_seconds"]), float(task["budget_seconds"]))
    deadline = time.monotonic() + budget
    torch.manual_seed(int(config["seed"]))
    torch.set_num_threads(int(config["threads"]))
    torch.set_num_interop_threads(int(config["threads"]))
    cv2.setNumThreads(int(config["threads"]))
    np.random.seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    os.environ["OMP_NUM_THREADS"] = str(config["threads"])
    os.environ["MKL_NUM_THREADS"] = str(config["threads"])
    os.environ["OPENBLAS_NUM_THREADS"] = str(config["threads"])
    installed = {}
    for name, required in _runtime_packages(backend).items():
        actual = metadata.version(name)
        if actual != required:
            raise RuntimeError(f"CPU repair requires {name}=={required}; found {actual}")
        installed[name] = actual
    formal = ModelRuntime(Settings(mode="rules", model_dir=Path(config["model_path"]).parent.parent, device="cpu"))
    formal.load()
    if sha256_file(Path(config["model_path"])) != config["model_sha256"]:
        raise RuntimeError("source checkpoint changed after formal loader validation")
    del formal

    started = time.monotonic()
    artifact = verify_artifact_manifest(Path(artifact_path), backend=backend, imgsz=imgsz,
                                        source_sha256=config["model_sha256"]) if artifact_path else None
    capture = _disable_auto_install_and_pin_runtime_threads(backend, int(config["threads"]), artifact)
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    parity_rows: dict[tuple[str, bool], dict[str, Any]] = {}
    detections: dict[str, list[dict[str, Any]]] = {}
    partial_reason = None
    task_deadline = deadline

    def append_row(row: dict[str, Any]) -> None:
        row.setdefault("config_id", task["config_id"])
        row.setdefault("task_id", task["task_id"])
        row.setdefault("task_kind", task_kind)
        row.setdefault("global_round", task["global_round"])
        row.setdefault("config_sha256", config["config_sha256"])
        row.setdefault("performance_manifest_sha256", manifest_sha)
        row.setdefault("source_checkpoint_sha256", config["model_sha256"])
        row.setdefault("artifact_manifest_sha256", artifact.get("manifest_sha256") if artifact else None)
        rows.append(row)
        with (output / "rows.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()

    rss_all = _RSSSampler()
    try:
        rss_all.__enter__()
        load_started = time.perf_counter_ns()
        model = _load_model(backend, imgsz, artifact, config)
        load_ms = (time.perf_counter_ns() - load_started) / 1_000_000
        runtime = _BackendRuntime(model, backend=backend, imgsz=imgsz)
        with _RSSSampler() as rss:
            if task_kind == "measurement":
                for include in task["mode_order"]:
                    for warm_index in range(int(config["warmups"])):
                        if time.monotonic() >= task_deadline:
                            partial_reason = "budget_exhausted_pipeline_warmup"
                            break
                        sample = samples[warm_index % len(samples)]
                        start_ns = time.perf_counter_ns()
                        _ = run_single_image(Path(config["dataset_root"]) / sample["image_path"],
                                             runtime, include_detections=include)
                        append_row({"kind": "pipeline", "task_kind": task_kind, "task_id": task["task_id"],
                                    "config_id": task["config_id"], "global_round": task["global_round"],
                                    "phase": "warmup", "request_index": warm_index,
                                    "sample_id": sample["sample_id"], "include_detections": include,
                                    "elapsed_ms": (time.perf_counter_ns() - start_ns) / 1_000_000})
                    if partial_reason:
                        break
                    for sample_index, sample in enumerate(samples):
                        if time.monotonic() >= task_deadline:
                            partial_reason = "budget_exhausted_pipeline_measurement"
                            break
                        image_path = Path(config["dataset_root"]) / sample["image_path"]
                        start_ns = time.perf_counter_ns()
                        response = run_single_image(image_path, runtime, include_detections=include)
                        elapsed = (time.perf_counter_ns() - start_ns) / 1_000_000
                        from traffic_risk.evaluate_detection import pipeline_failed
                        row = {"kind": "pipeline", "task_kind": task_kind, "task_id": task["task_id"],
                               "config_id": task["config_id"], "global_round": task["global_round"],
                               "round": task["global_round"], "phase": "measurement",
                               "request_index": sample_index, "sample_id": sample["sample_id"],
                               "include_detections": include, "elapsed_ms": elapsed,
                               "detector_ms": runtime.last_detector_ms,
                               "pipeline_failure": pipeline_failed(response),
                               "reason_code": response.get("reason_code"), "status": response.get("status"),
                               "risk": response.get("risk")}
                        append_row(row)
                        normalized = _normalize_config_output(response)
                        objects = runtime.last_objects
                        round_record = {"sample_id": sample["sample_id"], "include_detections": include,
                                        "global_round": task["global_round"], "response": normalized,
                                        "objects": objects}
                        with (output / "round_outputs.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(round_record, ensure_ascii=False, sort_keys=True) + "\n")
                            stream.flush()
                        if task["global_round"] == 0:
                            parity_rows[(sample["sample_id"], include)] = normalized
                            detections[sample["sample_id"]] = objects
                    if partial_reason:
                        break
        def append_api_row(row: dict[str, Any]) -> None:
            row.update({"task_kind": task_kind, "task_id": task["task_id"],
                        "config_id": task["config_id"]})
            append_row(row)

        asyncio.run(_profile_api(runtime, config, samples, output, task_deadline, task, append_api_row))
        if time.monotonic() >= task_deadline:
            partial_reason = partial_reason or "budget_exhausted_api"

        if task["global_round"] == 0:
            (output / "parity_outputs.jsonl").write_text("".join(
                json.dumps({"sample_id": sid, "include_detections": include, "response": response,
                            "objects": detections.get(sid, []) if include is False else None},
                           ensure_ascii=False, sort_keys=True) + "\n"
                for (sid, include), response in sorted(parity_rows.items())), encoding="utf-8")
        provider = _backend_provider(runtime, backend)
        shapes = capture.get("input_shapes", [])
        expected_shape = [1, 3, imgsz, imgsz]
        actual_shapes = sorted({json.dumps(item, sort_keys=True) for item in shapes})
        shape_valid = bool(actual_shapes) and all(json.loads(item).get("shape") == expected_shape
                                                  for item in actual_shapes)
        if not shape_valid:
            raise RuntimeError("task did not observe the expected square FP32 predictor input")
        task_api_completeness = _task_api_completeness(
            [row for row in rows if row["kind"] == "api"], config, task,
            [str(row["sample_id"]) for row in samples],
        )
        pipeline_rows = [row for row in rows if row["kind"] == "pipeline"]
        pipeline_complete = True
        if task_kind == "measurement":
            expected_pipeline_keys = {
                (include, "warmup", index, samples[index % len(samples)]["sample_id"])
                for include in task["mode_order"] for index in range(int(config["warmups"]))
            } | {
                (include, "measurement", index, sample["sample_id"])
                for include in task["mode_order"] for index, sample in enumerate(samples)
            }
            actual_pipeline_keys = {
                (row.get("include_detections"), row.get("phase"), row.get("request_index"), row.get("sample_id"))
                for row in pipeline_rows
            }
            pipeline_complete = actual_pipeline_keys == expected_pipeline_keys and len(pipeline_rows) == len(expected_pipeline_keys)
        pipeline_counts = len(pipeline_rows)
        task_complete = (partial_reason is None and pipeline_complete and task_api_completeness["complete"])
        summary = {
            "schema_version": 2, "status": "complete" if task_complete else "partial",
            "task": task, "partial_reason": partial_reason,
            "config_id": task["config_id"], "task_id": task["task_id"], "task_kind": task_kind,
            "backend": backend, "imgsz": imgsz, "actual_input_shapes": [json.loads(item) for item in actual_shapes],
            "actual_provider": provider, "source_checkpoint_sha256": config["model_sha256"],
            "performance_manifest_sha256": manifest_sha, "config_sha256": config["config_sha256"],
            "artifact_manifest_sha256": artifact["manifest_sha256"] if artifact else None,
            "artifact_load_path": str(artifact["load_path"]) if artifact else None,
            "artifact_model_path": str(artifact["model_path"]) if artifact else None,
            "artifact_consumed_files": capture.get("openvino_reads", []),
            "libraries": installed, "model_load_ms": load_ms,
            "peak_process_rss_bytes": max(rss.peak_bytes, rss_all.peak_bytes),
            "pipeline_rows": pipeline_counts,
            "task_api_completeness": task_api_completeness,
            "pipeline_completeness": {"complete": pipeline_complete, "rows": pipeline_counts},
            "task_elapsed_seconds": time.monotonic() - started,
            "measurement_tool_sha256": sha256_file(Path(__file__).resolve()),
            "rows_sha256": sha256_file(output / "rows.jsonl") if (output / "rows.jsonl").is_file() else None,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                                             encoding="utf-8")
        return summary
    finally:
        rss_all.__exit__(None, None, None)
        _restore_patches(backend, capture)


def box_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def compare_detections(reference: list[dict[str, Any]], candidate: list[dict[str, Any]], *,
                       iou_threshold: float = 0.99, confidence_tolerance: float = 0.001) -> dict[str, Any]:
    def box(row: dict[str, Any]) -> list[float]:
        if isinstance(row.get("bbox"), list):
            return [float(value) for value in row["bbox"]]
        coordinates = row.get("box2d") or {}
        return [float(coordinates[key]) for key in ("x1", "y1", "x2", "y2")]

    remaining = set(range(len(candidate)))
    matched = []
    unmatched_reference = []
    for reference_index in sorted(range(len(reference)), key=lambda index: -float(reference[index]["confidence"])):
        row = reference[reference_index]
        category = row["category"]
        possibilities = [(box_iou(box(row), box(candidate[index])), index)
                         for index in remaining if candidate[index]["category"] == category]
        overlap, candidate_index = max(possibilities, default=(0.0, -1))
        if candidate_index < 0:
            unmatched_reference.append({"reference_index": reference_index, "category": category,
                                        "confidence": row["confidence"], "reason": "no_same_class_candidate"})
            continue
        remaining.remove(candidate_index)
        cand = candidate[candidate_index]
        confidence_delta = abs(float(row["confidence"]) - float(cand["confidence"]))
        matched.append({"reference_index": reference_index, "candidate_index": candidate_index,
                        "category": category, "iou": overlap, "confidence_delta": confidence_delta,
                        "within_tolerance": overlap >= iou_threshold and confidence_delta <= confidence_tolerance})
    unmatched_candidate = [{"candidate_index": index, "category": candidate[index]["category"],
                            "confidence": candidate[index]["confidence"]}
                           for index in sorted(remaining)]
    return {"reference_count": len(reference), "candidate_count": len(candidate),
            "matched_class_count": len(matched),
            "within_tolerance_count": sum(row["within_tolerance"] for row in matched),
            "unmatched_reference": unmatched_reference, "unmatched_candidate": unmatched_candidate,
            "matched_pairs": matched, "iou_threshold": iou_threshold,
            "confidence_tolerance": confidence_tolerance}


def compare_responses(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ref, cand = reference["response"], candidate["response"]
    exact_fields = ("status", "risk", "reason_code", "reason", "features", "image_size")
    differences = [field for field in exact_fields if ref.get(field) != cand.get(field)]
    return {"exact_fields_match": not differences, "different_fields": differences,
            "risk_transition": f"{ref.get('risk')}->{cand.get('risk')}",
            "high_unknown_transition": {ref.get("risk"), cand.get("risk")} == {"high", "unknown"},
            "reference_reason_code": ref.get("reason_code"), "candidate_reason_code": cand.get("reason_code")}


def _round_stability(config_outputs: dict[str, Path]) -> dict[str, Any]:
    """Compare each configuration's two later runs to its first fixed run."""
    results = {}
    all_pass = True
    for config_id, folder in sorted(config_outputs.items()):
        path = folder / "round_outputs.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_round = {round_index: {
            (row.get("sample_id"), row.get("include_detections")): row
            for row in rows if row.get("global_round") == round_index
        } for round_index in range(3)}
        expected_keys = set(by_round[0])
        missing = {str(index): len(expected_keys - set(by_round[index])) for index in (1, 2)}
        comparison_count = 0
        response_mismatches = 0
        detector_mismatches = 0
        for round_index in (1, 2):
            for key in sorted(expected_keys & set(by_round[round_index])):
                reference, candidate = by_round[0][key], by_round[round_index][key]
                comparison_count += 1
                if not compare_responses(reference, candidate)["exact_fields_match"]:
                    response_mismatches += 1
                if not compare_detections(reference.get("objects", []), candidate.get("objects", []))["within_tolerance_count"] == len(reference.get("objects", [])) == len(candidate.get("objects", [])):
                    detector_mismatches += 1
        passed = (len(expected_keys) == 2 * 100 and len(by_round[1]) == len(expected_keys)
                  and len(by_round[2]) == len(expected_keys) and comparison_count == 400
                  and response_mismatches == 0 and detector_mismatches == 0)
        results[config_id] = {"status": "pass" if passed else "fail", "reference_rows": len(expected_keys),
                              "round_1_rows": len(by_round[1]), "round_2_rows": len(by_round[2]),
                              "comparisons": comparison_count, "response_mismatches": response_mismatches,
                              "detector_mismatches": detector_mismatches, "missing": missing}
        all_pass = all_pass and passed
    return {"status": "pass" if all_pass and len(results) == len(BACKENDS) * len(SIZES) else "fail",
            "configurations": results,
            "detector_iou_threshold": 0.99, "confidence_tolerance": 0.001,
            "response_features_exact": True}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parity(config_outputs: dict[str, Path]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    gates = {}
    for imgsz in SIZES:
        reference = _read_jsonl(config_outputs[f"torch-{imgsz}"] / "parity_outputs.jsonl")
        for backend in ("openvino", "onnx"):
            candidate = _read_jsonl(config_outputs[f"{backend}-{imgsz}"] / "parity_outputs.jsonl")
            ref_by_key = {(row["sample_id"], row["include_detections"]): row for row in reference}
            cand_by_key = {(row["sample_id"], row["include_detections"]): row for row in candidate}
            if set(ref_by_key) != set(cand_by_key):
                gates[f"{backend}-{imgsz}"] = {"status": "fail", "reason": "sample-output-key-mismatch"}
                continue
            detection_rows = []
            response_rows = []
            for key in sorted(ref_by_key):
                baseline_row, candidate_row = ref_by_key[key], cand_by_key[key]
                if not key[1]:
                    detection_rows.append({"sample_id": key[0], **compare_detections(
                        baseline_row.get("objects") or [], candidate_row.get("objects") or [])})
                response_rows.append({"sample_id": key[0], "include_detections": key[1],
                                      **compare_responses(baseline_row, candidate_row)})
            all_matches = all(row["unmatched_reference"] == [] and row["unmatched_candidate"] == []
                              and row["within_tolerance_count"] == row["reference_count"]
                              and row["reference_count"] == row["candidate_count"]
                              for row in detection_rows)
            high_unknown = [row for row in response_rows if row["high_unknown_transition"]]
            comparisons[f"{backend}-{imgsz}"] = {"detector_rows": detection_rows,
                                                  "response_rows": response_rows,
                                                  "detector_equal_within_tolerance": all_matches,
                                                  "response_field_mismatches": sum(not row["exact_fields_match"] for row in response_rows),
                                                  "high_unknown_transitions": high_unknown}
            response_mismatches = sum(not row["exact_fields_match"] for row in response_rows)
            gates[f"{backend}-{imgsz}"] = {"status": "pass" if all_matches and not high_unknown and response_mismatches == 0 else "fail",
                                            "detector_equal_within_tolerance": all_matches,
                                            "high_unknown_transition_count": len(high_unknown),
                                            "response_field_mismatch_count": response_mismatches}
    return {"comparisons": comparisons, "gates": gates}


def _performance_gates(config_outputs: dict[str, Path]) -> dict[str, Any]:
    gates = {}
    for imgsz in SIZES:
        torch_summary = json.loads((config_outputs[f"torch-{imgsz}"] / "summary.json").read_text(encoding="utf-8"))
        for backend in ("openvino", "onnx"):
            candidate = json.loads((config_outputs[f"{backend}-{imgsz}"] / "summary.json").read_text(encoding="utf-8"))
            ratios: dict[str, float | None] = {}
            no_regression = True
            twenty_percent = True
            failures_not_increased = candidate["pipeline_failures"] <= torch_summary["pipeline_failures"]
            for prefix in ("pipeline", "api"):
                for mode in ("false", "true"):
                    ref_ms = torch_summary[f"{prefix}_{mode}"]["median_round_p95_ms"]
                    cand_ms = candidate[f"{prefix}_{mode}"]["median_round_p95_ms"]
                    ratio = cand_ms / ref_ms if ref_ms and cand_ms is not None else None
                    key = f"{prefix}_{mode}"
                    ratios[key] = ratio
                    if ratio is None or ratio > 1.05:
                        no_regression = False
                    if ratio is None or ratio > 0.8:
                        twenty_percent = False
                if prefix == "pipeline" and candidate[f"{prefix}_true"]["failures"] > torch_summary[f"{prefix}_true"]["failures"]:
                    failures_not_increased = False
                if prefix == "api":
                    for mode in ("false", "true"):
                        for failure_kind in ("busy_503", "other_failure"):
                            if candidate[f"{prefix}_{mode}"][failure_kind] > torch_summary[f"{prefix}_{mode}"][failure_kind]:
                                failures_not_increased = False
                        if (candidate[f"{prefix}_{mode}"].get("pipeline_failures", 0)
                                > torch_summary[f"{prefix}_{mode}"].get("pipeline_failures", 0)):
                            failures_not_increased = False
            rss_ratio = candidate["peak_process_rss_bytes"] / torch_summary["peak_process_rss_bytes"] if torch_summary["peak_process_rss_bytes"] else None
            gates[f"{backend}-{imgsz}"] = {
                "p95_ratios_vs_same_size_square_torch": ratios,
                "latency_non_regression_5pct": no_regression,
                "at_least_20pct_pipeline_and_api_speedup_both_modes": twenty_percent,
                "no_failure_increase": failures_not_increased,
                "peak_rss_ratio": rss_ratio,
                "rss_within_1_25x": rss_ratio is not None and rss_ratio <= 1.25,
                "adoption_candidate": no_regression and twenty_percent and failures_not_increased
                and rss_ratio is not None and rss_ratio <= 1.25,
            }
    return gates


def adoption_completeness(config_outputs: dict[str, Path], statuses: dict[str, Any]) -> bool:
    """Require complete inputs and successful sequential API work before adoption."""
    if not measurement_completeness(config_outputs, statuses):
        return False
    return all(
        json.loads((config_outputs[f"{backend}-{size}"] / "summary.json").read_text(encoding="utf-8"))
        .get("completeness", {}).get("sequential_api_600_success_per_mode", False)
        and json.loads((config_outputs[f"{backend}-{size}"] / "summary.json").read_text(encoding="utf-8"))
        .get("api_completeness", {}).get("api_acceptable", False)
        for backend in BACKENDS for size in SIZES
    )


def measurement_completeness(config_outputs: dict[str, Path], statuses: dict[str, Any]) -> bool:
    """Require the full frozen workload, independent of whether a backend passes quality gates."""
    required_tasks = {task["task_id"] for task in build_measurement_schedule() + build_concurrency_schedule()}
    if set(statuses) != required_tasks or any(statuses[key].get("status") != "complete" for key in required_tasks):
        return False
    for key in (f"{backend}-{size}" for backend in BACKENDS for size in SIZES):
        folder = config_outputs.get(key)
        summary_path = folder / "summary.json" if folder else None
        if summary_path is None or not summary_path.is_file():
            return False
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        completeness = summary.get("completeness", {})
        if (not completeness.get("pipeline_600_per_mode")
                or not completeness.get("sequential_api_600_complete_per_mode")
                or not completeness.get("concurrency_complete")
                or not completeness.get("api_complete")
                or not completeness.get("round_outputs_complete")
                or not completeness.get("all_tasks_complete")):
            return False
    return True


def _aggregate_config_outputs(output: Path, task_statuses: dict[str, Any], config: dict[str, Any],
                              samples: list[dict[str, Any]]) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    sample_ids = [str(row["sample_id"]) for row in samples]
    for backend in BACKENDS:
        for imgsz in SIZES:
            config_id = f"{backend}-{imgsz}"
            task_list = [task for task in build_measurement_schedule() + build_concurrency_schedule()
                         if task["config_id"] == config_id]
            task_rows: list[dict[str, Any]] = []
            task_summaries: list[dict[str, Any]] = []
            for task in task_list:
                folder = output / "tasks" / task["task_id"]
                row_path = folder / "rows.jsonl"
                if row_path.is_file():
                    task_rows.extend(json.loads(line) for line in row_path.read_text(encoding="utf-8").splitlines()
                                     if line.strip())
                summary_path = folder / "summary.json"
                if summary_path.is_file():
                    task_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            folder = output / config_id
            folder.mkdir(parents=True, exist_ok=False)
            _write_csv(folder / "requests.csv", task_rows)
            (folder / "requests.jsonl").write_text("".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in task_rows), encoding="utf-8")
            round_records = []
            parity_records = []
            for task in task_list:
                task_folder = output / "tasks" / task["task_id"]
                round_path = task_folder / "round_outputs.jsonl"
                if round_path.is_file():
                    round_records.extend(json.loads(line) for line in round_path.read_text(encoding="utf-8").splitlines()
                                         if line.strip())
                parity_path = task_folder / "parity_outputs.jsonl"
                if parity_path.is_file():
                    parity_records.extend(json.loads(line) for line in parity_path.read_text(encoding="utf-8").splitlines()
                                          if line.strip())
            (folder / "round_outputs.jsonl").write_text("".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in round_records), encoding="utf-8")
            (folder / "parity_outputs.jsonl").write_text("".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in parity_records), encoding="utf-8")

            expected_pipeline = {
                (round_index, include, sample_id)
                for round_index in range(3) for include in MODES for sample_id in sample_ids
            }
            actual_pipeline = {
                (row.get("global_round"), row.get("include_detections"), row.get("sample_id"))
                for row in task_rows if row.get("kind") == "pipeline" and row.get("phase") == "measurement"
            }
            pipeline_rows = [row for row in task_rows if row.get("kind") == "pipeline"]
            pipeline_measurements = [row for row in pipeline_rows if row.get("phase") == "measurement"]
            pipeline_warmups = [row for row in pipeline_rows if row.get("phase") == "warmup"]
            expected_pipeline_warmups = {
                (task["task_id"], include, index)
                for task in build_measurement_schedule() if task["config_id"] == config_id
                for include in task["mode_order"] for index in range(int(config["warmups"]))
            }
            actual_pipeline_warmups = {
                (row.get("task_id"), row.get("include_detections"), row.get("request_index"))
                for row in pipeline_warmups
            }
            measurement_task_by_round = {
                task["global_round"]: task for task in build_measurement_schedule()
                if task["config_id"] == config_id
            }
            pipeline_binding_valid = all(
                row.get("config_id") == config_id
                and row.get("task_kind") == "measurement"
                and row.get("global_round") in measurement_task_by_round
                and row.get("task_id") == measurement_task_by_round[row["global_round"]]["task_id"]
                for row in pipeline_rows
            )
            pipeline_complete = (len(pipeline_measurements) == 600 and actual_pipeline == expected_pipeline
                                 and len(pipeline_warmups) == len(expected_pipeline_warmups)
                                 and actual_pipeline_warmups == expected_pipeline_warmups
                                 and pipeline_binding_valid)
            api_rows = [row for row in task_rows if row.get("kind") == "api"]
            api_completeness = _api_completeness(api_rows, config, len(samples), sample_ids)
            stats = {}
            for kind in ("pipeline", "api"):
                for mode, label in ((False, "false"), (True, "true")):
                    values = _stats(task_rows, mode, kind)
                    round_p95s = [item["p95_ms"] for item in values["rounds"] if item.get("p95_ms") is not None]
                    values["median_round_p95_ms"] = statistics.median(round_p95s) if round_p95s else None
                    stats[f"{kind}_{label}"] = values
            all_task_summaries = {entry.get("task_id"): entry for entry in task_summaries}
            statuses_for_config = {task["task_id"]: task_statuses.get(task["task_id"], {"status": "missing"})
                                   for task in task_list}
            task_summary_ids = {entry.get("task_id") for entry in task_summaries}
            expected_task_summary_ids = {task["task_id"] for task in task_list}
            rows_inputs_consistent = all(
                row.get("config_sha256") == config["config_sha256"]
                and row.get("performance_manifest_sha256") == task_summaries[0].get("performance_manifest_sha256")
                and row.get("source_checkpoint_sha256") == config["model_sha256"]
                and row.get("artifact_manifest_sha256")
                == task_summaries[0].get("artifact_manifest_sha256")
                for row in task_rows
            ) if task_summaries else False
            summaries_consistent = (task_summary_ids == expected_task_summary_ids
                                    and all(entry.get("status") == "complete"
                                            and entry.get("config_sha256") == config["config_sha256"]
                                            and entry.get("performance_manifest_sha256")
                                            == task_summaries[0].get("performance_manifest_sha256")
                                            and entry.get("artifact_manifest_sha256")
                                            == task_summaries[0].get("artifact_manifest_sha256")
                                            and entry.get("actual_provider") == task_summaries[0].get("actual_provider")
                                            for entry in task_summaries))
            job_complete = (all(status.get("status") == "complete" for status in statuses_for_config.values())
                            and summaries_consistent and rows_inputs_consistent)
            provider = task_summaries[0].get("actual_provider", {}) if task_summaries else {}
            if backend == "torch":
                provider_valid = (provider.get("backend") == "PyTorch" and provider.get("device") == "cpu"
                                  and provider.get("dtype") == "torch.float32")
            elif backend == "openvino":
                provider_valid = (provider.get("backend") == "OpenVINO"
                                  and provider.get("execution_devices") == ["CPU"]
                                  and provider.get("input_type") == "f32")
            else:
                provider_valid = (provider.get("backend") == "ONNX Runtime"
                                  and provider.get("providers") == ["CPUExecutionProvider"]
                                  and provider.get("input_type") == "tensor(float)")
            if not provider_valid:
                job_complete = False
            expected_round_output_keys = {
                (round_index, sample_id, include)
                for round_index in range(3) for sample_id in sample_ids for include in MODES
            }
            actual_round_output_keys = {
                (row.get("global_round"), row.get("sample_id"), row.get("include_detections"))
                for row in round_records
            }
            round_outputs_complete = (len(round_records) == 3 * len(samples) * len(MODES)
                                      and actual_round_output_keys == expected_round_output_keys)
            completeness = {
                "pipeline_600_per_mode": pipeline_complete,
                "pipeline_formal_rows": len(pipeline_measurements),
                "pipeline_warmup_rows": len(pipeline_warmups),
                "sequential_api_600_complete_per_mode": api_completeness["sequential_api_600_complete_per_mode"],
                "sequential_api_600_success_per_mode": api_completeness["sequential_api_600_success_per_mode"],
                "api_complete": api_completeness["api_complete"],
                "api_acceptable": api_completeness["api_acceptable"],
                "concurrency_complete": api_completeness["concurrency_complete"],
                "round_outputs_complete": round_outputs_complete,
                "all_tasks_complete": job_complete,
                "provider_verified": provider_valid,
            }
            summary = {
                "schema_version": 2, "status": "complete" if job_complete and pipeline_complete
                and api_completeness["api_complete"] and round_outputs_complete else "partial",
                "config_id": config_id, "backend": backend, "imgsz": imgsz,
                "completeness": completeness, "api_completeness": api_completeness,
                **stats,
                "pipeline_failures": sum(row.get("pipeline_failure", False) for row in pipeline_measurements),
                "api_failures": {label: {"503": stats[f"api_{label}"]["busy_503"],
                                           "other": stats[f"api_{label}"]["other_failure"],
                                           "pipeline": stats[f"api_{label}"]["pipeline_failures"]}
                                 for label in ("false", "true")},
                "peak_process_rss_bytes": max((entry.get("peak_process_rss_bytes", 0)
                                                for entry in task_summaries), default=0),
                "task_summaries": all_task_summaries,
                "task_status": statuses_for_config,
                "actual_provider": provider,
                "artifact_manifest_sha256": task_summaries[0].get("artifact_manifest_sha256")
                if task_summaries else None,
                "performance_manifest_sha256": task_summaries[0].get("performance_manifest_sha256")
                if task_summaries else None,
            }
            (folder / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                                                 encoding="utf-8")
            outputs[config_id] = folder
    return outputs


def compare(config_path: Path, output: Path, root: Path | None = None) -> dict[str, Any]:
    root = (root or Path.cwd()).resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError("run from the repository root")
    started = time.monotonic()
    config, config_sha = _read_config(config_path, root)
    config["config_sha256"] = config_sha
    samples, manifest_sha = _read_samples(config)
    source_model_sha = config["model_sha256"]
    artifact_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    for key, value in config["backend_artifacts"].items():
        path = Path(value)
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"artifact manifest does not exist: {key}")
        backend, size_text = key.rsplit("-", 1)
        verified = verify_artifact_manifest(path, backend=backend, imgsz=int(size_text), source_sha256=source_model_sha)
        artifact_paths[key] = path
        artifact_hashes[key] = verified["manifest_sha256"]
        config["backend_artifacts"][key] = str(path)
    output = output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite existing backend comparison directory")
    for input_path in [Path(config["model_path"]), Path(config["performance_manifest"]), *artifact_paths.values()]:
        if output == input_path or output.is_relative_to(input_path.parent if input_path.is_file() else input_path):
            raise ValueError("comparison output overlaps an input path")
    if output == Path(config["dataset_root"]) or output.is_relative_to(Path(config["dataset_root"])):
        raise ValueError("comparison output must be outside dataset_root")
    output.mkdir(parents=True, exist_ok=False)
    inputs = {"config_sha256": config_sha, "performance_manifest_sha256": manifest_sha,
              "source_checkpoint_sha256": source_model_sha, "artifact_manifests_sha256": artifact_hashes,
              "comparison_tool_sha256": sha256_file(Path(__file__).resolve())}
    (output / "input_hashes.json").write_text(json.dumps(inputs, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Freeze every job and budget before the first inference. Each job is one
    # backend/size/round (or post-round concurrency probe) in a fresh process.
    measurement_schedule = build_measurement_schedule()
    concurrency_schedule = build_concurrency_schedule()
    schedule = measurement_schedule + concurrency_schedule
    for task in schedule:
        task["budget_cap_seconds"] = float(config["budget_per_config_seconds"])
    if not validate_measurement_schedule(measurement_schedule):
        raise RuntimeError("round-robin schedule failed its internal balance check")
    schedule_document = {
        "schema_version": 2, "schedule": schedule,
        "sample_ids": [str(row["sample_id"]) for row in samples],
        "performance_manifest_sha256": manifest_sha,
        "source_checkpoint_sha256": source_model_sha,
        "config_sha256": config_sha,
        "artifact_manifest_sha256": artifact_hashes,
        "budget_per_config_seconds": float(config["budget_per_config_seconds"]),
        "total_budget_seconds": float(config["time_budget_seconds"]),
        "threads": int(config["threads"]), "seed": int(config["seed"]),
        "schema_note": "schema v2; formal rows are separated from pipeline/API warmups and concurrency",
    }
    schedule_bytes = (json.dumps(schedule_document, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    schedule_sha = hashlib.sha256(schedule_bytes).hexdigest()
    (output / "schedule.json").write_bytes(schedule_bytes)
    (output / "schedule.sha256").write_text(schedule_sha + "\n", encoding="ascii")
    inputs["schedule_sha256"] = schedule_sha
    (output / "input_hashes.json").write_text(json.dumps(inputs, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")

    task_statuses: dict[str, Any] = {}
    config_elapsed = {f"{backend}-{size}": 0.0 for backend in BACKENDS for size in SIZES}
    total_budget = float(config["time_budget_seconds"])
    per_config_budget = float(config["budget_per_config_seconds"])
    formal_ok = True
    for task in schedule:
        if task["task_kind"] == "concurrency" and not formal_ok:
            task_statuses[task["task_id"]] = {"status": "partial", "reason": "formal_rounds_incomplete"}
            (output / "task_status.json").write_text(
                json.dumps(task_statuses, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            continue
        config_id = task["config_id"]
        remaining_total = total_budget - (time.monotonic() - started)
        remaining_config = per_config_budget - config_elapsed[config_id]
        budget = min(remaining_total, remaining_config)
        if budget <= 0:
            status = "total_budget_exhausted" if remaining_total <= 0 else "config_budget_exhausted"
            task_statuses[task["task_id"]] = {"status": "partial", "reason": status, "budget_seconds": 0}
            if task["task_kind"] == "measurement":
                formal_ok = False
            continue
        backend, imgsz = task["backend"], int(task["imgsz"])
        artifact_manifest = str(artifact_paths[f"{backend}-{imgsz}"]) if backend != "torch" else None
        task = {**task, "budget_seconds": budget}
        child_output = output / "tasks" / task["task_id"]
        worker_path = output / f"worker-{task['task_id']}.json"
        worker_data = {"config": config, "task": task, "artifact_manifest": artifact_manifest,
                       "samples": samples, "performance_manifest_sha256": manifest_sha,
                       "output": str(child_output)}
        worker_path.write_text(json.dumps(worker_data, ensure_ascii=False), encoding="utf-8")
        task_started = time.monotonic()
        try:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_worker-json", str(worker_path)],
                                     cwd=root, capture_output=True, text=True, timeout=budget, check=False,
                                     env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                                          "OPENBLAS_NUM_THREADS": "2", "PYTHONPATH": str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")})
            elapsed = time.monotonic() - task_started
            config_elapsed[config_id] += elapsed
            worker_path.unlink(missing_ok=True)
            (output / f"{task['task_id']}-stdout.log").write_text(process.stdout[-200_000:], encoding="utf-8")
            (output / f"{task['task_id']}-stderr.log").write_text(process.stderr[-200_000:], encoding="utf-8")
            summary_path = child_output / "summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                status = summary.get("status") if process.returncode in {0, 3} else "failed"
                task_statuses[task["task_id"]] = {
                    "status": status, "summary_path": f"tasks/{task['task_id']}/summary.json",
                    "return_code": process.returncode, "elapsed_seconds": elapsed,
                    "budget_seconds": budget, "remaining_config_seconds": max(0, per_config_budget-config_elapsed[config_id]),
                }
            else:
                task_statuses[task["task_id"]] = {"status": "failed", "return_code": process.returncode,
                                                   "reason": "worker did not write summary",
                                                   "elapsed_seconds": elapsed, "budget_seconds": budget}
            if task["task_kind"] == "measurement" and task_statuses[task["task_id"]]["status"] != "complete":
                formal_ok = False
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - task_started
            config_elapsed[config_id] += elapsed
            worker_path.unlink(missing_ok=True)
            child_output.mkdir(parents=True, exist_ok=True)
            summary_path = child_output / "summary.json"
            if not summary_path.exists():
                summary_path.write_text(json.dumps({"schema_version": 2, "status": "partial",
                                                    "task": task, "partial_reason": "budget_exhausted"}) + "\n",
                                        encoding="utf-8")
            task_statuses[task["task_id"]] = {"status": "partial", "reason": "budget_exhausted",
                                               "summary_path": f"tasks/{task['task_id']}/summary.json",
                                               "elapsed_seconds": elapsed, "budget_seconds": budget}
            if task["task_kind"] == "measurement":
                formal_ok = False
        (output / "task_status.json").write_text(json.dumps(task_statuses, indent=2, sort_keys=True) + "\n",
                                                 encoding="utf-8")
    outputs = _aggregate_config_outputs(output, task_statuses, config, samples)
    matrix_complete = measurement_completeness(outputs, task_statuses)
    api_adoption_input_complete = adoption_completeness(outputs, task_statuses)
    parity = _parity(outputs) if matrix_complete else {
        "comparisons": {}, "gates": {}, "reason": "not_all_six_configurations_completed"}
    performance = _performance_gates(outputs) if matrix_complete else {}
    stability = _round_stability(outputs) if matrix_complete else {"status": "not_run", "reason": "matrix_partial"}
    if matrix_complete:
        parity_pass = all(gate.get("status") == "pass" for gate in parity["gates"].values())
        if not parity_pass or stability.get("status") != "pass":
            for gate in performance.values():
                gate["adoption_candidate"] = False
    adoption_ready = (api_adoption_input_complete and all(gate.get("adoption_candidate", False)
                                                          for gate in performance.values())
                      and bool(performance))
    result = {"schema_version": 2, "legacy_schema": 1,
              "status": "complete" if matrix_complete else "partial",
              "measurement_complete": matrix_complete, "adoption_ready": adoption_ready,
              "run_id": output.name, "measurement_schedule_sha256": schedule_sha,
              "configuration_status": {key: value["status"] for key, value in task_statuses.items()},
              "task_status": task_statuses, "configuration_elapsed_seconds": config_elapsed,
              "total_budget_seconds": total_budget, "budget_per_config_seconds": per_config_budget,
              "fixed_manifest_sha256": manifest_sha, "source_checkpoint_sha256": source_model_sha,
              "input_hashes": inputs,
              "parity": parity, "round_stability": stability,
              "performance_gates": performance,
              "adoption": {key: value.get("adoption_candidate", False) for key, value in performance.items()},
              "elapsed_seconds": time.monotonic() - started,
              "concurrency_started_after_all_formal_rounds": formal_ok,
              "notes": [
                  "schema v2 uses 18 serial backend/size/round jobs plus six serial post-round concurrency jobs; legacy schema v1 runs remain untouched",
                  "formal p95 uses sequential measurement rows only; warmups and concurrency probes are kept separate",
                  "every worker is a fresh process and cumulative startup/load/warmup/measurement time is capped per configuration and globally",
                  "a complete run may still fail parity, stability, failure-rate, latency, or RSS adoption gates",
              ]}
    (output / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                                               encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_worker-json", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args._worker_json:
            worker = json.loads(args._worker_json.read_text(encoding="utf-8"))
            result = _worker_task(worker["config"], worker["task"], worker.get("artifact_manifest"),
                                  worker["samples"], worker["performance_manifest_sha256"],
                                  Path(worker["output"]))
            print(json.dumps({"status": result["status"], "task_id": result["task_id"]}, indent=2))
            return 0 if result["status"] == "complete" else 3
        if not args.config or not args.output:
            parser.error("--config and --output are required")
        result = compare(args.config, args.output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as exc:
        print(f"backend comparison failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                      "configuration_status": result["configuration_status"],
                      "adoption": result["adoption"]}, indent=2))
    return 0 if result["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
