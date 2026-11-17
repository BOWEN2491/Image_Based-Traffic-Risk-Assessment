"""Export a SHA-pinned YOLO11n checkpoint to FP32 CPU backend artifacts.

The tool expects the dependencies to be preinstalled in its active environment.
Ultralytics' requirement checks are patched to disable automatic installation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


BACKENDS = {"openvino", "onnx"}
ALLOWED_SIZES = {640, 960}
MAX_BUDGET_SECONDS = 900


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _load_config(path: Path, root: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = json.loads(raw.decode("utf-8-sig"))
    required = {"backend", "model_path", "model_sha256", "imgsz", "budget_seconds"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"missing config fields: {', '.join(missing)}")
    if config["backend"] not in BACKENDS:
        raise ValueError("backend must be openvino or onnx")
    if int(config["imgsz"]) not in ALLOWED_SIZES:
        raise ValueError("imgsz must be 640 or 960")
    if not 0 < float(config["budget_seconds"]) <= MAX_BUDGET_SECONDS:
        raise ValueError(f"budget_seconds must be in (0, {MAX_BUDGET_SECONDS}]")
    model_path = Path(config["model_path"])
    model_path = model_path.resolve() if model_path.is_absolute() else (root / model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError("model_path does not exist")
    actual = sha256_file(model_path)
    if actual.lower() != str(config["model_sha256"]).lower():
        raise ValueError("model SHA256 does not match config")
    config["model_path"] = str(model_path)
    config["model_sha256"] = actual
    config["imgsz"] = int(config["imgsz"])
    config["budget_seconds"] = float(config["budget_seconds"])
    return config, hashlib.sha256(raw).hexdigest()


def _disable_ultralytics_auto_install() -> None:
    """Keep exporter/backend dependency checks fail-closed in this process."""
    from ultralytics.engine import exporter
    from ultralytics.nn.backends import onnx as onnx_backend
    from ultralytics.nn.backends import openvino as openvino_backend
    from ultralytics.utils import checks

    original = checks.check_requirements

    def no_install(requirements: Any = None, exclude: Any = (), install: bool = True,
                   cmds: str = "", constrain: Any = ()) -> Any:
        return original(requirements, exclude=exclude, install=False, cmds=cmds, constrain=constrain)

    checks.check_requirements = no_install
    exporter.check_requirements = no_install
    onnx_backend.check_requirements = no_install
    openvino_backend.check_requirements = no_install


def _preflight(config: dict[str, Any]) -> dict[str, str]:
    packages = {"ultralytics": "8.4.126", "torch": "2.13.0+cpu"}
    packages["openvino"] = "2025.4.1" if config["backend"] == "openvino" else "not-required-for-export"
    if config["backend"] == "onnx":
        packages["onnx"] = "1.23.0"
    installed: dict[str, str] = {}
    for name, expected in packages.items():
        if expected == "not-required-for-export":
            continue
        actual = metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"isolated environment requires {name}=={expected}; found {actual}")
        installed[name] = actual
    return installed


def _file_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"worker-config.json", "worker-result.json"}:
            records.append({"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size,
                            "sha256": sha256_file(path)})
    return records


def _verify_fp32(config: dict[str, Any], artifact_path: Path) -> dict[str, Any]:
    if config["backend"] == "onnx":
        import onnx
        from onnx import TensorProto

        model = onnx.load(str(artifact_path), load_external_data=False)
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except (onnx.onnx_cpp2py_export.shape_inference.InferenceError, ValueError):
            # Existing graph value_info remains checked; inability to infer shapes is recorded below.
            pass
        graph = model.graph
        graph_values = [*graph.input, *graph.output, *graph.value_info]
        value_types = [value.type.tensor_type.elem_type for value in graph_values
                       if value.type.HasField("tensor_type")]
        input_types = [value.type.tensor_type.elem_type for value in graph.input
                       if value.type.HasField("tensor_type")]
        output_types = [value.type.tensor_type.elem_type for value in graph.output
                        if value.type.HasField("tensor_type")]
        initializer_types = [value.data_type for value in graph.initializer]
        known_types = {*input_types, *output_types, *value_types, *initializer_types}
        if not input_types or any(value != TensorProto.FLOAT for value in input_types + output_types):
            raise RuntimeError("ONNX input/output tensors are not all FP32")
        if TensorProto.FLOAT16 in known_types:
            raise RuntimeError("ONNX graph contains FP16 tensors or initializers; FP32 experiment cannot proceed")
        return {"input_element_types": [onnx.TensorProto.DataType.Name(value) for value in input_types],
                "output_element_types": [onnx.TensorProto.DataType.Name(value) for value in output_types],
                "graph_value_element_types": sorted({onnx.TensorProto.DataType.Name(value) for value in value_types}),
                "initializer_element_types": sorted({TensorProto.DataType.Name(value) for value in initializer_types}),
                "shape_inference_applied": bool(graph.value_info), "fp32_verified": True}

    import openvino as ov

    model = ov.Core().read_model(str(artifact_path))
    element_types = sorted({node.get_output_element_type(0).get_type_name()
                            for node in model.get_ordered_ops() if node.get_output_size()})
    if "f16" in element_types or "bf16" in element_types:
        raise RuntimeError(f"OpenVINO graph has non-FP32 floating types: {element_types}")
    inputs = [port.get_element_type().get_type_name() for port in model.inputs]
    if any(value != "f32" for value in inputs):
        raise RuntimeError(f"OpenVINO model inputs are not FP32: {inputs}")
    return {"input_element_types": inputs, "graph_element_types": element_types, "fp32_verified": True}


def _export_worker(worker_config_path: Path) -> int:
    worker = json.loads(worker_config_path.read_text(encoding="utf-8"))
    config = worker["config"]
    output = Path(worker["output"])
    started = time.perf_counter()
    result_path = output / "worker-result.json"
    try:
        versions = _preflight(config)
        _disable_ultralytics_auto_install()
        import cv2
        import torch
        from traffic_risk.config import Settings
        from traffic_risk.model_runtime import ModelRuntime
        from ultralytics import YOLO

        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "2"
        torch.manual_seed(42)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        cv2.setNumThreads(2)

        source_model = Path(config["model_path"])
        model_dir = source_model.parent.parent
        verified_runtime = ModelRuntime(Settings(mode="rules", model_dir=model_dir, device="cpu"))
        verified_runtime.load()
        del verified_runtime
        gc.collect()

        if sha256_file(source_model) != config["model_sha256"]:
            raise RuntimeError("source checkpoint changed after formal loader verification")
        source_copy = output / "source" / source_model.name
        source_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_model, source_copy)
        if sha256_file(source_copy) != config["model_sha256"]:
            raise RuntimeError("copied source checkpoint hash mismatch")

        model = YOLO(str(source_copy))
        kwargs: dict[str, Any] = {
            "format": config["backend"], "imgsz": config["imgsz"], "device": "cpu",
            "batch": 1, "quantize": 32, "dynamic": False, "nms": False,
            "verbose": False,
        }
        if config["backend"] == "onnx":
            kwargs.update({"opset": 17, "simplify": False})
        exported = model.export(**kwargs)
        artifact = Path(str(exported)).resolve()
        if not artifact.is_relative_to(output.resolve()) or not artifact.exists():
            raise RuntimeError("exporter returned an artifact outside the new output directory")
        artifact_model = artifact / source_copy.with_suffix(".xml").name if config["backend"] == "openvino" else artifact
        if not artifact_model.is_file():
            raise RuntimeError("exported model file is missing")
        precision = _verify_fp32(config, artifact_model)
        files = _file_records(output)
        manifest = {
            "schema_version": 1,
            "backend": config["backend"],
            "requested_device": "CPU",
            "source_checkpoint": {"sha256": config["model_sha256"], "copied_path": source_copy.relative_to(output).as_posix()},
            "artifact_root": artifact.relative_to(output).as_posix(),
            "model_file": artifact_model.relative_to(output).as_posix(),
            "files": files,
            "export": {"imgsz": config["imgsz"], "batch": 1, "quantize": 32, "dynamic": False,
                       "nms": False, "simplify": kwargs.get("simplify"),
                       "opset": kwargs.get("opset"), "device": "cpu"},
            "precision": precision,
            "classes": {str(key): str(value) for key, value in model.names.items()},
            "packages": versions,
            "ultralytics": metadata.version("ultralytics"),
            "export_tool_sha256": sha256_file(Path(__file__).resolve()),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _json(output / "backend_artifact_manifest.json", manifest)
        _json(result_path, {"status": "complete", "manifest_sha256": sha256_file(output / "backend_artifact_manifest.json")})
        return 0
    except Exception as exc:
        _json(result_path, {"status": "failed", "error_type": type(exc).__name__,
                            "error": str(exc)[:1000], "elapsed_seconds": time.perf_counter() - started,
                            "partial_files": _file_records(output)})
        return 2


def export(config_path: Path, output: Path, root: Path | None = None) -> dict[str, Any]:
    root = (root or Path.cwd()).resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError("run from the repository root")
    config, config_sha = _load_config(config_path, root)
    output = output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing export directory")
    model_path = Path(config["model_path"])
    if output == model_path.parent or output.is_relative_to(model_path.parent):
        raise ValueError("output must be outside the source checkpoint directory")
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "export-input.json", {"config_sha256": config_sha, "backend": config["backend"],
                                        "model_sha256": config["model_sha256"], "imgsz": config["imgsz"]})
    worker_config_path = output / "worker-config.json"
    _json(worker_config_path, {"config": config, "output": str(output)})
    try:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_worker", str(worker_config_path)],
                              cwd=root, capture_output=True, text=True,
                              timeout=float(config["budget_seconds"]), check=False,
                              env={**os.environ, "YOLO_AUTOINSTALL": "false"})
    except subprocess.TimeoutExpired as exc:
        _json(output / "export-status.json", {"status": "partial", "reason": "budget_exhausted",
                                               "budget_seconds": config["budget_seconds"],
                                               "partial_files": _file_records(output)})
        raise TimeoutError("backend export exceeded its configured time budget; partial artifacts retained") from exc
    (output / "export-stdout.log").write_text(proc.stdout[-200_000:], encoding="utf-8")
    (output / "export-stderr.log").write_text(proc.stderr[-200_000:], encoding="utf-8")
    worker_result = json.loads((output / "worker-result.json").read_text(encoding="utf-8")) if (
        output / "worker-result.json").is_file() else {"status": "failed", "reason": "worker produced no result"}
    worker_config_path.unlink(missing_ok=True)
    if proc.returncode != 0 or worker_result.get("status") != "complete":
        _json(output / "export-status.json", {"status": worker_result.get("status", "failed"),
                                               "worker_exit_code": proc.returncode, "result": worker_result})
        raise RuntimeError(f"backend export failed; inspect {output.name}/export-stderr.log")
    manifest_path = output / "backend_artifact_manifest.json"
    return {"status": "complete", "backend": config["backend"], "imgsz": config["imgsz"],
            "manifest_sha256": sha256_file(manifest_path), "artifact_directory": str(output.name)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args._worker:
            return _export_worker(args._worker.resolve())
        if not args.config or not args.output:
            parser.error("--config and --output are required")
        result = export(args.config, args.output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as exc:
        print(f"backend export failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
