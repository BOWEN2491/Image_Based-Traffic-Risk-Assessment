"""Security and load binding checks for locally exported CPU artifacts."""

import hashlib
import json
import sys
import types

import pytest

from tools.compare_cpu_backends import _load_model, verify_artifact_manifest


def _bundle(tmp_path, *, root_name="bundle"):
    bundle = tmp_path / root_name
    model_dir = bundle / "assets" / "model_openvino_model"
    model_dir.mkdir(parents=True)
    files = {
        "assets/model_openvino_model/net.xml": b"<net/>\n",
        "assets/model_openvino_model/net.bin": b"weights",
        "assets/model_openvino_model/metadata.yaml": (
            b"task: detect\nbatch: 1\nimgsz: [640, 640]\n"
            b"names: {0: person, 1: car}\n"
            b"args: {quantize: 32, dynamic: false, nms: false}\nend2end: false\n"
        ),
    }
    for relative, payload in files.items():
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    record = {
        "schema_version": 1,
        "backend": "openvino",
        "source_checkpoint": {"sha256": "a" * 64},
        "artifact_root": "assets/model_openvino_model",
        "model_file": "assets/model_openvino_model/net.xml",
        "files": [
            {"path": key, "size_bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
            for key, value in files.items()
        ],
        "export": {"imgsz": 640, "batch": 1, "quantize": 32, "dynamic": False, "nms": False},
        "precision": {"fp32_verified": True},
        "classes": {"0": "person", "1": "car"},
    }
    manifest = bundle / "backend_artifact_manifest.json"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    return bundle, manifest, record


def _verify(manifest):
    return verify_artifact_manifest(manifest, backend="openvino", imgsz=640, source_sha256="a" * 64)


def test_openvino_load_binding_returns_xml_bundle_and_loads_only_that_directory(tmp_path, monkeypatch):
    bundle, manifest, _ = _bundle(tmp_path)
    artifact = _verify(manifest)
    expected_dir = (bundle / "assets" / "model_openvino_model").resolve()
    assert artifact["model_path"] == expected_dir / "net.xml"
    assert artifact["load_path"] == expected_dir
    calls = []
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = lambda value: calls.append(value) or "model"
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    assert _load_model("openvino", 640, artifact, {"model_path": "unused", "model_sha256": "a" * 64}) == "model"
    assert calls == [str(expected_dir)]


def test_openvino_rejects_manifest_root_separated_from_registered_xml(tmp_path):
    bundle, manifest, record = _bundle(tmp_path)
    other = bundle / "other"
    other.mkdir()
    record["artifact_root"] = "other"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="parent|artifact_root"):
        _verify(manifest)


@pytest.mark.parametrize("mutation, message", [
    ("extra_xml", "XML"),
    ("missing_metadata", "metadata"),
    ("metadata_mismatch", "metadata"),
    ("bad_bin", "BIN"),
    ("duplicate_normalized_path", "duplicate"),
])
def test_openvino_rejects_unbound_or_conflicting_bundle_files(tmp_path, mutation, message):
    bundle, manifest, record = _bundle(tmp_path)
    model_dir = bundle / "assets" / "model_openvino_model"
    if mutation == "extra_xml":
        (model_dir / "other.xml").write_bytes(b"extra")
    elif mutation == "missing_metadata":
        record["files"] = [entry for entry in record["files"] if not entry["path"].endswith("metadata.yaml")]
        (model_dir / "metadata.yaml").unlink()
    elif mutation == "metadata_mismatch":
        data = (model_dir / "metadata.yaml").read_text(encoding="utf-8").replace("task: detect", "task: classify")
        (model_dir / "metadata.yaml").write_text(data, encoding="utf-8")
        entry = next(x for x in record["files"] if x["path"].endswith("metadata.yaml"))
        raw = (model_dir / "metadata.yaml").read_bytes()
        entry.update(size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    elif mutation == "bad_bin":
        record["files"] = [entry for entry in record["files"] if not entry["path"].endswith("net.bin")]
    elif mutation == "duplicate_normalized_path":
        record["files"].append({**record["files"][0], "path": "assets/model_openvino_model/../model_openvino_model/net.xml"})
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _verify(manifest)


def test_openvino_rejects_path_traversal_and_symlink_escape(tmp_path):
    bundle, manifest, record = _bundle(tmp_path)
    record["artifact_root"] = "../outside"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="escape|artifact_root"):
        _verify(manifest)


def _onnx_bundle(tmp_path, model_bytes=b"onnx-placeholder"):
    bundle = tmp_path / "onnx-bundle"
    model_path = bundle / "model.onnx"
    bundle.mkdir()
    model_path.write_bytes(model_bytes)
    files = [{"path": "model.onnx", "size_bytes": len(model_bytes),
              "sha256": hashlib.sha256(model_bytes).hexdigest()}]
    record = {
        "schema_version": 1, "backend": "onnx",
        "source_checkpoint": {"sha256": "a" * 64}, "model_file": "model.onnx",
        "files": files,
        "export": {"imgsz": 640, "batch": 1, "quantize": 32, "dynamic": False, "nms": False},
        "precision": {"fp32_verified": True}, "classes": {"0": "person"},
    }
    manifest = bundle / "backend_artifact_manifest.json"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    return manifest


def test_onnx_bundle_requires_valid_model_without_external_tensor_data(tmp_path, monkeypatch):
    class Tensor:
        def __init__(self, location=0, external_data=()):
            self.data_location = location
            self.external_data = list(external_data)

    def fake_onnx(tensor):
        module = types.ModuleType("onnx")
        module.TensorProto = types.SimpleNamespace(EXTERNAL=1)
        module.load_model = lambda _path, load_external_data=False: types.SimpleNamespace(
            graph=types.SimpleNamespace(initializer=[tensor], sparse_initializer=[], node=[])
        )
        return module

    manifest = _onnx_bundle(tmp_path)
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx(Tensor()))
    verify_artifact_manifest(manifest, backend="onnx", imgsz=640, source_sha256="a" * 64)

    monkeypatch.setitem(sys.modules, "onnx", fake_onnx(Tensor(location=1)))
    with pytest.raises(ValueError, match="external ONNX"):
        verify_artifact_manifest(manifest, backend="onnx", imgsz=640, source_sha256="a" * 64)
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx(Tensor(external_data=[object()])))
    with pytest.raises(ValueError, match="external ONNX"):
        verify_artifact_manifest(manifest, backend="onnx", imgsz=640, source_sha256="a" * 64)


def test_onnx_bundle_rejects_malformed_model(tmp_path, monkeypatch):
    module = types.ModuleType("onnx")
    module.TensorProto = types.SimpleNamespace(EXTERNAL=1)

    def malformed(*_args, **_kwargs):
        raise RuntimeError("parse failure")

    module.load_model = malformed
    monkeypatch.setitem(sys.modules, "onnx", module)
    manifest = _onnx_bundle(tmp_path)
    with pytest.raises(ValueError, match="malformed"):
        verify_artifact_manifest(manifest, backend="onnx", imgsz=640, source_sha256="a" * 64)
