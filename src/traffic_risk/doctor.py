"""Offline installation diagnostics; never deserialize or download checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def diagnose(config: dict | None = None) -> dict:
    config = config or {}
    checks = []

    def check(name, ok, detail, remedy):
        checks.append(dict(name=name, ok=bool(ok), detail=detail, remedy=None if ok else remedy))

    check("python", (3, 11) <= sys.version_info[:2] <= (3, 12), sys.version.split()[0],
          "Create a Python 3.11 or 3.12 environment; reinstall requirements-cpu.lock.")
    for package in ("torch", "torchvision", "ultralytics", "fastapi", "numpy", "pydantic"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        check(package, version is not None, version, "python -m pip install --no-deps -r requirements-cpu.lock")
    try:
        from packaging.requirements import Requirement
        requirements = importlib.metadata.requires("image-based-traffic-risk-assessment") or []
        for raw in requirements:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                version = importlib.metadata.version(requirement.name)
                valid = requirement.specifier.contains(version)
            except importlib.metadata.PackageNotFoundError:
                version, valid = None, False
            check(f"requirement:{requirement.name}", valid, version,
                  "python -m pip install --no-deps -r requirements-cpu.lock; python -m pip check")
    except (ImportError, importlib.metadata.PackageNotFoundError):
        check("project-install", False, "Project metadata unavailable", "python -m pip install --no-deps -e .")
    mode = config.get("mode", os.getenv("RISK_MODE", "rules"))
    check("mode", mode in {"rules", "models"}, mode, "Set RISK_MODE=rules or models.")
    root = Path(config.get("model_dir", os.getenv("RISK_MODEL_DIR", "models"))).resolve()
    manifest_path = config.get("model_manifest", os.getenv("RISK_MODEL_MANIFEST"))
    try:
        if mode == "models" and not manifest_path:
            raise ValueError("models mode requires a local manifest")
        from .contracts import ModelManifest
        manifest = ModelManifest.from_file(Path(manifest_path) if manifest_path else Path(__file__).with_name("model_manifest.json"))
        if manifest.is_withdrawn():
            raise ValueError("withdrawn manifest")
        if mode == "rules" and manifest.distribution_status != "public":
            raise ValueError("rules mode requires a public manifest")
        assets = [manifest.asset("yolo/yolo11n.pt")] if mode == "rules" else manifest.assets
        for asset in assets:
            path = (root / asset.destination).resolve()
            if root not in path.parents:
                raise ValueError("asset escapes model directory")
            ok = path.is_file() and path.stat().st_size == asset.size
            if ok:
                with path.open("rb") as stream:
                    ok = hashlib.file_digest(stream, "sha256").hexdigest() == asset.sha256.lower()
            check(asset.destination, ok, str(path), "traffic-risk download-models (rules); rebuild and verify the local bundle (models).")
        check("manifest", True, str(manifest_path or "packaged"), "")
    except (ValueError, OSError, KeyError, ImportError) as exc:
        check("manifest", False, str(exc), "Repair the manifest or unset RISK_MODEL_MANIFEST for public rules mode.")
    dataset = config.get("dataset_root") or os.getenv("RISK_DATASET_ROOT")
    if dataset:
        check("dataset", Path(dataset).is_dir(), str(dataset), "Set dataset_root to your existing dataset directory.")
    return {"ok": all(c["ok"] for c in checks), "mode": mode, "checks": checks,
            "note": "Integrity checks only; model execution is verified separately by smoke-rules."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
        if not isinstance(config, dict):
            raise ValueError("configuration must be a JSON object")
        result = diagnose(config)
    except (ValueError, OSError, TypeError) as exc:
        result = {"ok": False, "error": str(exc), "remedy": "Provide a readable JSON object with model_dir, mode, and optional dataset_root."}
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
