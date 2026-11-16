"""Exercise each verified release model and the final response contract."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image

from traffic_risk.config import Settings
from traffic_risk.inference_outcome import REASON_CODES, is_processing_failure
from traffic_risk.model_runtime import ModelRuntime
from traffic_risk.pipeline import run_single_image
from traffic_risk.schema import MODEL_FEATURES
from traffic_risk.smoke_rules import smoke as smoke_rules


def smoke(model_dir: Path, manifest: Path) -> dict[str, object]:
    import torch

    runtime = ModelRuntime(Settings(mode="models", model_manifest=manifest, model_dir=model_dir, device="cpu"))
    runtime.load()
    with tempfile.TemporaryDirectory(prefix="risk-model-smoke-") as temporary:
        image_path = Path(temporary) / "smoke.jpg"
        Image.new("RGB", (640, 480), "gray").save(image_path)
        yolo_result = runtime.yolo.predict(source=str(image_path), verbose=False)
        assert len(yolo_result) == 1
        with torch.inference_mode():
            cnn_logits = runtime.cnn(torch.zeros((1, 3, 128, 128), device="cpu"))
        assert tuple(cnn_logits.shape) == (1, 4)
        risk = runtime.predict_risk(dict.fromkeys(MODEL_FEATURES, 0.0))
        assert risk in {"low", "medium", "high"}
        response = run_single_image(image_path, runtime)
    _validate_pipeline_response(response, runtime.versions)
    if response["reason_code"] != "no_supported_detections":
        raise RuntimeError("Synthetic gray image should produce no_supported_detections")
    rules_api = smoke_rules(model_dir)
    if rules_api["ready"].get("mode") != "rules":
        raise RuntimeError("Public API readiness did not report rules mode")
    return {
        "yolo_results": len(yolo_result),
        "cnn_shape": list(cnn_logits.shape),
        "xgboost_risk": risk,
        "pipeline": response,
        "rules_api": rules_api,
    }


def _validate_pipeline_response(response: dict[str, object], expected_versions: dict[str, str]) -> None:
    required = {"status", "risk", "reason", "reason_code", "features", "model_versions"}
    if not required <= response.keys():
        raise RuntimeError("Pipeline response is missing required fields")
    if response["status"] not in {"ok", "uncertain"}:
        raise RuntimeError("Pipeline returned an unsupported status")
    if response["risk"] not in {"low", "medium", "high", "unknown"}:
        raise RuntimeError("Pipeline returned an unsupported risk")
    if (response["status"] == "uncertain") != (response["risk"] == "unknown"):
        raise RuntimeError("Pipeline status and risk are inconsistent")
    code = response["reason_code"]
    if code not in REASON_CODES:
        raise RuntimeError("Pipeline returned an unknown reason_code")
    if is_processing_failure(code):
        raise RuntimeError(f"Pipeline reported an inference failure: {code}")
    if response["model_versions"] != expected_versions:
        raise RuntimeError("Pipeline model versions do not match the loaded runtime")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path.cwd() / "models")
    parser.add_argument("--manifest", type=Path, required=True, help="Verified local full-model manifest")
    args = parser.parse_args()
    print(json.dumps(smoke(args.model_dir, args.manifest), indent=2))


if __name__ == "__main__":
    main()
