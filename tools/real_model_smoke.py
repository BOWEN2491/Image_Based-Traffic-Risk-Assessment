"""Exercise each verified release model and the final response contract."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image

from src.config import Settings
from src.model_runtime import ModelRuntime
from src.pipeline import run_single_image
from src.schema import MODEL_FEATURES


def smoke(model_dir: Path) -> dict[str, object]:
    import torch

    runtime = ModelRuntime(Settings(model_dir=model_dir, device="cpu"))
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
    assert response["status"] in {"ok", "uncertain"}
    assert response["risk"] in {"low", "medium", "high", "unknown"}
    assert (response["status"] == "uncertain") == (response["risk"] == "unknown")
    assert set(response) == {"status", "risk", "reason", "features", "model_versions"}
    assert response["model_versions"] == runtime.versions
    return {
        "yolo_results": len(yolo_result),
        "cnn_shape": list(cnn_logits.shape),
        "xgboost_risk": risk,
        "pipeline": response,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path.cwd() / "models")
    args = parser.parse_args()
    print(json.dumps(smoke(args.model_dir), indent=2))


if __name__ == "__main__":
    main()
