"""Run a small end-to-end smoke test for the public rules runtime."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from PIL import Image

from .app import create_app
from .config import Settings
from .download_models import download
from .inference_outcome import REASON_CODES, is_processing_failure


def smoke(model_dir: Path, image: Path | None = None) -> dict[str, object]:
    """Download/verify YOLO and exercise ``/ready`` plus ``/api/predict``."""
    from fastapi.testclient import TestClient

    download(model_dir)
    settings = Settings(mode="rules", model_dir=model_dir)
    stream = io.BytesIO()
    if image is None:
        Image.new("RGB", (64, 64), "gray").save(stream, format="PNG")
    else:
        with Image.open(image) as source:
            source.convert("RGB").save(stream, format="PNG")
    with TestClient(create_app(settings)) as client:
        ready = client.get("/ready")
        if ready.status_code != 200 or ready.json() != {"status": "ready", "mode": "rules"}:
            raise RuntimeError(f"rules runtime is not ready: {ready.status_code} {ready.text}")
        response = client.post(
            "/api/predict",
            files={"file": ("smoke.png", stream.getvalue(), "image/png")},
        )
        if response.status_code != 200:
            raise RuntimeError(f"rules prediction failed: {response.status_code} {response.text}")
        payload = response.json()
        required = {"status", "risk", "reason", "reason_code", "features", "model_versions", "image_id"}
        if not required <= payload.keys():
            raise RuntimeError("rules smoke response is missing required fields")
        if payload["status"] not in {"ok", "uncertain"}:
            raise RuntimeError("rules smoke returned an unsupported status")
        if payload.get("risk") not in {"high", "unknown"}:
            raise RuntimeError("rules smoke returned a fabricated low/medium risk")
        if (payload["status"] == "uncertain") != (payload["risk"] == "unknown"):
            raise RuntimeError("rules smoke returned inconsistent status and risk")
        code = payload["reason_code"]
        if code not in REASON_CODES or is_processing_failure(code):
            raise RuntimeError(f"rules smoke reported an inference failure: {code}")
        allowed = {"hard_rule_triggered", "no_supported_detections", "no_rule_triggered"}
        if code not in allowed:
            raise RuntimeError(f"rules smoke returned an unexpected reason_code: {code}")
        if code == "hard_rule_triggered" and (payload["status"], payload["risk"]) != ("ok", "high"):
            raise RuntimeError("hard_rule_triggered must return ok/high")
        if code in {"no_supported_detections", "no_rule_triggered"} and (payload["status"], payload["risk"]) != ("uncertain", "unknown"):
            raise RuntimeError(f"{code} must return uncertain/unknown")
        if image is None and code != "no_supported_detections":
            raise RuntimeError("Synthetic gray image should produce no_supported_detections")
        return {"ready": ready.json(), "prediction": payload}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path.cwd() / "models")
    parser.add_argument("--image", type=Path, help="Optional local road image; otherwise use an empty synthetic image")
    args = parser.parse_args(argv)
    print(json.dumps(smoke(args.model_dir, args.image), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
