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


def smoke(model_dir: Path) -> dict[str, object]:
    """Download/verify YOLO and exercise ``/ready`` plus ``/api/predict``."""
    from fastapi.testclient import TestClient

    download(model_dir)
    settings = Settings(mode="rules", model_dir=model_dir)
    stream = io.BytesIO()
    Image.new("RGB", (64, 64), "gray").save(stream, format="PNG")
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
        if payload.get("risk") not in {"high", "unknown"}:
            raise RuntimeError("rules smoke returned a fabricated low/medium risk")
        return {"ready": ready.json(), "prediction": payload}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path.cwd() / "models")
    args = parser.parse_args(argv)
    print(json.dumps(smoke(args.model_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    main()
