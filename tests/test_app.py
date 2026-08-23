from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.datastructures import UploadFile

from src.app import create_app
from src import app as app_module
from src.config import Settings
from src.model_runtime import ModelUnavailableError


class Runtime:
    versions = {"test": "1"}

    def __init__(self, _settings, *, failure=None, delay=0):
        self.failure = failure
        self.delay = delay

    def load(self):
        if self.failure == "load":
            raise ModelUnavailableError("missing")

    def detect(self, _path):
        time.sleep(self.delay)
        if isinstance(self.failure, Exception):
            raise self.failure
        return {"frames": [{"objects": [{"category": "car", "box2d": {"x1": 1, "y1": 1, "x2": 3, "y2": 3}, "attributes": {}}]}]}

    def predict_risk(self, _features):
        return "low"


def image_bytes(fmt="PNG", size=(20, 20)):
    stream = io.BytesIO()
    Image.new("RGB", size, "white").save(stream, format=fmt)
    return stream.getvalue()


def settings(tmp_path, **changes):
    values = {"model_dir": tmp_path / "models", "upload_dir": tmp_path / "uploads", "max_upload_bytes": 10_000, "max_image_pixels": 1_000, "inference_concurrency": 1, "busy_timeout_seconds": 0.02, "cors_origins": ("http://localhost:5173",), "device": "cpu"}
    values.update(changes)
    return Settings(**values)


def test_health_ready_and_success_use_uuid_and_cleanup(tmp_path):
    app = create_app(settings(tmp_path), Runtime)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ready"}
        response = client.post("/api/predict", files={"file": ("../../escape.png", image_bytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["risk"] == "low"
    assert body["image_id"] not in {"escape", "../../escape"}
    assert not list((tmp_path / "uploads").iterdir())


def test_models_not_ready(tmp_path):
    app = create_app(settings(tmp_path), lambda value: Runtime(value, failure="load"))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/api/predict", files={"file": ("x.png", image_bytes(), "image/png")}).status_code == 503


def test_empty_unsupported_mismatch_corrupt_and_oversize(tmp_path):
    app = create_app(settings(tmp_path, max_upload_bytes=1_000, max_image_pixels=50), Runtime)
    with TestClient(app) as client:
        assert client.post("/api/predict", files={"file": ("x.png", b"", "image/png")}).status_code == 400
        assert client.post("/api/predict", files={"file": ("x.gif", b"GIF89a", "image/gif")}).status_code == 415
        disguised = client.post("/api/predict", files={"file": ("x.exe", image_bytes(), "image/png")})
        assert disguised.status_code == 415
        assert disguised.json()["detail"]["code"] == "unsupported_extension"
        assert client.post("/api/predict", files={"file": ("x.jpg", image_bytes(), "image/jpeg")}).status_code == 415
        assert client.post("/api/predict", files={"file": ("x.png", b"\x89PNG\r\n\x1a\ninvalid", "image/png")}).status_code == 422
        assert client.post("/api/predict", files={"file": ("x.png", b"x" * 1_001, "image/png")}).status_code == 413
        assert client.post("/api/predict", files={"file": ("x.png", image_bytes(size=(10, 10)), "image/png")}).status_code == 413


@pytest.mark.parametrize("bomb", [Image.DecompressionBombError("bomb"), Image.DecompressionBombWarning("bomb")])
def test_pillow_decompression_bomb_is_rejected(monkeypatch, tmp_path, bomb):
    class BombContext:
        def __enter__(self):
            raise bomb

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(app_module.Image, "open", lambda *_args, **_kwargs: BombContext())
    with TestClient(create_app(settings(tmp_path), Runtime)) as client:
        response = client.post("/api/predict", files={"file": ("x.png", image_bytes(), "image/png")})
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "image_too_large"


def test_early_validation_failures_close_upload(monkeypatch, tmp_path):
    closed: list[str | None] = []
    original_close = UploadFile.close

    async def record_close(upload):
        closed.append(upload.filename)
        await original_close(upload)

    monkeypatch.setattr(UploadFile, "close", record_close)
    app = create_app(settings(tmp_path), Runtime)
    with TestClient(app) as client:
        response = client.post(
            "/api/predict",
            files={"file": ("not-an-image.exe", b"invalid", "application/octet-stream")},
        )

    assert response.status_code == 415
    assert "not-an-image.exe" in closed
    assert not list((tmp_path / "uploads").iterdir())


def test_runtime_exception_is_sanitized_and_file_removed(tmp_path):
    app = create_app(settings(tmp_path), lambda value: Runtime(value, failure=RuntimeError("secret C:/path")))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/predict", files={"file": ("x.png", image_bytes(), "image/png")})
    assert response.status_code == 500
    assert "secret" not in response.text and "C:/" not in response.text
    assert not list((tmp_path / "uploads").iterdir())


def test_busy_service_returns_503(tmp_path):
    app = create_app(settings(tmp_path), lambda value: Runtime(value, delay=0.1))
    with TestClient(app) as client:
        def request():
            return client.post("/api/predict", files={"file": ("x.png", image_bytes(), "image/png")})

        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = [future.result() for future in (pool.submit(request), pool.submit(request))]
    assert sorted(response.status_code for response in responses) == [200, 503]
    assert not list((tmp_path / "uploads").iterdir())


def test_cors_is_restricted(tmp_path):
    with TestClient(create_app(settings(tmp_path), Runtime)) as client:
        allowed = client.options("/api/predict", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
        denied = client.options("/api/predict", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in denied.headers
