"""FastAPI application with bounded, validated image uploads."""

from __future__ import annotations

import asyncio
import io
import logging
import warnings
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import anyio
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, model_validator

from .config import Settings
from .model_runtime import ModelRuntime, ModelUnavailableError
from .pipeline import run_single_image


LOGGER = logging.getLogger(__name__)
ALLOWED_MIME = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
FORMAT_SUFFIX = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class PredictionResponse(BaseModel):
    status: Literal["ok", "uncertain"]
    risk: Literal["low", "medium", "high", "unknown"]
    image_id: str
    reason: str
    features: dict[str, Any]
    model_versions: dict[str, str]

    @model_validator(mode="after")
    def status_matches_risk(self) -> PredictionResponse:
        if (self.status == "uncertain") != (self.risk == "unknown"):
            raise ValueError("uncertain and unknown must occur together")
        return self


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _magic_format(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def _validate_extension(filename: str | None) -> None:
    if Path(filename or "").suffix.lower() not in ALLOWED_EXTENSIONS:
        raise _error(415, "unsupported_extension", "Filename must end in .jpg, .jpeg, .png or .webp")


def _validate_image(data: bytes, mime: str | None, max_pixels: int) -> str:
    expected = ALLOWED_MIME.get(mime or "")
    if expected is None:
        raise _error(415, "unsupported_media_type", "Only JPEG, PNG and WebP images are accepted")
    magic = _magic_format(data)
    if magic != expected:
        raise _error(415, "content_mismatch", "Declared MIME type does not match image content")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format != expected:
                    raise _error(415, "content_mismatch", "Image decoder format does not match MIME type")
                width, height = image.size
                if width <= 0 or height <= 0 or width * height > max_pixels:
                    raise _error(413, "image_too_large", f"Decoded image exceeds {max_pixels} pixels")
                image.verify()
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise _error(413, "image_too_large", "Decoded image exceeds the safe pixel limit") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise _error(422, "invalid_image", "Image data could not be decoded") from exc
    return expected


async def _read_upload(upload: UploadFile, maximum: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while chunk := await upload.read(min(1024 * 1024, maximum + 1 - size)):
        size += len(chunk)
        if size > maximum:
            raise _error(413, "file_too_large", f"Upload exceeds {maximum} bytes")
        chunks.append(chunk)
    if not chunks:
        raise _error(400, "empty_upload", "Uploaded file is empty")
    return b"".join(chunks)


def create_app(
    settings: Settings | None = None,
    runtime_factory: Callable[[Settings], Any] = ModelRuntime,
) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.ready = False
        application.state.load_error = None
        application.state.inference_gate = asyncio.Semaphore(config.inference_concurrency)
        config.upload_dir.mkdir(parents=True, exist_ok=True)
        runtime = runtime_factory(config)
        application.state.runtime = runtime
        try:
            await anyio.to_thread.run_sync(runtime.load)
            application.state.ready = True
        except ModelUnavailableError as exc:
            application.state.load_error = str(exc)
            LOGGER.warning("Models are unavailable: %s", exc)
        except Exception:
            application.state.load_error = "Models could not be loaded"
            LOGGER.exception("Model runtime failed during startup")
        yield

    application = FastAPI(title="Traffic Scene Risk Assessment API", version="0.2.1", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/ready")
    async def ready(request: Request) -> dict[str, str]:
        if not request.app.state.ready:
            raise _error(503, "model_unavailable", "Models are not ready")
        return {"status": "ready", "mode": config.mode}

    @application.post("/api/predict", response_model=PredictionResponse)
    async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
        temp_path: Path | None = None
        acquired = False
        try:
            if not request.app.state.ready:
                raise _error(503, "model_unavailable", "Models are not ready")
            content_length = request.headers.get("content-length")
            if (
                content_length
                and content_length.isdecimal()
                and int(content_length) > config.max_upload_bytes + 64_000
            ):
                raise _error(413, "file_too_large", f"Upload exceeds {config.max_upload_bytes} bytes")
            _validate_extension(file.filename)
            data = await _read_upload(file, config.max_upload_bytes)
            image_format = _validate_image(data, file.content_type, config.max_image_pixels)
            image_id = str(uuid4())
            temp_path = config.upload_dir / f"{image_id}{FORMAT_SUFFIX[image_format]}"
            gate = request.app.state.inference_gate
            await anyio.Path(temp_path).write_bytes(data)
            try:
                await asyncio.wait_for(gate.acquire(), timeout=config.busy_timeout_seconds)
                acquired = True
            except TimeoutError as exc:
                raise _error(503, "service_busy", "Inference service is busy") from exc
            result = await anyio.to_thread.run_sync(
                run_single_image,
                temp_path,
                request.app.state.runtime,
            )
            return PredictionResponse(image_id=image_id, **result)
        except HTTPException:
            raise
        except ModelUnavailableError as exc:
            raise _error(503, "model_unavailable", "Models became unavailable") from exc
        except Exception as exc:
            LOGGER.exception("Inference failed")
            raise _error(500, "inference_failed", "Inference could not be completed") from exc
        finally:
            if acquired:
                request.app.state.inference_gate.release()
            if temp_path is not None:
                await anyio.Path(temp_path).unlink(missing_ok=True)
            await file.close()

    return application


app = create_app()

