"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in os.getenv(name, default).split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    mode: str = field(default_factory=lambda: os.getenv("RISK_MODE", "rules").strip().lower())
    model_manifest: Path | None = field(
        default_factory=lambda: Path(os.environ["RISK_MODEL_MANIFEST"]) if os.getenv("RISK_MODEL_MANIFEST") else None
    )
    model_dir: Path = field(
        default_factory=lambda: Path(os.getenv("RISK_MODEL_DIR", Path.cwd() / "models"))
    )
    upload_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "RISK_UPLOAD_DIR",
                Path(tempfile.gettempdir()) / "traffic-risk-assessment" / "uploads",
            )
        )
    )
    max_upload_bytes: int = field(
        default_factory=lambda: int(os.getenv("RISK_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    )
    max_image_pixels: int = field(
        default_factory=lambda: int(os.getenv("RISK_MAX_IMAGE_PIXELS", "20000000"))
    )
    inference_concurrency: int = field(
        default_factory=lambda: max(1, int(os.getenv("RISK_INFERENCE_CONCURRENCY", "1")))
    )
    busy_timeout_seconds: float = field(
        default_factory=lambda: max(0.01, float(os.getenv("RISK_BUSY_TIMEOUT_SECONDS", "0.1")))
    )
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: _csv_env("RISK_CORS_ORIGINS", "http://localhost:5173")
    )
    device: str = field(default_factory=lambda: os.getenv("RISK_DEVICE", "cpu"))

    @property
    def yolo_path(self) -> Path:
        return self.model_dir / "yolo" / "yolo11n.pt"

    @property
    def cnn_path(self) -> Path:
        return self.model_dir / "cnn" / "best_model.pth"

    @property
    def class_mapping_path(self) -> Path:
        return self.model_dir / "cnn" / "class_indices.json"

    @property
    def risk_model_path(self) -> Path:
        return self.model_dir / "risk" / "risk_xgb.ubj"

    @property
    def feature_order_path(self) -> Path:
        return self.model_dir / "risk" / "feature_order.json"

    @property
    def model_metadata_path(self) -> Path:
        return self.model_dir / "risk" / "model_metadata.json"

    def __post_init__(self) -> None:
        if self.mode not in {"rules", "models"}:
            raise ValueError("RISK_MODE must be 'rules' or 'models'")
        if self.mode == "models" and self.model_manifest is None:
            raise ValueError("RISK_MODEL_MANIFEST is required when RISK_MODE=models")


settings = Settings()

# Compatibility aliases used by older offline scripts.
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
YOLO_DIR = MODELS_DIR / "yolo"
LIGHT_CNN_DIR = MODELS_DIR / "cnn_out"
RISK_XGB_DIR = MODELS_DIR / "risk_xgb"

