"""Shared, strict model bundle contracts used by trainers and runtime."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AssetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="allow")
    contract: str = "traffic-risk"
    schema_version: str = "2.0.0"
    distribution_status: str = "local"
    mode: str = "models"
    assets: dict[str, AssetRecord]

    def is_withdrawn(self) -> bool:
        return self.distribution_status.lower() == "withdrawn"


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")
    contract: str = "traffic-risk"
    schema_version: str = "2.0.0"
    feature_order: list[str]


