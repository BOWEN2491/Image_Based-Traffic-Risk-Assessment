"""Shared, strict contracts for model bundles and their provenance.

The release manifest is consumed by both the downloader and runtime.  Keeping
the parser here prevents the two entry points from silently accepting different
schemas or asset sets.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from collections.abc import Mapping
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


def _safe_destination(value: str) -> str:
    """Validate a bundle-relative POSIX path and return its normalized form."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("asset destination must be a non-empty string")
    candidate = PurePosixPath(value.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts or ":" in candidate.parts[0]:
        raise ValueError("asset destination must stay inside the model directory")
    normalized = candidate.as_posix()
    if normalized in {".", ""}:
        raise ValueError("asset destination must name a file")
    return normalized


class AssetRecord(BaseModel):
    """Integrity and provenance record for one model-bundle asset.

    ``path`` is retained as a backwards-compatible input for local callers;
    published manifests use ``destination``.  Unknown provenance fields are
    retained so the release manifest can document loader constraints without
    making the runtime depend on them.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str | None = None
    destination: str | None = None
    path: str | None = None
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")

    @model_validator(mode="before")
    @classmethod
    def normalize_path(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        destination = data.get("destination") or data.get("path")
        if destination is not None:
            destination = _safe_destination(destination)
            data["destination"] = destination
            data.setdefault("path", destination)
            data.setdefault("name", PurePosixPath(destination).name)
        return data

    @model_validator(mode="after")
    def require_destination(self) -> AssetRecord:
        if self.destination is None and self.path is None:
            raise ValueError("asset requires destination or path")
        if self.destination is None:
            self.destination = _safe_destination(self.path or "")
        else:
            self.destination = _safe_destination(self.destination)
        self.path = self.destination
        if not self.name:
            self.name = PurePosixPath(self.destination).name
        return self


class ModelManifest(BaseModel):
    """Canonical manifest accepted by downloader, runtime, and trainers."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    release: str | None = None
    distribution_status: str = "local"
    contract: str = "traffic-risk"
    schema_version: str = Field(
        default="2.0.0",
        validation_alias=AliasChoices("schema_version", "feature_schema"),
    )
    model_versions: dict[str, str] = Field(default_factory=dict)
    assets: list[AssetRecord] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_assets(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        assets = data.get("assets")
        if isinstance(assets, Mapping):
            normalized = []
            for key, record in assets.items():
                item = dict(record)
                item.setdefault("name", key)
                normalized.append(item)
            data["assets"] = normalized
        if "schema_version" not in data and "feature_schema" in data:
            data["schema_version"] = data["feature_schema"]
        return data

    @model_validator(mode="after")
    def validate_contract(self) -> ModelManifest:
        if self.contract != "traffic-risk":
            raise ValueError("manifest contract must be traffic-risk")
        if self.schema_version != "2.0.0":
            raise ValueError("manifest feature schema must be 2.0.0")
        status = self.distribution_status.lower()
        if status not in {"public", "local", "withdrawn"}:
            raise ValueError("manifest distribution_status is invalid")
        self.distribution_status = status
        if status == "withdrawn":
            return self
        destinations = [asset.destination for asset in self.assets]
        if not destinations or any(destination is None for destination in destinations):
            raise ValueError("manifest must contain at least one asset")
        if len(destinations) != len(set(destinations)):
            raise ValueError("manifest contains duplicate asset destinations")
        return self

    @property
    def feature_schema(self) -> str:
        """Compatibility view used by older callers and release tooling."""
        return self.schema_version

    def is_withdrawn(self) -> bool:
        return self.distribution_status == "withdrawn"

    def asset(self, destination: str) -> AssetRecord:
        normalized = _safe_destination(destination)
        for asset in self.assets:
            if asset.destination == normalized:
                return asset
        raise KeyError(destination)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible normalized representation."""
        data = self.model_dump(exclude_none=True)
        # Emit both spellings during the v0.2.x compatibility window.  New
        # callers should use ``feature_schema``; older runtime tooling reads
        # ``schema_version``.
        data["feature_schema"] = data["schema_version"]
        return data

    @classmethod
    def from_file(cls, path: Path) -> ModelManifest:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return cls.model_validate(payload)
        except FileNotFoundError:
            raise
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid model manifest: {path}") from exc


class ModelMetadata(BaseModel):
    """Shared metadata contract emitted by local training workflows."""

    model_config = ConfigDict(extra="allow")
    contract: str = "traffic-risk"
    schema_version: str = "2.0.0"
    feature_order: list[str]
