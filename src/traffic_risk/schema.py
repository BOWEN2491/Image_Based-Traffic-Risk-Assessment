"""Versioned feature contract shared by training, CLI and online inference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "2.0.0"
MODEL_FEATURES = (
    "n_human",
    "near_person_count",
    "near_vehicle_count",
    "max_box_area_person",
    "max_box_area_vehicle",
    "sum_box_area_person",
    "sum_box_area_vehicle",
    "bottom_half_person_ratio",
    "center_region_vehicle_ratio",
    "object_density_per_mp",
    "mean_overlap_iou",
)


class FeatureSchemaError(ValueError):
    """Raised when inference features do not satisfy the model contract."""


def validate_model_features(values: Mapping[str, Any]) -> list[float]:
    missing = [name for name in MODEL_FEATURES if name not in values]
    if missing:
        raise FeatureSchemaError(f"Missing model features: {', '.join(missing)}")
    result: list[float] = []
    for name in MODEL_FEATURES:
        try:
            value = float(values[name])
        except (TypeError, ValueError) as exc:
            raise FeatureSchemaError(f"Feature {name!r} must be numeric") from exc
        if not math.isfinite(value):
            raise FeatureSchemaError(f"Feature {name!r} must be finite")
        result.append(value)
    return result


def public_features(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON-safe features while preserving useful traffic-light signals."""
    output: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, bool | str | int):
            output[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            output[key] = value
    return output

