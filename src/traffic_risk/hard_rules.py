"""Deterministic safety policy shared by all inference entry points."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


THRESHOLDS = {
    "person_area_near_strict": 0.008,
    "bottom_half_min": 0.60,
    "vehicle_area_near": 0.16043790055171955,
    "near_vehicle_count": 2,
    "center_block_ratio": 0.80,
    "vehicle_sum_for_block": 0.20,
    "density_high": 30.0,
    "vehicle_sum_for_dense": 0.30,
    "mean_iou_high": 0.012615208626149536,
}


def _number(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def apply_hard_rules(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Return a high-risk override, or ``(None, None)`` for model evaluation."""
    if _number(row, "central_tl_is_red") == 1:
        return "high", "Central traffic light is red"
    if _number(row, "near_person_count") >= 1:
        return "high", "A pedestrian is detected in the near zone"
    if (
        _number(row, "max_box_area_person") >= THRESHOLDS["person_area_near_strict"]
        and _number(row, "bottom_half_person_ratio") >= THRESHOLDS["bottom_half_min"]
    ):
        return "high", "A pedestrian is very close in the forward zone"
    if _number(row, "max_box_area_vehicle") >= THRESHOLDS["vehicle_area_near"]:
        return "high", "A vehicle is extremely close"
    if _number(row, "near_vehicle_count") >= THRESHOLDS["near_vehicle_count"]:
        return "high", "Multiple vehicles are close"
    if (
        _number(row, "center_region_vehicle_ratio") >= THRESHOLDS["center_block_ratio"]
        and _number(row, "sum_box_area_vehicle") >= THRESHOLDS["vehicle_sum_for_block"]
    ):
        return "high", "Vehicles block the center region"
    if (
        _number(row, "object_density_per_mp") >= THRESHOLDS["density_high"]
        and _number(row, "sum_box_area_vehicle") >= THRESHOLDS["vehicle_sum_for_dense"]
    ):
        return "high", "The scene is highly congested"
    if (
        _number(row, "mean_overlap_iou") >= THRESHOLDS["mean_iou_high"]
        and (
            _number(row, "max_box_area_vehicle") >= 0.12
            or _number(row, "near_vehicle_count") >= 2
        )
    ):
        return "high", "Close vehicles overlap substantially"
    return None, None


def apply_signal_policy(base_risk: str, row: Mapping[str, Any]) -> tuple[str, str | None]:
    """Apply bounded traffic-light adjustments after the statistical model."""
    levels = ["low", "medium", "high"]
    if base_risk not in levels:
        raise ValueError(f"Invalid model risk: {base_risk!r}")
    pressure = any(
        _number(row, key) > 0
        for key in ("near_person_count", "near_vehicle_count", "center_region_vehicle_ratio")
    ) or _number(row, "object_density_per_mp") >= 5
    if _number(row, "central_tl_is_yellow") == 1 and pressure:
        return levels[max(1, levels.index(base_risk))], "Central yellow light with nearby actors"
    if _number(row, "central_tl_code") == 0 and _number(row, "n_tl_red") > 0 and pressure:
        return levels[min(2, levels.index(base_risk) + 1)], "Nearby red lights increase uncertainty"
    return base_risk, None

