"""Lightweight outcome codes for classifying inference results."""

from __future__ import annotations

from typing import Final

REASON_CODES: Final[frozenset[str]] = frozenset({
    "hard_rule_triggered",
    "model_prediction",
    "no_supported_detections",
    "no_rule_triggered",
    "feature_schema_error",
    "perception_error",
    "invalid_features",
    "invalid_model_output",
    "legacy_unspecified",
})

PROCESSING_FAILURE_CODES: Final[frozenset[str]] = frozenset({
    "feature_schema_error",
    "perception_error",
    "invalid_features",
    "invalid_model_output",
    "legacy_unspecified",
})


def is_processing_failure(reason_code: str | None) -> bool:
    """Return true unless a known reason code proves processing succeeded."""
    return reason_code not in REASON_CODES or reason_code in PROCESSING_FAILURE_CODES
