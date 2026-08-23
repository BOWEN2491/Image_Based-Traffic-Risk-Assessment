"""Compatibility helpers for feature-row risk prediction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from .hard_rules import apply_hard_rules, apply_signal_policy
from .schema import validate_model_features


class Predictor(Protocol):
    def predict_risk(self, features: dict[str, Any]) -> str: ...


def predict_with_combined(row: Mapping[str, Any], predictor: Predictor) -> tuple[str, str]:
    features = dict(row)
    validate_model_features(features)
    risk, reason = apply_hard_rules(features)
    if risk is not None:
        return risk, reason or "Hard rule triggered"
    risk, signal_reason = apply_signal_policy(predictor.predict_risk(features), features)
    return risk, signal_reason or "Risk model prediction; no hard rule triggered"
