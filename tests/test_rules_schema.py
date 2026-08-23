from __future__ import annotations

import math

import pytest

from traffic_risk.hard_rules import apply_hard_rules, apply_signal_policy
from traffic_risk.schema import MODEL_FEATURES, FeatureSchemaError, validate_model_features


def baseline(**changes):
    result = dict.fromkeys(MODEL_FEATURES, 0.0)
    result.update(changes)
    return result


@pytest.mark.parametrize(
    "changes,phrase",
    [
        ({"central_tl_is_red": 1}, "red"),
        ({"near_person_count": 1}, "pedestrian"),
        ({"max_box_area_person": 0.01, "bottom_half_person_ratio": 0.7}, "pedestrian"),
        ({"max_box_area_vehicle": 0.17}, "vehicle"),
        ({"near_vehicle_count": 2}, "Multiple"),
        ({"center_region_vehicle_ratio": 0.8, "sum_box_area_vehicle": 0.2}, "center"),
        ({"object_density_per_mp": 30, "sum_box_area_vehicle": 0.3}, "congested"),
        ({"mean_overlap_iou": 0.02, "max_box_area_vehicle": 0.12}, "overlap"),
    ],
)
def test_each_hard_rule(changes, phrase):
    risk, reason = apply_hard_rules(baseline(**changes))
    assert risk == "high" and phrase.lower() in reason.lower()


def test_rule_priority_and_nan_safety():
    risk, reason = apply_hard_rules(baseline(central_tl_is_red=1, near_person_count=math.nan))
    assert risk == "high" and "red" in reason.lower()
    assert apply_hard_rules(baseline(max_box_area_vehicle=math.nan)) == (None, None)


def test_signal_policy_yellow_and_other_red():
    row = baseline(central_tl_is_yellow=1, near_vehicle_count=1)
    assert apply_signal_policy("low", row)[0] == "medium"
    row = baseline(central_tl_code=0, n_tl_red=1, near_vehicle_count=1)
    assert apply_signal_policy("medium", row)[0] == "high"
    assert apply_signal_policy("low", baseline(central_tl_code=0, n_tl_red=1)) == ("low", None)
    with pytest.raises(ValueError):
        apply_signal_policy("unknown", row)


def test_schema_rejects_missing_non_numeric_and_nan():
    valid = baseline()
    assert len(validate_model_features(valid)) == len(MODEL_FEATURES)
    with pytest.raises(FeatureSchemaError, match="Missing"):
        validate_model_features({})
    with pytest.raises(FeatureSchemaError, match="numeric"):
        validate_model_features({**valid, "n_human": "bad"})
    with pytest.raises(FeatureSchemaError, match="finite"):
        validate_model_features({**valid, "n_human": float("nan")})
