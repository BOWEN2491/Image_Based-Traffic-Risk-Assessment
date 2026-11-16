from __future__ import annotations

import pytest

from tools.real_model_smoke import _validate_pipeline_response


def response(**overrides):
    payload = {
        "status": "uncertain",
        "risk": "unknown",
        "reason": "wording can change",
        "reason_code": "no_supported_detections",
        "features": {},
        "model_versions": {"test": "1"},
    }
    payload.update(overrides)
    return payload


def test_smoke_accepts_normal_abstention_without_matching_english_reason():
    _validate_pipeline_response(response(), {"test": "1"})


@pytest.mark.parametrize("reason_code", ["perception_error", "legacy_unspecified", "unrecognized"])
def test_smoke_rejects_failures_and_unknown_codes(reason_code):
    with pytest.raises(RuntimeError):
        _validate_pipeline_response(response(reason_code=reason_code), {"test": "1"})


def test_smoke_rejects_missing_fields_and_inconsistent_status():
    with pytest.raises(RuntimeError, match="missing required fields"):
        payload = response()
        payload.pop("reason_code")
        _validate_pipeline_response(payload, {"test": "1"})
    with pytest.raises(RuntimeError, match="inconsistent"):
        _validate_pipeline_response(response(status="ok"), {"test": "1"})
