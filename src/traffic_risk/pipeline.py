"""Shared image-to-risk inference pipeline and command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Protocol

from .build_features import build_features_from_bdd
from .config import Settings
from .hard_rules import apply_hard_rules, apply_signal_policy
from .model_runtime import ModelRuntime, ModelUnavailableError, PerceptionError
from .schema import FeatureSchemaError, public_features, validate_model_features


LOGGER = logging.getLogger(__name__)


class Runtime(Protocol):
    versions: dict[str, str]

    def detect(self, image_path: str | Path) -> dict[str, Any]: ...
    def predict_risk(self, features: dict[str, Any]) -> str: ...


def run_single_image(
    image_path: str | Path, runtime: Runtime, *, include_detections: bool = False
) -> dict[str, Any]:
    path = Path(image_path)
    phase = "features"
    details = None
    try:
        detected = runtime.detect(path)
        frames = detected.get("frames", [])
        objects = frames[0].get("objects", []) if frames else []
        details = _detection_details(path, objects) if include_detections else None
        if not objects:
            return _with_detection_details(
                _uncertain("No supported road users or signals were detected", runtime, reason_code="no_supported_detections"),
                details,
            )
        features = build_features_from_bdd(detected, path)
        validate_model_features(features)
        override, reason = apply_hard_rules(features)
        risk = override
        if getattr(runtime, "settings", None) is not None and runtime.settings.mode == "rules":
            if risk == "high":
                return _with_detection_details({
                    "status": "ok", "risk": "high", "reason": reason or "Conservative rule triggered",
                    "reason_code": "hard_rule_triggered",
                    "features": public_features(features), "model_versions": runtime.versions,
                }, details)
            return _with_detection_details(_uncertain(
                "No conservative high-risk rule triggered; statistical models are disabled",
                runtime,
                features,
                reason_code="no_rule_triggered",
            ), details)
        if risk is None:
            phase = "model"
            risk = runtime.predict_risk(features)
            risk, signal_reason = apply_signal_policy(risk, features)
            reason = signal_reason or "Risk model prediction; no hard rule triggered"
        if risk not in {"low", "medium", "high"}:
            return _with_detection_details(_uncertain("Risk model returned an unsupported class", runtime, features, reason_code="invalid_model_output"), details)
        return _with_detection_details({
            "status": "ok",
            "risk": risk,
            "reason": reason,
            "reason_code": "hard_rule_triggered" if override == "high" else "model_prediction",
            "features": public_features(features),
            "model_versions": runtime.versions,
        }, details)
    except FeatureSchemaError:
        LOGGER.exception("Feature schema validation failed during inference")
        return _with_detection_details(
            _uncertain("Extracted features did not match the model schema", runtime, reason_code="feature_schema_error"),
            details,
        )
    except PerceptionError:
        LOGGER.exception("Perception failed during inference")
        return _with_detection_details(
            _uncertain("Perception could not produce a reliable result", runtime, reason_code="perception_error"),
            details,
        )
    except ValueError:
        LOGGER.exception("Invalid data encountered during inference")
        code = "invalid_model_output" if phase == "model" else "invalid_features"
        return _with_detection_details(
            _uncertain("Image features could not be processed reliably", runtime, reason_code=code),
            details,
        )


def _detection_details(image_path: Path, objects: list[dict[str, Any]]) -> dict[str, Any]:
    """Build bounded normalized overlay data from the same runtime detection."""
    from PIL import Image, ImageOps

    with Image.open(image_path) as source:
        oriented = ImageOps.exif_transpose(source)
        width, height = oriented.size
    detections = []
    for obj in objects[:300]:
        box = obj.get("box2d", {})
        coordinates = [float(box[key]) for key in ("x1", "y1", "x2", "y2")]
        x1, y1, x2, y2 = coordinates
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("detector returned non-finite bbox coordinates")
        if x1 >= x2 or y1 >= y2:
            raise ValueError("detector returned an unordered bbox")
        normalized = [min(1.0, max(0.0, coordinates[0] / width)),
                      min(1.0, max(0.0, coordinates[1] / height)),
                      min(1.0, max(0.0, coordinates[2] / width)),
                      min(1.0, max(0.0, coordinates[3] / height))]
        if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
            raise ValueError("detector returned a bbox outside the image bounds")
        category = str(obj["category"])
        attributes = obj.get("attributes") or {}
        color = attributes.get("trafficLightColor") if category == "traffic light" else None
        if category == "traffic light" and color not in {"green", "red", "yellow", "unknown"}:
            color = "unknown"
        confidence = float(obj["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("detector returned an invalid confidence score")
        detections.append({
            "id": int(obj["id"]), "category": category, "bbox": normalized,
            "confidence": confidence, "traffic_light_color": color,
        })
    return {"image_size": {"width": width, "height": height}, "detections": detections}


def _with_detection_details(result: dict[str, Any], details: dict[str, Any] | None) -> dict[str, Any]:
    if details is not None:
        result.update(details)
    return result


def _uncertain(
    reason: str,
    runtime: Runtime,
    features: dict[str, Any] | None = None,
    *,
    reason_code: str,
) -> dict[str, Any]:
    return {
        "status": "uncertain",
        "risk": "unknown",
        "reason": reason,
        "reason_code": reason_code,
        "features": public_features(features or {}),
        "model_versions": runtime.versions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic scene risk assessment demo")
    parser.add_argument("--img", type=Path, required=True)
    args = parser.parse_args()
    runtime = ModelRuntime(Settings())
    try:
        runtime.load()
    except ModelUnavailableError as exc:
        parser.error(str(exc))
    result = run_single_image(args.img, runtime)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

