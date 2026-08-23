"""Shared image-to-risk inference pipeline and command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
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


def run_single_image(image_path: str | Path, runtime: Runtime) -> dict[str, Any]:
    path = Path(image_path)
    try:
        detected = runtime.detect(path)
        frames = detected.get("frames", [])
        objects = frames[0].get("objects", []) if frames else []
        if not objects:
            return _uncertain("No supported road users or signals were detected", runtime)
        features = build_features_from_bdd(detected, path)
        validate_model_features(features)
        override, reason = apply_hard_rules(features)
        risk = override
        if getattr(runtime, "settings", None) is not None and runtime.settings.mode == "rules":
            if risk == "high":
                return {
                    "status": "ok", "risk": "high", "reason": reason or "Conservative rule triggered",
                    "features": public_features(features), "model_versions": runtime.versions,
                }
            return _uncertain("No conservative high-risk rule triggered; statistical models are disabled", runtime, features)
        if risk is None:
            risk = runtime.predict_risk(features)
            risk, signal_reason = apply_signal_policy(risk, features)
            reason = signal_reason or "Risk model prediction; no hard rule triggered"
        if risk not in {"low", "medium", "high"}:
            return _uncertain("Risk model returned an unsupported class", runtime, features)
        return {
            "status": "ok",
            "risk": risk,
            "reason": reason,
            "features": public_features(features),
            "model_versions": runtime.versions,
        }
    except FeatureSchemaError:
        LOGGER.exception("Feature schema validation failed during inference")
        return _uncertain("Extracted features did not match the model schema", runtime)
    except PerceptionError:
        LOGGER.exception("Perception failed during inference")
        return _uncertain("Perception could not produce a reliable result", runtime)
    except ValueError:
        LOGGER.exception("Invalid data encountered during inference")
        return _uncertain("Image features could not be processed reliably", runtime)


def _uncertain(reason: str, runtime: Runtime, features: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": "uncertain",
        "risk": "unknown",
        "reason": reason,
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

