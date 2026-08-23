"""Integrity-checked model loading and synchronous perception runtime."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import Settings
from .schema import MODEL_FEATURES, SCHEMA_VERSION, validate_model_features


MANIFEST = files(__package__).joinpath("model_manifest.json")
EXPECTED_MAPPING = {"green": 0, "red": 1, "unknown": 2, "yellow": 3}
EXPECTED_DESTINATIONS = {
    "yolo/yolo11n.pt",
    "cnn/best_model.pth",
    "cnn/class_indices.json",
    "risk/risk_xgb.ubj",
    "risk/feature_order.json",
    "risk/model_metadata.json",
}


class ModelUnavailableError(RuntimeError):
    """Raised when a model asset or its contract cannot be trusted."""


class PerceptionError(RuntimeError):
    """Raised when loaded perception models cannot produce a usable result."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.yolo: Any = None
        self.cnn: Any = None
        self.cnn_transform: Any = None
        self.class_names: dict[int, str] = {}
        self.booster: Any = None
        self._versions: dict[str, str] = {}

    @property
    def versions(self) -> dict[str, str]:
        return dict(self._versions)

    def _verify_manifest(self) -> dict[str, Any]:
        try:
            manifest_path = self.settings.model_manifest or Path(str(MANIFEST))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("distribution_status", "").lower() == "withdrawn":
                raise ModelUnavailableError("This model bundle has been withdrawn and cannot be loaded")
            if self.settings.mode == "models" and manifest.get("schema_version", manifest.get("feature_schema")) not in {SCHEMA_VERSION, "2.0.0"}:
                raise ModelUnavailableError("Release manifest feature schema is incompatible")
            if self.settings.mode == "rules":
                assets = manifest.get("assets", [])
                yolo_assets = [a for a in assets if a.get("destination", a.get("path")) == "yolo/yolo11n.pt"]
                if not yolo_assets:
                    raise ModelUnavailableError("Rules manifest does not contain a YOLO asset")
                for asset in yolo_assets:
                    destination = (self.settings.model_dir / asset.get("destination", asset.get("path"))).resolve()
                    if not destination.is_file() or destination.stat().st_size != asset["size"] or _sha256(destination) != asset["sha256"]:
                        raise ModelUnavailableError("YOLO asset failed integrity verification")
                self._versions = {"mode": "rules", "feature_schema": "2.0.0", "yolo": str(manifest.get("model_versions", {}).get("yolo", "validated"))}
                return manifest
            versions = manifest["model_versions"]
            if set(versions) != {"yolo", "traffic_light_cnn", "risk_xgb"} or not all(
                isinstance(value, str) and value for value in versions.values()
            ):
                raise ModelUnavailableError("Release manifest model versions are invalid")
            assets = manifest["assets"]
            destinations = [asset["destination"] for asset in assets]
            if len(destinations) != len(set(destinations)) or set(destinations) != EXPECTED_DESTINATIONS:
                raise ModelUnavailableError("Release manifest asset set is incompatible")

            model_root = self.settings.model_dir.resolve()
            for asset in assets:
                destination = (model_root / asset["destination"]).resolve()
                if destination != model_root and model_root not in destination.parents:
                    raise ModelUnavailableError("Release manifest contains an unsafe asset path")
                if not destination.is_file():
                    raise ModelUnavailableError("Required model assets are missing; run the model downloader")
                if destination.stat().st_size != asset["size"] or _sha256(destination) != asset["sha256"]:
                    raise ModelUnavailableError("A model asset failed integrity verification")
            self._versions = {
                "mode": "models",
                "release": manifest.get("release", "local"),
                "feature_schema": manifest.get("schema_version", manifest.get("feature_schema", SCHEMA_VERSION)),
                **versions,
            }
            return manifest
        except ModelUnavailableError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelUnavailableError("Release manifest is invalid") from exc

    def _validate_sidecars(self, manifest: dict[str, Any]) -> None:
        try:
            mapping = json.loads(self.settings.class_mapping_path.read_text(encoding="utf-8"))
            if mapping != EXPECTED_MAPPING or any(type(value) is not int for value in mapping.values()):
                raise ModelUnavailableError("Traffic-light class mapping is incompatible")
            if set(mapping.values()) != set(range(4)):
                raise ModelUnavailableError("Traffic-light class mapping indices are incompatible")

            assets = {asset["name"]: asset for asset in manifest["assets"]}
            if assets["best_model.pth"]["class_mapping"] != EXPECTED_MAPPING:
                raise ModelUnavailableError("CNN manifest class mapping is incompatible")

            feature_order = json.loads(self.settings.feature_order_path.read_text(encoding="utf-8"))
            metadata = json.loads(self.settings.model_metadata_path.read_text(encoding="utf-8"))
            risk_contract = assets["risk_xgb.ubj"]["feature_schema"]
            if feature_order != list(MODEL_FEATURES):
                raise ModelUnavailableError("Risk model feature order is incompatible")
            if risk_contract.get("version") != SCHEMA_VERSION or risk_contract.get("features") != list(MODEL_FEATURES):
                raise ModelUnavailableError("Risk model manifest schema is incompatible")
            if metadata.get("train_cols_nosignal") != list(MODEL_FEATURES):
                raise ModelUnavailableError("Risk model metadata schema is incompatible")
            self.class_names = {index: name for name, index in mapping.items()}
        except ModelUnavailableError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelUnavailableError("Model sidecar metadata is invalid") from exc

    def load(self) -> None:
        manifest = self._verify_manifest()
        try:
            if self.settings.mode == "models":
                self._validate_sidecars(manifest)
            from ultralytics import YOLO
            self.yolo = YOLO(str(self.settings.yolo_path))
            if self.settings.mode == "rules":
                return
            import torch
            import torch.nn as nn
            import xgboost as xgb
            from torchvision import models, transforms
            cnn = models.resnet18(weights=None)
            cnn.fc = nn.Linear(cnn.fc.in_features, len(EXPECTED_MAPPING))
            state = torch.load(self.settings.cnn_path, map_location=self.settings.device, weights_only=True)
            if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
                raise ModelUnavailableError("CNN asset is not a state dictionary")
            cnn.load_state_dict(state, strict=True)
            self.cnn = cnn.eval().to(self.settings.device)
            self.cnn_transform = transforms.Compose(
                [
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            self.booster = xgb.Booster()
            self.booster.load_model(self.settings.risk_model_path)
            if self.booster.num_features() != len(MODEL_FEATURES):
                raise ModelUnavailableError("Risk model feature count does not match the schema")
            if self.booster.feature_names != list(MODEL_FEATURES):
                raise ModelUnavailableError("Risk model feature names do not match the schema")
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError("Model assets could not be loaded") from exc

    def _traffic_light_color(self, image: np.ndarray, box: tuple[int, ...]) -> str:
        import torch
        from PIL import Image

        x1, y1, x2, y2 = box
        height, width = image.shape[:2]
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        if x2 - x1 < 6 or y2 - y1 < 6:
            return "unknown"
        rgb = cv2.cvtColor(image[int(y1) : int(y2), int(x1) : int(x2)], cv2.COLOR_BGR2RGB)
        tensor = self.cnn_transform(Image.fromarray(rgb)).unsqueeze(0).to(self.settings.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.cnn(tensor), dim=1)[0]
        confidence, index = probabilities.max(dim=0)
        return self.class_names[int(index)] if float(confidence) >= 0.5 else "unknown"

    def detect(self, image_path: str | Path) -> dict[str, Any]:
        if self.yolo is None or (self.settings.mode == "models" and self.cnn is None):
            raise ModelUnavailableError("Models are not loaded")
        image = cv2.imread(str(image_path))
        if image is None:
            raise PerceptionError("Image could not be decoded")
        try:
            result = self.yolo.predict(source=str(image_path), conf=0.25, iou=0.45, verbose=False)[0]
            names = result.names
            objects: list[dict[str, Any]] = []
            supported = {
                "person",
                "bicycle",
                "motorcycle",
                "car",
                "bus",
                "truck",
                "train",
                "traffic light",
                "traffic sign",
            }
            for identifier, detected in enumerate(result.boxes):
                category = str(names[int(detected.cls.item())]).lower().replace("_", " ")
                if category not in supported:
                    continue
                coords = detected.xyxy.cpu().numpy().reshape(-1).tolist()
                color = "unknown"
                if category == "traffic light":
                    color = self._traffic_light_color(image, tuple(round(value) for value in coords))
                objects.append(
                    {
                        "id": identifier,
                        "category": category,
                        "attributes": {"trafficLightColor": color},
                        "box2d": dict(zip(("x1", "y1", "x2", "y2"), map(float, coords), strict=True)),
                    }
                )
            return {"frames": [{"timestamp": 0, "objects": objects}]}
        except (ModelUnavailableError, PerceptionError):
            raise
        except Exception as exc:
            raise PerceptionError("Object perception failed") from exc

    def predict_risk(self, features: dict[str, Any]) -> str:
        if self.settings.mode == "rules":
            raise ModelUnavailableError("Risk model is unavailable in rules mode")
        if self.booster is None:
            raise ModelUnavailableError("Risk model is not loaded")
        import xgboost as xgb

        values = validate_model_features(features)
        matrix = xgb.DMatrix([values], feature_names=list(MODEL_FEATURES))
        raw = np.asarray(self.booster.predict(matrix)).reshape(-1)
        if raw.size == 3:
            prediction = int(raw.argmax())
        elif raw.size == 1 and float(raw[0]).is_integer():
            prediction = int(raw[0])
        else:
            raise PerceptionError("Risk model returned an invalid prediction")
        try:
            return {0: "low", 1: "medium", 2: "high"}[prediction]
        except KeyError as exc:
            raise PerceptionError("Risk model returned an unknown class") from exc
