"""Standalone traffic-light color classifier using the runtime CNN contract."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from .cnn_contract import CLASS_MAPPING, classify_logits, contract, inference_transform, make_model, validate_contract


def _load_mapping(mapping_path: Path) -> dict[int, str]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if (not isinstance(mapping, dict) or mapping != CLASS_MAPPING
            or any(type(index) is not int for index in mapping.values())):
        raise ValueError("Traffic-light class mapping must exactly match the four-class runtime contract")
    return {index: name for name, index in mapping.items()}


def load_model(model_path: Path, mapping_path: Path, device: str = "cpu", *, metadata_path: Path | None = None):
    """Load a CNN checkpoint, retaining the historical ``(model, names)`` API.

    Metadata may be supplied explicitly or discovered beside the checkpoint.
    With no metadata, only the historical ResNet18/128 contract is assumed;
    architecture is never inferred from a filename or state-dict keys.
    """
    import torch

    names = _load_mapping(mapping_path)
    selected_metadata = metadata_path
    legacy = False
    if selected_metadata is None:
        sidecar = model_path.parent / "model_metadata.json"
        if sidecar.is_file():
            selected_metadata = sidecar
        else:
            legacy = True
            cnn_contract = contract("resnet18")
    if selected_metadata is not None:
        metadata = json.loads(selected_metadata.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or "cnn_contract" not in metadata:
            raise ValueError("CNN metadata must contain a cnn_contract")
        cnn_contract = validate_contract(metadata["cnn_contract"])
    if legacy:
        warnings.warn(
            "CNN metadata is missing; assuming the legacy ResNet18 128x128 contract",
            RuntimeWarning,
            stacklevel=2,
        )

    model = make_model(cnn_contract["architecture"], num_classes=len(names))
    state = torch.load(model_path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError("CNN checkpoint must be a state dictionary")
    model.load_state_dict(state, strict=True)
    model = model.eval().to(device)
    model._traffic_risk_cnn_contract = cnn_contract
    model._traffic_risk_inference_transform = inference_transform(cnn_contract["image_size"])
    return model, names


def predict_image(model, names: dict[int, str], image_path: Path, device: str = "cpu") -> tuple[str, float]:
    import torch
    from PIL import Image

    expected_names = {index: name for name, index in CLASS_MAPPING.items()}
    if names != expected_names:
        raise ValueError("Traffic-light class names do not match the runtime contract")
    transform = getattr(model, "_traffic_risk_inference_transform", None) or inference_transform()
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        label, top_probability, _, _ = classify_logits(model(tensor))
    return label, top_probability


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--metadata", type=Path, help="Optional CNN metadata JSON (defaults to sidecar or legacy ResNet18)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable classification details")
    args = parser.parse_args(argv)
    try:
        model, names = load_model(args.model, args.mapping, args.device, metadata_path=args.metadata)
        label, top_probability, raw_label, abstained = _predict_details(model, args.image, args.device)
        if args.json:
            print(json.dumps({
                "label": label,
                "top_probability": top_probability,
                "raw_label": raw_label,
                "architecture": model._traffic_risk_cnn_contract["architecture"],
                "abstained": abstained,
            }))
        else:
            print(f"{label} (top class probability {top_probability:.3f})")
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


def _predict_details(model, image_path: Path, device: str) -> tuple[str, float, str, bool]:
    import torch
    from PIL import Image

    transform = model._traffic_risk_inference_transform
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        return classify_logits(model(tensor))


if __name__ == "__main__":
    raise SystemExit(main())
