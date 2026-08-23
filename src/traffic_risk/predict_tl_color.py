"""Standalone traffic-light color classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_model(model_path: Path, mapping_path: Path, device: str = "cpu"):
    import torch
    from torchvision import models

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(mapping))
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device), {int(value): key for key, value in mapping.items()}


def predict_image(model, names: dict[int, str], image_path: Path, device: str = "cpu") -> tuple[str, float]:
    import torch
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    confidence, index = probabilities.max(dim=0)
    return names[int(index)], float(confidence)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model, names = load_model(args.model, args.mapping, args.device)
    label, confidence = predict_image(model, names, args.image, args.device)
    print(f"{label} ({confidence:.3f})")


if __name__ == "__main__":
    main()
