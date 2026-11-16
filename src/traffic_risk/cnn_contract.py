"""Architecture and preprocessing shared by local training and inference."""
from __future__ import annotations

CLASS_MAPPING = {"green": 0, "red": 1, "unknown": 2, "yellow": 3}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def contract(architecture: str = "resnet18") -> dict:
    if architecture not in {"resnet18", "mobilenetv3"}:
        raise ValueError("Runtime CNN architecture must be resnet18 or mobilenetv3")
    return {"version": 1, "architecture": architecture, "image_size": 128,
            "mean": MEAN, "std": STD, "class_mapping": CLASS_MAPPING}


def validate_contract(value: dict) -> dict:
    if not isinstance(value, dict) or value != contract(value.get("architecture")):
        raise ValueError("CNN preprocessing or class contract is incompatible")
    return value


def make_model(name: str, num_classes: int = 4, pretrained: bool = False):
    # torchvision model builders: https://docs.pytorch.org/vision/stable/models.html
    from torch import nn
    from torchvision import models
    contract(name)
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def inference_transform(image_size: int = 128):
    from torchvision import transforms
    return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


def classify_logits(logits, threshold: float = 0.5) -> tuple[str, float, str, bool]:
    """Apply the runtime's shared confidence abstention policy to CNN logits.

    Returns ``(label, top_probability, raw_label, abstained)``. The raw class
    remains available for experiment reports while all production callers use
    the same threshold and class mapping.
    """
    import torch

    _validate_threshold(threshold)
    values = torch.as_tensor(logits)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1 or values.numel() != len(CLASS_MAPPING):
        raise ValueError("CNN logits must contain exactly one score per contract class")
    if not values.is_floating_point():
        values = values.to(dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("CNN logits contain non-finite values")
    return classify_probabilities(torch.softmax(values, dim=0), threshold)


def classify_probabilities(probabilities, threshold: float = 0.5) -> tuple[str, float, str, bool]:
    """Apply the shared confidence policy to one four-class probability row."""
    import torch

    _validate_threshold(threshold)
    names = {index: name for name, index in CLASS_MAPPING.items()}
    values = torch.as_tensor(probabilities)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1 or values.numel() != len(CLASS_MAPPING):
        raise ValueError("CNN probabilities must contain exactly one value per contract class")
    if not values.is_floating_point():
        values = values.to(dtype=torch.float32)
    if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("CNN probabilities must be finite and between 0 and 1")
    if not torch.isclose(values.sum(), values.new_tensor(1.0), rtol=1e-5, atol=1e-6):
        raise ValueError("CNN probabilities must sum to 1")
    confidence, index = values.max(dim=0)
    top_probability = float(confidence)
    raw_label = names[int(index)]
    abstained = top_probability < threshold
    return ("unknown" if abstained else raw_label, top_probability, raw_label, abstained)


def _validate_threshold(threshold: float) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
        raise ValueError("CNN confidence threshold must be between 0 and 1")
