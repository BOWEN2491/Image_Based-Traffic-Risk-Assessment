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
