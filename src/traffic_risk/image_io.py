"""Validated, single-pass OpenCV image decoding for inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

@dataclass(frozen=True)
class DecodedImage:
    """A display-oriented, uint8, H×W×3, C-contiguous OpenCV BGR image.

    OpenCV applies supported EXIF orientation during ``imread``. Callers must
    treat the per-request array as read-only; the frozen dataclass does not
    make the underlying ndarray immutable.
    """

    bgr: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        image = self.bgr
        if not isinstance(image, np.ndarray):
            raise TypeError("bgr must be a numpy array")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("bgr must be a uint8 HxWx3 array")
        if not image.flags.c_contiguous:
            raise ValueError("bgr must be C-contiguous")
        height, width = image.shape[:2]
        if width <= 0 or height <= 0 or (self.width, self.height) != (width, height):
            raise ValueError("decoded image dimensions must match its array")


def decode_image(image_path: str | Path) -> DecodedImage:
    """Decode a path once into an oriented BGR array or raise PerceptionError."""
    from .model_runtime import PerceptionError

    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    except (cv2.error, OSError, ValueError) as exc:
        raise PerceptionError("Image could not be decoded") from exc
    if image is None:
        raise PerceptionError("Image could not be decoded")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise PerceptionError("Image decoder returned an unsupported image")
    image = np.ascontiguousarray(image)
    height, width = image.shape[:2]
    return DecodedImage(bgr=image, width=width, height=height)
