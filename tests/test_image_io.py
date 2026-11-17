from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageOps

from traffic_risk.image_io import DecodedImage, decode_image
from traffic_risk.model_runtime import PerceptionError


@pytest.mark.parametrize("image_format,suffix", [("JPEG", ".jpg"), ("PNG", ".png"), ("WEBP", ".webp")])
@pytest.mark.parametrize("orientation", range(1, 9))
def test_decode_image_applies_exif_orientation_once(tmp_path, image_format, suffix, orientation):
    path = tmp_path / f"orientation-{orientation}{suffix}"
    exif = Image.Exif()
    exif[274] = orientation
    Image.new("RGB", (13, 7), (200, 40, 10)).save(
        path, format=image_format, exif=exif, lossless=image_format == "WEBP"
    )

    expected = ImageOps.exif_transpose(Image.open(path))
    decoded = decode_image(path)

    assert (decoded.width, decoded.height) == expected.size
    assert decoded.bgr.shape == (decoded.height, decoded.width, 3)
    assert decoded.bgr.dtype == np.uint8
    assert decoded.bgr.flags.c_contiguous
    assert np.all(decoded.bgr[:, :, ::-1] == np.asarray(expected))


@pytest.mark.parametrize("image_format,suffix,mode", [
    ("JPEG", ".jpg", "L"), ("JPEG", ".jpg", "RGB"),
    ("PNG", ".png", "L"), ("PNG", ".png", "RGB"), ("PNG", ".png", "RGBA"),
    ("WEBP", ".webp", "L"), ("WEBP", ".webp", "RGB"), ("WEBP", ".webp", "RGBA"),
])
def test_decode_image_normalizes_supported_modes_and_formats(tmp_path, image_format, suffix, mode):
    color = {"L": 120, "RGB": (20, 80, 160), "RGBA": (20, 80, 160, 30)}[mode]
    path = tmp_path / f"mode-{mode}{suffix}"
    Image.new(mode, (11, 17), color).save(path, format=image_format)

    decoded = decode_image(path)

    assert (decoded.width, decoded.height) == (11, 17)
    assert decoded.bgr.shape == (17, 11, 3)
    assert decoded.bgr.dtype == np.uint8
    assert decoded.bgr.flags.c_contiguous


def test_decode_image_rejects_corrupt_image_as_perception_error(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")

    with pytest.raises(PerceptionError, match="could not be decoded"):
        decode_image(path)


@pytest.mark.parametrize("image", [
    np.zeros((4, 5), dtype=np.uint8),
    np.zeros((4, 5, 3), dtype=np.float32),
    np.zeros((4, 5, 4), dtype=np.uint8),
    np.zeros((4, 5, 3), dtype=np.uint8)[:, ::-1],
])
def test_decoded_image_contract_rejects_invalid_arrays(image):
    with pytest.raises((TypeError, ValueError)):
        DecodedImage(image, width=5, height=4)
