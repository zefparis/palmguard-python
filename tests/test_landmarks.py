import base64

import cv2
import numpy as np
import pytest

from services.landmarks import decode_image, extract_landmarks


def _make_blank_image_b64(w: int = 320, h: int = 240) -> str:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()


def test_decode_image_returns_ndarray():
    b64 = _make_blank_image_b64()
    img = decode_image(b64)
    assert isinstance(img, np.ndarray)
    assert img.shape == (240, 320, 3)


def test_decode_image_invalid_raises():
    with pytest.raises(ValueError, match="Failed to decode"):
        decode_image("not_valid_base64_image")


def test_extract_landmarks_no_hand_raises():
    b64 = _make_blank_image_b64()
    with pytest.raises(ValueError, match="NO_HAND_DETECTED"):
        extract_landmarks(b64)
