import base64

import cv2
import numpy as np
import pytest

from services.pipeline import extract_palm_vector, VECTOR_DIM


def _make_blank_image_b64() -> str:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()


def test_extract_palm_vector_no_hand_raises():
    b64 = _make_blank_image_b64()
    with pytest.raises(ValueError, match="NO_HAND_DETECTED"):
        extract_palm_vector(b64)


def test_vector_dim_constant():
    assert VECTOR_DIM == 75
