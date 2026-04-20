import base64
import io

import numpy as np
import pytest
from PIL import Image as PILImage

from services.pipeline import extract_palm_vector, VECTOR_DIM


def _make_blank_image_b64() -> str:
    img = PILImage.fromarray(np.zeros((480, 640, 3), dtype=np.uint8), mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return base64.b64encode(buf.getvalue()).decode()


def test_extract_palm_vector_no_hand_raises():
    b64 = _make_blank_image_b64()
    with pytest.raises(ValueError, match="NO_HAND_DETECTED"):
        extract_palm_vector(b64)


def test_vector_dim_constant():
    assert VECTOR_DIM == 75
