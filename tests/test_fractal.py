import numpy as np
import pytest

from services.fractal import box_count, theil_sen_slope, lacunarity, compute_fractal_features


def test_box_count_full():
    binary = np.ones((64, 64), dtype=np.uint8)
    assert box_count(binary, 8) == 64


def test_box_count_empty():
    binary = np.zeros((64, 64), dtype=np.uint8)
    assert box_count(binary, 8) == 0


def test_theil_sen_slope_linear():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 2.0, 4.0, 6.0])
    slope = theil_sen_slope(x, y)
    assert abs(slope - 2.0) < 1e-6


def test_lacunarity_uniform():
    binary = np.ones((32, 32), dtype=np.uint8)
    lac = lacunarity(binary, 4)
    assert lac == pytest.approx(0.0, abs=1e-6)


def test_compute_fractal_features_shape():
    lines = [np.random.randint(0, 2, (64, 64), dtype=np.uint8) * 255 for _ in range(4)]
    features = compute_fractal_features(lines)
    assert features.shape == (12,)
    assert features.dtype == np.float32


def test_compute_fractal_features_empty_lines():
    lines = [np.zeros((64, 64), dtype=np.uint8) for _ in range(4)]
    features = compute_fractal_features(lines)
    assert features.shape == (12,)
    assert np.isfinite(features).all()
