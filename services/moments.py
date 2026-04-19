from typing import Tuple

import numpy as np


def _log_sign(v: float) -> float:
    return float(np.sign(v) * np.log10(1.0 + abs(v)))


def compute_hu_moments(roi: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    region = roi[y0:y1, x0:x1].astype(np.float64) / 255.0
    h, w = region.shape
    if h == 0 or w == 0:
        return np.zeros(3, dtype=np.float32)

    xs = np.arange(w, dtype=np.float64) - w / 2.0
    ys = (np.arange(h, dtype=np.float64) - h / 2.0).reshape(-1, 1)

    m00 = region.sum() + 1e-10
    mu20 = float((region * xs ** 2).sum() / m00)
    mu02 = float((region * ys ** 2).sum() / m00)
    mu11 = float((region * xs * ys).sum() / m00)
    mu30 = float((region * xs ** 3).sum() / m00)
    mu03 = float((region * ys ** 3).sum() / m00)
    mu12 = float((region * xs * ys ** 2).sum() / m00)
    mu21 = float((region * xs ** 2 * ys).sum() / m00)

    h1 = mu20 + mu02
    h2 = (mu20 - mu02) ** 2 + 4 * mu11 ** 2
    h3 = (mu30 - 3 * mu12) ** 2 + (3 * mu21 - mu03) ** 2

    return np.array([_log_sign(h1), _log_sign(h2), _log_sign(h3)], dtype=np.float32)


def compute_hu_features(roi: np.ndarray) -> np.ndarray:
    H, W = roi.shape[:2]
    hw, hh = W // 2, H // 2
    quads: list[Tuple[int, int, int, int]] = [
        (0, 0, hw, hh),
        (hw, 0, W, hh),
        (0, hh, hw, H),
        (hw, hh, W, H),
    ]
    return np.concatenate([compute_hu_moments(roi, *q) for q in quads]).astype(np.float32)
