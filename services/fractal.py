from typing import List

import numpy as np

SCALES = [2, 4, 8, 16, 32, 64]


def box_count(binary: np.ndarray, scale: int) -> int:
    h, w = binary.shape
    count = 0
    for r in range(0, h, scale):
        for c in range(0, w, scale):
            if binary[r : r + scale, c : c + scale].any():
                count += 1
    return count


def theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    slopes = []
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if abs(dx) > 1e-10:
                slopes.append((y[j] - y[i]) / dx)
    return float(np.median(slopes)) if slopes else 1.0


def lacunarity(binary: np.ndarray, scale: int) -> float:
    h, w = binary.shape
    counts = []
    for r in range(0, h, scale):
        for c in range(0, w, scale):
            counts.append(int(binary[r : r + scale, c : c + scale].sum()))
    arr = np.array(counts, dtype=float)
    mean = arr.mean()
    return float(arr.var() / mean**2) if mean > 0 else 0.0


def compute_fractal_features(line_images: List[np.ndarray]) -> np.ndarray:
    features: List[float] = []
    for px in line_images:
        binary = (px > 0).astype(np.uint8)
        total = binary.size
        active = int(binary.sum())
        density = active / total if total > 0 else 0.0

        counts = [box_count(binary, s) for s in SCALES]
        valid = [(s, c) for s, c in zip(SCALES, counts) if c > 0]

        if len(valid) >= 2:
            log_r = np.log([1.0 / s for s, _ in valid])
            log_n = np.log([float(c) for _, c in valid])
            D = float(np.clip(theil_sen_slope(log_r, log_n), 1.0, 2.0))
        else:
            D = 1.0

        lac = lacunarity(binary, valid[0][0] if valid else 2)
        features.extend([D, lac, density])

    return np.array(features, dtype=np.float32)  # [12]
