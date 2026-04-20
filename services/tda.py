from typing import List

import numpy as np
from ripser import ripser
from scipy.ndimage import convolve

KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
MAX_PTS = 64
OUT_DIM = 32


def extract_nodes_from_landmarks(landmarks: list) -> np.ndarray:
    return np.array([[lm["x"], lm["y"]] for lm in landmarks], dtype=np.float32)


def extract_nodes(skeleton: np.ndarray) -> np.ndarray:
    binary = (skeleton > 0).astype(np.uint8)
    neighbor_counts = convolve(binary, KERNEL, mode="constant", cval=0)
    y, x = np.where(binary & (neighbor_counts >= 3))
    if len(x) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.column_stack([x, y]).astype(np.float32)


def persistent_homology(nodes: np.ndarray) -> np.ndarray:
    vec = np.zeros(OUT_DIM, dtype=np.float32)
    if len(nodes) < 2:
        return vec

    if len(nodes) > MAX_PTS:
        idx = np.linspace(0, len(nodes) - 1, MAX_PTS, dtype=int)
        nodes = nodes[idx]

    diff = nodes[:, None] - nodes[None, :]
    dist = np.sqrt((diff ** 2).sum(axis=2)).astype(np.float64)

    result = ripser(dist, maxdim=1, distance_matrix=True)
    h0 = result["dgms"][0]
    h1 = result["dgms"][1] if len(result["dgms"]) > 1 else np.zeros((0, 2))

    h0_pers = sorted([float(d - b) for b, d in h0 if np.isfinite(d)], reverse=True)
    h1_pers = sorted([float(d - b) for b, d in h1 if np.isfinite(d)], reverse=True)

    max_val = max(max(h0_pers, default=0.0), max(h1_pers, default=0.0), 1e-9)

    def stats_block(arr: list) -> List[float]:
        if not arr:
            return [0.0, 0.0, 0.0, 0.0]
        a = np.array(arr) / max_val
        return [float(a.mean()), float(a.std()), float(a.max()), min(len(a) / 16.0, 1.0)]

    vec[0:4] = stats_block(h0_pers)
    vec[4:8] = stats_block(h1_pers)
    for i, v in enumerate(h0_pers[:16]):
        vec[8 + i] = v / max_val
    for i, v in enumerate(h1_pers[:8]):
        vec[24 + i] = v / max_val

    return vec
