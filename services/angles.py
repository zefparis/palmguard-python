from typing import List

import numpy as np

FINGER_CHAINS = [
    [0, 1, 2, 3, 4],    # Thumb
    [0, 5, 6, 7, 8],    # Index
    [0, 9, 10, 11, 12], # Middle
    [0, 13, 14, 15, 16],# Ring
    [0, 17, 18, 19, 20],# Pinky
]


def procrustes_align(landmarks: List[dict]) -> np.ndarray:
    pts = np.array([[lm["x"], lm["y"]] for lm in landmarks], dtype=np.float64)
    pts -= pts[0]
    ref = float(np.hypot(pts[9, 0], pts[9, 1]))
    if ref > 1e-9:
        pts /= ref
    theta = np.arctan2(pts[9, 0], pts[9, 1])
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    return (pts @ R.T).astype(np.float32)


def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


INTER_FINGER_TRIPLES = [
    (1,  0,  5),   # thumb-base  ↔ index-base  spread at wrist
    (5,  0,  9),   # index-base  ↔ middle-base spread at wrist
    (9,  0, 13),   # middle-base ↔ ring-base   spread at wrist
    (13, 0, 17),   # ring-base   ↔ pinky-base  spread at wrist
    (4,  0,  8),   # thumb-tip   ↔ index-tip   spread at wrist
    (8,  0, 12),   # index-tip   ↔ middle-tip  spread at wrist
    (12, 0, 16),   # middle-tip  ↔ ring-tip    spread at wrist
    (16, 0, 20),   # ring-tip    ↔ pinky-tip   spread at wrist
    (5,  9, 17),   # palm arch: index-base ↔ pinky-base at middle-base
]


def compute_inter_finger_angles(landmarks: List[dict]) -> np.ndarray:
    if not landmarks or len(landmarks) < 21:
        return np.zeros(9, dtype=np.float32)
    pts = procrustes_align(landmarks)
    return np.array(
        [compute_angle(pts[a], pts[b], pts[c]) for a, b, c in INTER_FINGER_TRIPLES],
        dtype=np.float32,
    )


def compute_angle_features(landmarks: List[dict]) -> np.ndarray:
    if not landmarks or len(landmarks) < 21:
        return np.zeros(18, dtype=np.float32)

    pts = procrustes_align(landmarks)
    a = np.zeros(18, dtype=np.float32)
    i = 0
    for f in FINGER_CHAINS:
        a[i] = compute_angle(pts[f[0]], pts[f[1]], pts[f[2]]); i += 1
        a[i] = compute_angle(pts[f[1]], pts[f[2]], pts[f[3]]); i += 1
        a[i] = compute_angle(pts[f[2]], pts[f[3]], pts[f[4]]); i += 1

    # 3 palm arch angles
    a[15] = compute_angle(pts[5], pts[0], pts[17])
    a[16] = compute_angle(pts[5], pts[9], pts[17])
    a[17] = compute_angle(pts[1], pts[0], pts[17])
    return a
