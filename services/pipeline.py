import time
from dataclasses import dataclass

import numpy as np

from .landmarks import extract_landmarks
from .angles import procrustes_align, compute_angle_features

VECTOR_DIM = 75  # landmarks[42] + angles[18] + distances[15]

DISTANCE_PAIRS = [
    (0, 9),   # wrist → middle base
    (1, 4),   # thumb length
    (5, 8),   # index finger length
    (9, 12),  # middle finger length
    (13, 16), # ring finger length
    (17, 20), # pinky length
    (5, 17),  # palm width: index→pinky base
    (1, 17),  # thumb base→pinky base
    (0, 5),   # wrist→index base
    (0, 17),  # wrist→pinky base
    (4, 8),   # thumb tip→index tip
    (8, 12),  # index tip→middle tip
    (12, 16), # middle tip→ring tip
    (16, 20), # ring tip→pinky tip
    (5, 9),   # index base→middle base
]


@dataclass
class PalmVectorResult:
    vector: np.ndarray        # [75] float32
    landmarks_vec: np.ndarray # [42] Procrustes-aligned (x,y) × 21
    angles: np.ndarray        # [18]
    distances: np.ndarray     # [15] normalised inter-landmark distances
    chirality: str
    confidence: float
    processing_ms: float


def compute_distance_features(pts: np.ndarray) -> np.ndarray:
    dists = np.array(
        [float(np.linalg.norm(pts[a] - pts[b])) for a, b in DISTANCE_PAIRS],
        dtype=np.float32,
    )
    ref = float(np.linalg.norm(pts[0] - pts[9]))
    if ref > 1e-9:
        dists /= ref
    return dists


def extract_palm_vector(image_b64: str) -> PalmVectorResult:
    t0 = time.monotonic()

    # 1. Landmarks via MediaPipe
    lm_result = extract_landmarks(image_b64)
    landmarks = lm_result["landmarks"]

    # 2. Procrustes-aligned (x,y) coords → [42]
    pts = procrustes_align(landmarks)  # (21, 2) float32
    landmarks_vec = pts.flatten()

    # 3. Angle features [18]
    angles = compute_angle_features(landmarks)

    # 4. Normalised inter-landmark distances [15]
    distances = compute_distance_features(pts)

    # 5. Concatenate → [75]
    combined = np.concatenate([landmarks_vec, angles, distances]).astype(np.float32)
    assert len(combined) == VECTOR_DIM, f"Vector length mismatch: {len(combined)} != {VECTOR_DIM}"

    ms = (time.monotonic() - t0) * 1000.0

    return PalmVectorResult(
        vector=combined,
        landmarks_vec=landmarks_vec,
        angles=angles,
        distances=distances,
        chirality=lm_result["chirality"],
        confidence=lm_result["confidence"],
        processing_ms=ms,
    )
