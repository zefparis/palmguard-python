import time
from dataclasses import dataclass

import numpy as np

from .landmarks import extract_landmarks
from .angles import compute_angle_features, compute_inter_finger_angles

VECTOR_DIM = 27  # angles[18] + inter_finger_angles[9] — fully scale/distance/rotation invariant


@dataclass
class PalmVectorResult:
    vector: np.ndarray        # [27] float32 — angles only
    landmarks_vec: np.ndarray # [42] zeros — kept for schema compatibility
    angles: np.ndarray        # [18] finger-chain angles
    extra_angles: np.ndarray  # [9]  inter-finger spread angles
    chirality: str
    confidence: float
    processing_ms: float


def compare_vectors(a: list, b: list) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    cosine = float(np.dot(va, vb) / (norm_a * norm_b)) if norm_a > 1e-9 and norm_b > 1e-9 else 0.0
    l2_sim = float(1 / (1 + np.linalg.norm(va - vb)))
    return float(0.6 * cosine + 0.4 * l2_sim)


def extract_palm_vector(image_b64: str) -> PalmVectorResult:
    t0 = time.monotonic()

    # 1. Landmarks via MediaPipe
    lm_result = extract_landmarks(image_b64)
    landmarks = lm_result["landmarks"]

    # 2. Finger-chain joint angles (18) — scale/rotation invariant via Procrustes
    angles = compute_angle_features(landmarks)

    # 3. Inter-finger spread angles (9) — same invariance
    extra_angles = compute_inter_finger_angles(landmarks)

    # 4. Concatenate → [27] angles only
    combined = np.concatenate([angles, extra_angles]).astype(np.float32)
    assert len(combined) == VECTOR_DIM, f"Vector length mismatch: {len(combined)} != {VECTOR_DIM}"

    ms = (time.monotonic() - t0) * 1000.0

    return PalmVectorResult(
        vector=combined,
        landmarks_vec=np.zeros(42, dtype=np.float32),
        angles=angles,
        extra_angles=extra_angles,
        chirality=lm_result["chirality"],
        confidence=lm_result["confidence"],
        processing_ms=ms,
    )
