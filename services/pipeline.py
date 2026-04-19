import time
from dataclasses import dataclass

import numpy as np

from .landmarks import extract_landmarks
from .skeleton import crop_palm_roi, extract_line_segments, skeletonize
from .fractal import compute_fractal_features
from .tda import extract_nodes, persistent_homology
from .angles import compute_angle_features
from .moments import compute_hu_features

VECTOR_DIM = 74  # fractal[12] + angles[18] + hu[12] + tda[32]


@dataclass
class PalmVectorResult:
    vector: np.ndarray      # [74] float32
    fractal: np.ndarray     # [12]
    tda: np.ndarray         # [32]
    angles: np.ndarray      # [18]
    hu_moments: np.ndarray  # [12]
    chirality: str
    confidence: float
    processing_ms: float


def extract_palm_vector(image_b64: str) -> PalmVectorResult:
    t0 = time.monotonic()

    # 1. Landmarks
    lm_result = extract_landmarks(image_b64)
    landmarks = lm_result["landmarks"]
    image = lm_result["image"]

    # 2. Palm ROI (256×256 grayscale)
    roi = crop_palm_roi(image, landmarks)

    # 3. Skeleton + 4 line segments
    skeleton = skeletonize(roi)
    lines = extract_line_segments(skeleton)

    # 4. Fractal features [12]
    fractal = compute_fractal_features(lines)

    # 5. TDA persistent homology [32]
    nodes = extract_nodes(skeleton)
    tda = persistent_homology(nodes)

    # 6. Angle features [18]
    angles = compute_angle_features(landmarks)

    # 7. Hu moments [12]
    hu = compute_hu_features(roi)

    # 8. Concatenate → [74]
    combined = np.concatenate([fractal, angles, hu, tda]).astype(np.float32)
    assert len(combined) == VECTOR_DIM, f"Vector length mismatch: {len(combined)} != {VECTOR_DIM}"

    ms = (time.monotonic() - t0) * 1000.0

    return PalmVectorResult(
        vector=combined,
        fractal=fractal,
        tda=tda,
        angles=angles,
        hu_moments=hu,
        chirality=lm_result["chirality"],
        confidence=lm_result["confidence"],
        processing_ms=ms,
    )
