from fastapi import APIRouter
import numpy as np

from models.schemas import (
    ExtractRequest, ExtractResponse,
    CompareRequest, CompareResponse,
)
from services.pipeline import extract_palm_vector

router = APIRouter(tags=["biometric"])

THRESHOLD = 0.97


@router.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest):
    try:
        result = extract_palm_vector(req.image_b64)
        return ExtractResponse(
            success=True,
            vector=result.vector.tolist(),
            landmarks_vec=result.landmarks_vec.tolist(),
            angles=result.angles.tolist(),
            distances=result.distances.tolist(),
            hand_detected=True,
            confidence=round(float(result.confidence), 4),
            chirality=result.chirality,
            processing_ms=round(result.processing_ms, 1),
        )
    except ValueError as e:
        if "NO_HAND_DETECTED" in str(e):
            return ExtractResponse(success=False, hand_detected=False, error="NO_HAND_DETECTED")
        return ExtractResponse(success=False, hand_detected=False, error=str(e))
    except Exception as e:
        return ExtractResponse(success=False, hand_detected=False, error=str(e))


@router.post("/compare", response_model=CompareResponse)
async def compare(req: CompareRequest):
    a = np.array(req.vector_a, dtype=np.float32)
    b = np.array(req.vector_b, dtype=np.float32)

    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        cosine = 0.0
    else:
        cosine = float(np.dot(a, b) / (norm_a * norm_b))

    l2_sim = float(1.0 / (1.0 + np.linalg.norm(a - b)))
    similarity = 0.6 * cosine + 0.4 * l2_sim

    return CompareResponse(
        similarity=round(similarity, 4),
        matched=similarity >= THRESHOLD,
        threshold=THRESHOLD,
        cosine_similarity=round(cosine, 4),
        l2_similarity=round(l2_sim, 4),
        chirality_match=True,
    )
