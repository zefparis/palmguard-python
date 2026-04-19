from pydantic import BaseModel
from typing import List, Optional


class ExtractRequest(BaseModel):
    image_b64: str
    session_id: Optional[str] = None


class ExtractResponse(BaseModel):
    success: bool
    vector: Optional[List[float]] = None
    fractal: Optional[List[float]] = None
    tda: Optional[List[float]] = None
    angles: Optional[List[float]] = None
    hu_moments: Optional[List[float]] = None
    hand_detected: bool = False
    confidence: Optional[float] = None
    chirality: Optional[str] = None
    processing_ms: Optional[float] = None
    error: Optional[str] = None


class CompareRequest(BaseModel):
    vector_a: List[float]
    vector_b: List[float]


class CompareResponse(BaseModel):
    similarity: float
    matched: bool
    threshold: float
    cosine_similarity: float
    l2_similarity: float
    chirality_match: bool
