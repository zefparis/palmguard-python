import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

VECTOR_DIM = 74


def _rand_vec() -> list:
    v = np.random.rand(VECTOR_DIM).astype(np.float32)
    return v.tolist()


def test_compare_identical_vectors():
    v = _rand_vec()
    resp = client.post("/biometric/compare", json={"vector_a": v, "vector_b": v})
    assert resp.status_code == 200
    data = resp.json()
    assert data["cosine_similarity"] == pytest.approx(1.0, abs=1e-4)
    assert data["similarity"] > 0.95


def test_compare_orthogonal_vectors():
    a = [1.0] + [0.0] * (VECTOR_DIM - 1)
    b = [0.0, 1.0] + [0.0] * (VECTOR_DIM - 2)
    resp = client.post("/biometric/compare", json={"vector_a": a, "vector_b": b})
    assert resp.status_code == 200
    data = resp.json()
    assert data["cosine_similarity"] == pytest.approx(0.0, abs=1e-4)
    assert not data["matched"]


def test_compare_returns_expected_fields():
    v = _rand_vec()
    resp = client.post("/biometric/compare", json={"vector_a": v, "vector_b": v})
    data = resp.json()
    for key in ["similarity", "matched", "threshold", "cosine_similarity", "l2_similarity", "chirality_match"]:
        assert key in data


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
