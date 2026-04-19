import numpy as np
import pytest

from services.tda import extract_nodes, persistent_homology


def _make_skeleton_with_junctions() -> np.ndarray:
    sk = np.zeros((64, 64), dtype=np.uint8)
    sk[32, 10:54] = 255  # horizontal line
    sk[10:54, 32] = 255  # vertical line
    return sk


def test_extract_nodes_cross():
    sk = _make_skeleton_with_junctions()
    nodes = extract_nodes(sk)
    assert nodes.ndim == 2
    assert nodes.shape[1] == 2
    assert len(nodes) >= 1


def test_extract_nodes_empty():
    sk = np.zeros((64, 64), dtype=np.uint8)
    nodes = extract_nodes(sk)
    assert nodes.shape == (0, 2)


def test_persistent_homology_shape():
    nodes = np.random.rand(20, 2).astype(np.float32) * 64
    vec = persistent_homology(nodes)
    assert vec.shape == (32,)
    assert vec.dtype == np.float32


def test_persistent_homology_empty_nodes():
    vec = persistent_homology(np.zeros((0, 2), dtype=np.float32))
    assert vec.shape == (32,)
    assert (vec == 0).all()


def test_persistent_homology_normalized():
    nodes = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float32)
    vec = persistent_homology(nodes)
    assert (vec >= 0).all() and (vec <= 1.0 + 1e-6).all()
