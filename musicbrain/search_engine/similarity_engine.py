"""Vector similarities on 32-D MusicBrain embeddings.

- Cosine Similarity
- Dot Product (alignement; equals cosine on L2-normalised vectors)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

EPS = 1e-12


def _as_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size != 32:
        raise ValueError(f"Expected a 32-D vector, got shape {np.asarray(x).shape}")
    return arr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    A = _as_1d(a)
    B = _as_1d(b)
    na = float(np.linalg.norm(A))
    nb = float(np.linalg.norm(B))
    if na < EPS or nb < EPS:
        return 0.0
    return float(np.dot(A, B) / (na * nb))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Raw alignment score A·B (on unit vectors ≡ cosine)."""
    return float(np.dot(_as_1d(a), _as_1d(b)))


def cosine_similarity_matrix(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 32:
        raise ValueError(f"Expected (N, 32), got {X.shape}")
    if Y is None:
        return sk_cosine_similarity(X)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] != 32:
        raise ValueError(f"Expected (M, 32), got {Y.shape}")
    return sk_cosine_similarity(X, Y)


def dot_product_matrix(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y is None:
        return X @ X.T
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    return X @ Y.T


def explain_cosine(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    A = _as_1d(a)
    B = _as_1d(b)
    dot = float(np.dot(A, B))
    na = float(np.linalg.norm(A))
    nb = float(np.linalg.norm(B))
    sim = 0.0 if na < EPS or nb < EPS else dot / (na * nb)
    angle = float(np.degrees(np.arccos(np.clip(sim, -1.0, 1.0))))
    return {
        "dot_product": dot,
        "norm_a": na,
        "norm_b": nb,
        "cosine_similarity": sim,
        "angle_degrees": angle,
    }
