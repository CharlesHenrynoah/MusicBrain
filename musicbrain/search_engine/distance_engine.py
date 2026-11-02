"""Vector distances on 32-D MusicBrain embeddings.

- Euclidean (L2)
- Manhattan (L1)
- Minkowski (Lp)
- Chebyshev (L∞)
- Mahalanobis (correlation-aware)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import (
    euclidean_distances as sk_euclidean_distances,
    manhattan_distances as sk_manhattan_distances,
    pairwise_distances,
)

EPS = 1e-12


def _as_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size != 32:
        raise ValueError(f"Expected a 32-D vector, got shape {np.asarray(x).shape}")
    return arr


def _as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 32:
        raise ValueError(f"Expected (N, 32), got {arr.shape}")
    return arr


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    A, B = _as_1d(a), _as_1d(b)
    diff = A - B
    return float(np.sqrt(np.dot(diff, diff)))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L1 — sum of absolute feature-wise gaps."""
    A, B = _as_1d(a), _as_1d(b)
    return float(np.sum(np.abs(A - B)))


def minkowski_distance(a: np.ndarray, b: np.ndarray, p: float = 3.0) -> float:
    """Lp generalisation of L1/L2. p→∞ approaches Chebyshev."""
    if p <= 0:
        raise ValueError("p must be > 0")
    A, B = _as_1d(a), _as_1d(b)
    return float(np.sum(np.abs(A - B) ** p) ** (1.0 / p))


def chebyshev_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L∞ — largest absolute feature-wise gap."""
    A, B = _as_1d(a), _as_1d(b)
    return float(np.max(np.abs(A - B)))


def mahalanobis_distance(
    a: np.ndarray,
    b: np.ndarray,
    cov_inv: np.ndarray,
) -> float:
    """Mahalanobis distance using a precomputed inverse covariance."""
    A, B = _as_1d(a), _as_1d(b)
    VI = np.asarray(cov_inv, dtype=np.float64)
    if VI.shape != (32, 32):
        raise ValueError(f"cov_inv must be (32, 32), got {VI.shape}")
    diff = A - B
    return float(np.sqrt(max(diff @ VI @ diff, 0.0)))


def euclidean_distance_matrix(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    X = _as_2d(X)
    if Y is None:
        return sk_euclidean_distances(X)
    return sk_euclidean_distances(X, _as_2d(Y))


def manhattan_distance_matrix(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    X = _as_2d(X)
    if Y is None:
        return sk_manhattan_distances(X)
    return sk_manhattan_distances(X, _as_2d(Y))


def minkowski_distance_matrix(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    p: float = 3.0,
) -> np.ndarray:
    X = _as_2d(X)
    if Y is None:
        return pairwise_distances(X, metric="minkowski", p=p)
    return pairwise_distances(X, _as_2d(Y), metric="minkowski", p=p)


def chebyshev_distance_matrix(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    X = _as_2d(X)
    if Y is None:
        return pairwise_distances(X, metric="chebyshev")
    return pairwise_distances(X, _as_2d(Y), metric="chebyshev")


def mahalanobis_distance_matrix(
    X: np.ndarray,
    Y: np.ndarray | None,
    cov_inv: np.ndarray,
) -> np.ndarray:
    """Pairwise Mahalanobis distances (NumPy)."""
    X = _as_2d(X)
    Y = _as_2d(Y) if Y is not None else X
    VI = np.asarray(cov_inv, dtype=np.float64)
    # d_ij^2 = (x_i - y_j) VI (x_i - y_j)^T
    # Expand: x VI x^T + y VI y^T - 2 x VI y^T
    XVI = X @ VI
    YVI = Y @ VI
    x2 = np.sum(XVI * X, axis=1)[:, None]
    y2 = np.sum(YVI * Y, axis=1)[None, :]
    cross = XVI @ Y.T
    sq = np.maximum(x2 + y2 - 2.0 * cross, 0.0)
    return np.sqrt(sq)


def explain_euclidean(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    A, B = _as_1d(a), _as_1d(b)
    diff = A - B
    sq = float(np.dot(diff, diff))
    return {
        "sum_squared_diff": sq,
        "euclidean_distance": float(np.sqrt(sq)),
        "is_identical": float(np.sqrt(sq)) < EPS,
    }
