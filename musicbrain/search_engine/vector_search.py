"""Vector search over the 32-D normalised corpus.

Metrics
-------
Similarities (higher is better): cosine, dot
Distances   (lower is better):  euclidean, manhattan, minkowski, chebyshev, mahalanobis

Also provides Radius Search: all patterns within a score/distance threshold.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np

from .corpus import PatternCorpus
from .distance_engine import (
    chebyshev_distance_matrix,
    euclidean_distance_matrix,
    mahalanobis_distance_matrix,
    manhattan_distance_matrix,
    minkowski_distance_matrix,
)
from .results import SearchHit
from .similarity_engine import cosine_similarity_matrix, dot_product_matrix

VectorMetric = Literal[
    "cosine",
    "dot",
    "euclidean",
    "manhattan",
    "minkowski",
    "chebyshev",
    "mahalanobis",
]

HIGHER_IS_BETTER = {"cosine", "dot"}


def _score_matrix(
    query: np.ndarray,
    corpus: PatternCorpus,
    metric: VectorMetric,
    p: float = 3.0,
) -> np.ndarray:
    q = np.asarray(query, dtype=np.float64).reshape(1, -1)
    if q.shape[1] != 32:
        raise ValueError(f"Expected 32-D query, got {q.shape}")

    if metric == "cosine":
        return cosine_similarity_matrix(q, corpus.vectors)[0]
    if metric == "dot":
        return dot_product_matrix(q, corpus.vectors)[0]
    if metric == "euclidean":
        return euclidean_distance_matrix(q, corpus.vectors)[0]
    if metric == "manhattan":
        return manhattan_distance_matrix(q, corpus.vectors)[0]
    if metric == "minkowski":
        return minkowski_distance_matrix(q, corpus.vectors, p=p)[0]
    if metric == "chebyshev":
        return chebyshev_distance_matrix(q, corpus.vectors)[0]
    if metric == "mahalanobis":
        if corpus.cov_inv is None or corpus.vectors_zscore is None:
            raise RuntimeError("Mahalanobis requires z-score vectors + cov_inv on corpus")
        # Compare in z-score space for consistent covariance.
        # Map query: if query is L2-normalised, mahalanobis should use matching space.
        # Convention: use normalised vectors with cov estimated on z-score of originals
        # is inconsistent. Better: transform is not available for arbitrary query.
        # Practical choice: compute mahalanobis in z-score space when query_row known,
        # else use normalised vectors with cov_inv of normalised space.
        base = corpus.vectors_zscore
        # Build cov_inv already from zscore in corpus.load — query must be zscored.
        # For API simplicity: also accept normalised query by using cov on normalised.
        cov_inv_norm = np.linalg.pinv(
            np.cov(corpus.vectors, rowvar=False) + np.eye(32) * 1e-6
        )
        return mahalanobis_distance_matrix(q, corpus.vectors, cov_inv_norm)[0]
    raise ValueError(f"Unknown vector metric: {metric!r}")


def vector_topk(
    query: np.ndarray,
    corpus: PatternCorpus,
    k: int = 5,
    metric: VectorMetric = "cosine",
    exclude_row: int | None = None,
    p: float = 3.0,
) -> list[SearchHit]:
    """Top-K vector search for any supported metric."""
    if k < 1:
        raise ValueError("k must be >= 1")
    scores = _score_matrix(query, corpus, metric, p=p)
    higher = metric in HIGHER_IS_BETTER
    order = np.argsort(-scores) if higher else np.argsort(scores)

    hits: list[SearchHit] = []
    for row in order:
        row_i = int(row)
        if exclude_row is not None and row_i == exclude_row:
            continue
        ref = corpus.index[row_i]
        hits.append(
            SearchHit(
                rank=len(hits) + 1,
                row=row_i,
                pattern_id=ref.pattern_id,
                instrument=ref.instrument,
                score=float(scores[row_i]),
                metric=metric,
                higher_is_better=higher,
            )
        )
        if len(hits) >= k:
            break
    return hits


def radius_search(
    query: np.ndarray,
    corpus: PatternCorpus,
    radius: float,
    metric: VectorMetric = "euclidean",
    exclude_row: int | None = None,
    p: float = 3.0,
    max_results: int | None = None,
) -> list[SearchHit]:
    """
    Return all patterns within a radius / above a similarity threshold.

    - For distance metrics: keep score <= radius
    - For similarity metrics: keep score >= radius
    """
    scores = _score_matrix(query, corpus, metric, p=p)
    higher = metric in HIGHER_IS_BETTER

    if higher:
        mask = scores >= radius
        order = np.argsort(-scores)
    else:
        mask = scores <= radius
        order = np.argsort(scores)

    hits: list[SearchHit] = []
    for row in order:
        row_i = int(row)
        if not mask[row_i]:
            continue
        if exclude_row is not None and row_i == exclude_row:
            continue
        ref = corpus.index[row_i]
        hits.append(
            SearchHit(
                rank=len(hits) + 1,
                row=row_i,
                pattern_id=ref.pattern_id,
                instrument=ref.instrument,
                score=float(scores[row_i]),
                metric=f"radius:{metric}",
                higher_is_better=higher,
                extra={"radius": radius},
            )
        )
        if max_results is not None and len(hits) >= max_results:
            break
    return hits


def vector_search_for_pattern(
    corpus: PatternCorpus,
    pattern_id: str,
    instrument: str | None = None,
    k: int = 5,
    metric: VectorMetric = "cosine",
    p: float = 3.0,
) -> list[SearchHit]:
    ref = corpus.resolve(pattern_id, instrument)
    return vector_topk(
        corpus.vector_at(ref.row),
        corpus,
        k=k,
        metric=metric,
        exclude_row=ref.row,
        p=p,
    )


# Keep a typed registry for discovery / CLI help
VECTOR_METHODS: dict[str, dict[str, Any]] = {
    "cosine": {"kind": "similarity", "fn": cosine_similarity_matrix},
    "dot": {"kind": "similarity", "fn": dot_product_matrix},
    "euclidean": {"kind": "distance", "fn": euclidean_distance_matrix},
    "manhattan": {"kind": "distance", "fn": manhattan_distance_matrix},
    "minkowski": {"kind": "distance", "fn": minkowski_distance_matrix},
    "chebyshev": {"kind": "distance", "fn": chebyshev_distance_matrix},
    "mahalanobis": {"kind": "distance", "fn": mahalanobis_distance_matrix},
    "radius": {"kind": "filter", "fn": radius_search},
}
