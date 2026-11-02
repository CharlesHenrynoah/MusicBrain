"""K-Nearest Neighbors over the 32-D corpus (uses vector_search metrics)."""

from __future__ import annotations

from typing import Any

from .corpus import PatternCorpus
from .results import SearchHit
from .vector_search import VectorMetric, vector_search_for_pattern, vector_topk

# Backward-compatible alias
Neighbor = SearchHit


def knn_search(
    query,
    corpus: PatternCorpus,
    k: int = 5,
    metric: VectorMetric = "cosine",
    exclude_row: int | None = None,
    use_sklearn: bool = False,
    p: float = 3.0,
) -> list[SearchHit]:
    """Top-K neighbours for an explicit query vector."""
    del use_sklearn  # kept for API compatibility; vector_search is NumPy/sklearn based
    return vector_topk(
        query,
        corpus,
        k=k,
        metric=metric,
        exclude_row=exclude_row,
        p=p,
    )


def knn_for_pattern(
    corpus: PatternCorpus,
    pattern_id: str,
    instrument: str | None = None,
    k: int = 5,
    metric: VectorMetric = "cosine",
    use_sklearn: bool = False,
    p: float = 3.0,
) -> list[SearchHit]:
    del use_sklearn
    return vector_search_for_pattern(
        corpus,
        pattern_id,
        instrument,
        k=k,
        metric=metric,
        p=p,
    )


def knn_to_dicts(hits: list[SearchHit]) -> list[dict[str, Any]]:
    return [h.to_dict() for h in hits]
