"""MusicBrain search engines — vector, musical, hybrid."""

from .corpus import PatternCorpus
from .distance_engine import (
    chebyshev_distance,
    euclidean_distance,
    euclidean_distance_matrix,
    mahalanobis_distance,
    manhattan_distance,
    minkowski_distance,
)
from .hybrid_search import (
    exact_pattern_match,
    hybrid_search,
    metadata_filter,
    multi_vector_search,
    reciprocal_rank_fusion,
    weighted_similarity_search,
)
from .knn_search import knn_for_pattern, knn_search
from .musical_search import compare_musical, musical_topk
from .results import SearchHit
from .similarity_engine import cosine_similarity, cosine_similarity_matrix, dot_product
from .vector_search import radius_search, vector_search_for_pattern, vector_topk

__all__ = [
    "PatternCorpus",
    "SearchHit",
    # vector similarities / distances
    "cosine_similarity",
    "cosine_similarity_matrix",
    "dot_product",
    "euclidean_distance",
    "euclidean_distance_matrix",
    "manhattan_distance",
    "minkowski_distance",
    "chebyshev_distance",
    "mahalanobis_distance",
    "vector_topk",
    "vector_search_for_pattern",
    "radius_search",
    "knn_search",
    "knn_for_pattern",
    # musical
    "musical_topk",
    "compare_musical",
    # hybrid
    "weighted_similarity_search",
    "multi_vector_search",
    "reciprocal_rank_fusion",
    "metadata_filter",
    "exact_pattern_match",
    "hybrid_search",
]
