"""Tests for MusicBrain search engines (vector / musical / hybrid)."""

from __future__ import annotations

import numpy as np
import pytest

from musicbrain.search_engine import (
    PatternCorpus,
    chebyshev_distance,
    compare_musical,
    cosine_similarity,
    cosine_similarity_matrix,
    dot_product,
    euclidean_distance,
    exact_pattern_match,
    hybrid_search,
    knn_for_pattern,
    manhattan_distance,
    metadata_filter,
    minkowski_distance,
    multi_vector_search,
    musical_topk,
    radius_search,
    reciprocal_rank_fusion,
    vector_search_for_pattern,
    weighted_similarity_search,
)
from musicbrain.search_engine.musical_search import (
    edit_distance,
    is_contiguous_subsequence,
    jaccard,
    lcs_length,
)
from musicbrain.search_engine.similarity_engine import explain_cosine


@pytest.fixture(scope="module")
def corpus() -> PatternCorpus:
    return PatternCorpus.load()


def test_corpus_shape_and_events(corpus: PatternCorpus) -> None:
    assert len(corpus) == 305
    assert corpus.vectors.shape == (305, 32)
    assert corpus.events[0] is not None
    assert corpus.matrices[0] is not None
    norms = np.linalg.norm(corpus.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_cosine_297249_vs_297250(corpus: PatternCorpus) -> None:
    a = corpus.get_vector("297249")
    b = corpus.get_vector("297250")
    sim = cosine_similarity(a, b)
    steps = explain_cosine(a, b)
    assert abs(steps["norm_a"] - 1.0) < 1e-9
    assert abs(sim - steps["dot_product"]) < 1e-9
    assert abs(sim - dot_product(a, b)) < 1e-9
    sk = float(cosine_similarity_matrix(a.reshape(1, -1), b.reshape(1, -1))[0, 0])
    assert abs(sim - sk) < 1e-9


def test_vector_distances(corpus: PatternCorpus) -> None:
    a = corpus.get_vector("297249")
    b = corpus.get_vector("297250")
    assert manhattan_distance(a, a) < 1e-12
    assert chebyshev_distance(a, a) < 1e-12
    assert minkowski_distance(a, b, p=2) == pytest.approx(euclidean_distance(a, b), rel=1e-9)
    assert minkowski_distance(a, b, p=1) == pytest.approx(manhattan_distance(a, b), rel=1e-9)
    assert chebyshev_distance(a, b) <= manhattan_distance(a, b)


def test_radius_and_topk_metrics(corpus: PatternCorpus) -> None:
    ref = corpus.resolve("297249")
    q = corpus.vector_at(ref.row)
    for metric in ("cosine", "dot", "euclidean", "manhattan", "chebyshev", "minkowski", "mahalanobis"):
        hits = vector_search_for_pattern(corpus, "297249", k=3, metric=metric)
        assert len(hits) == 3
        assert hits[0].pattern_id != "297249" or hits[0].instrument != ref.instrument

    radius_hits = radius_search(q, corpus, radius=0.5, metric="euclidean", exclude_row=ref.row)
    assert all(h.score <= 0.5 for h in radius_hits)


def test_knn_sorted(corpus: PatternCorpus) -> None:
    neighbors = knn_for_pattern(corpus, "297249", k=5, metric="cosine")
    scores = [n.score for n in neighbors]
    assert scores == sorted(scores, reverse=True)


def test_musical_algorithms_unit() -> None:
    a = ["C", "D", "E", "G"]
    b = ["C", "D", "D", "E", "G"]
    assert edit_distance(a, b) == 1
    assert lcs_length(a, b) == 4
    assert is_contiguous_subsequence(["D", "E"], b)
    assert jaccard({"C", "D", "E"}, {"C", "D", "G"}) == 0.5


def test_musical_compare_and_topk(corpus: PatternCorpus) -> None:
    scores = compare_musical(corpus, "297249", "297250")
    assert set(scores) >= {"dtw", "edit", "lcs", "ngram", "subsequence", "jaccard"}
    assert 0.0 <= scores["jaccard"] <= 1.0
    hits = musical_topk(corpus, "297249", k=5, metric="jaccard")
    assert len(hits) == 5
    assert hits[0].metric.startswith("musical:")


def test_hybrid_methods(corpus: PatternCorpus) -> None:
    rows = metadata_filter(corpus, instrument="Synth lead", bpm_min=120, bpm_max=150)
    assert rows
    assert all(corpus.metadata[i]["instrument"] == "Synth lead" for i in rows)

    multi = multi_vector_search(corpus, "297249", k=3)
    assert set(multi) == {"rhythm", "melody", "harmony", "polyphony"}

    weighted = weighted_similarity_search(corpus, "297249", k=3)
    assert len(weighted) == 3

    fused = reciprocal_rank_fusion([multi["rhythm"], multi["melody"]], top_n=5)
    assert len(fused) <= 5
    assert fused[0].metric == "rrf"

    exact = exact_pattern_match(corpus, "297249", mode="tokens")
    assert isinstance(exact, list)

    hybrid = hybrid_search(
        corpus,
        "297249",
        k=5,
        metadata={"instrument": "Synth lead"},
        threshold=None,
    )
    assert len(hybrid) == 5
