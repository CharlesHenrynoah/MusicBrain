"""LangChain tools wrapping the MusicBrain search engine."""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.tools import tool

from musicbrain.search_engine import (
    PatternCorpus,
    compare_musical,
    exact_pattern_match,
    hybrid_search,
    metadata_filter,
    musical_topk,
    vector_search_for_pattern,
)
from musicbrain.search_engine.results import SearchHit

# Shared mutable bag filled during a chat turn so the API can highlight hits in 3D.
LAST_SEARCH_CONTEXT: dict[str, Any] = {
    "hits": [],
    "query_pattern_id": None,
    "query_instrument": None,
    "metric": None,
}


_CORPUS: PatternCorpus | None = None


def _corpus() -> PatternCorpus:
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = PatternCorpus.load()
    return _CORPUS


def _hits_payload(hits: list[SearchHit], limit: int = 12) -> list[dict[str, Any]]:
    return [h.to_dict() for h in hits[:limit]]


def _remember(
    hits: list[SearchHit],
    *,
    query_pattern_id: str | None = None,
    query_instrument: str | None = None,
    metric: str | None = None,
) -> str:
    LAST_SEARCH_CONTEXT["hits"] = _hits_payload(hits)
    LAST_SEARCH_CONTEXT["query_pattern_id"] = query_pattern_id
    LAST_SEARCH_CONTEXT["query_instrument"] = query_instrument
    LAST_SEARCH_CONTEXT["metric"] = metric
    return json.dumps(
        {
            "count": len(hits),
            "hits": _hits_payload(hits),
            "query_pattern_id": query_pattern_id,
            "metric": metric,
        },
        ensure_ascii=False,
    )


def reset_search_context() -> None:
    LAST_SEARCH_CONTEXT["hits"] = []
    LAST_SEARCH_CONTEXT["query_pattern_id"] = None
    LAST_SEARCH_CONTEXT["query_instrument"] = None
    LAST_SEARCH_CONTEXT["metric"] = None


@tool
def vector_search(
    pattern_id: str,
    metric: Literal[
        "cosine",
        "dot",
        "euclidean",
        "manhattan",
        "minkowski",
        "chebyshev",
        "mahalanobis",
    ] = "cosine",
    k: int = 5,
    instrument: str | None = None,
) -> str:
    """Top-K vector search over MusicBrain 32D embeddings.

    Use pattern_id (e.g. '297249'). Default metric: cosine.
    Optional instrument (e.g. 'Synth lead', 'Keys', 'Bass'). If omitted and the id
    exists on several instruments, Synth lead is chosen automatically when available.
    """
    hits = vector_search_for_pattern(
        _corpus(), pattern_id, instrument=instrument, k=k, metric=metric
    )
    return _remember(
        hits,
        query_pattern_id=pattern_id,
        query_instrument=instrument,
        metric=f"vector:{metric}",
    )


@tool
def musical_search(
    pattern_id: str,
    metric: Literal["dtw", "edit", "lcs", "ngram", "subsequence", "jaccard"] = "jaccard",
    k: int = 5,
    instrument: str | None = None,
) -> str:
    """Musical search over events (DTW, edit, LCS, ngram, subsequence, jaccard)."""
    hits = musical_topk(
        _corpus(), pattern_id, instrument=instrument, k=k, metric=metric
    )
    return _remember(
        hits,
        query_pattern_id=pattern_id,
        query_instrument=instrument,
        metric=f"musical:{metric}",
    )


@tool
def hybrid_pattern_search(
    pattern_id: str,
    k: int = 5,
    instrument: str | None = None,
    filter_instrument: str | None = None,
    bpm_min: float | None = None,
    bpm_max: float | None = None,
    original_key: str | None = None,
) -> str:
    """Hybrid search (vector + musical + RRF) with optional metadata filters."""
    metadata: dict[str, Any] = {}
    if filter_instrument:
        metadata["instrument"] = filter_instrument
    if bpm_min is not None:
        metadata["bpm_min"] = bpm_min
    if bpm_max is not None:
        metadata["bpm_max"] = bpm_max
    if original_key:
        metadata["original_key"] = original_key

    hits = hybrid_search(
        _corpus(),
        pattern_id,
        instrument=instrument,
        k=k,
        metadata=metadata or None,
    )
    return _remember(
        hits,
        query_pattern_id=pattern_id,
        query_instrument=instrument,
        metric="hybrid:rrf",
    )


@tool
def compare_two_patterns(
    pattern_a: str,
    pattern_b: str,
    instrument_a: str | None = None,
    instrument_b: str | None = None,
) -> str:
    """Compare two patterns via musical similarities (DTW, edit, LCS, ngram, jaccard...)."""
    scores = compare_musical(
        _corpus(),
        pattern_a,
        pattern_b,
        instrument_a=instrument_a,
        instrument_b=instrument_b,
    )
    return json.dumps(
        {"pattern_a": pattern_a, "pattern_b": pattern_b, "scores": scores},
        ensure_ascii=False,
    )


@tool
def get_pattern_info(pattern_id: str, instrument: str | None = None) -> str:
    """Return metadata and a short summary for a pattern (BPM, key, instrument, event counts)."""
    corpus = _corpus()
    ref = corpus.resolve(pattern_id, instrument)
    meta = corpus.get_metadata(ref.row)
    doc = corpus.events[ref.row] or {}
    n_events = sum(len(m["events"]) for m in doc.get("measures", []))
    n_attacks = sum(
        1
        for m in doc.get("measures", [])
        for e in m["events"]
        if e.get("type") != "rest"
    )
    return json.dumps(
        {
            **meta,
            "row": ref.row,
            "n_events": n_events,
            "n_attacks": n_attacks,
            "vector_preview": corpus.vector_at(ref.row)[:8].round(4).tolist(),
        },
        ensure_ascii=False,
    )


@tool
def filter_patterns(
    instrument: str | None = None,
    original_key: str | None = None,
    bpm_min: float | None = None,
    bpm_max: float | None = None,
    limit: int = 20,
) -> str:
    """Filter the corpus by metadata (instrument, key, BPM)."""
    corpus = _corpus()
    rows = metadata_filter(
        corpus,
        instrument=instrument,
        original_key=original_key,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
    )
    sample = []
    for row in rows[:limit]:
        ref = corpus.index[row]
        meta = corpus.get_metadata(row)
        sample.append(
            {
                "row": row,
                "pattern_id": ref.pattern_id,
                "instrument": ref.instrument,
                "bpm": meta.get("bpm"),
                "original_key": meta.get("original_key"),
            }
        )
    # Expose filtered rows as soft "hits" for 3D highlight
    pseudo_hits = [
        SearchHit(
            rank=i + 1,
            row=item["row"],
            pattern_id=item["pattern_id"],
            instrument=item["instrument"],
            score=1.0,
            metric="metadata_filter",
            higher_is_better=True,
        )
        for i, item in enumerate(sample)
    ]
    _remember(pseudo_hits, metric="metadata_filter")
    return json.dumps({"total": len(rows), "sample": sample}, ensure_ascii=False)


@tool
def exact_match_search(pattern_id: str, instrument: str | None = None) -> str:
    """Find patterns whose event-token structure is identical."""
    hits = exact_pattern_match(_corpus(), pattern_id, instrument=instrument, mode="tokens")
    return _remember(
        hits,
        query_pattern_id=pattern_id,
        query_instrument=instrument,
        metric="exact:tokens",
    )


@tool
def corpus_overview() -> str:
    """MusicBrain corpus overview (size, instruments, BPM range)."""
    corpus = _corpus()
    from collections import Counter

    instruments = Counter(r.instrument for r in corpus.index)
    bpms = [m["bpm"] for m in corpus.metadata if m.get("bpm") is not None]
    return json.dumps(
        {
            "n_patterns": len(corpus),
            "dims": 32,
            "instruments": dict(instruments),
            "bpm_min": min(bpms) if bpms else None,
            "bpm_max": max(bpms) if bpms else None,
            "example_ids": [corpus.index[i].pattern_id for i in (0, 12, 13, 50)],
        },
        ensure_ascii=False,
    )


MUSICBRAIN_TOOLS = [
    corpus_overview,
    get_pattern_info,
    vector_search,
    musical_search,
    hybrid_pattern_search,
    compare_two_patterns,
    filter_patterns,
    exact_match_search,
]
