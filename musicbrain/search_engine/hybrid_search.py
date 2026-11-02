"""Hybrid search — combine vector, musical and metadata criteria.

Methods
-------
- Weighted Similarity
- Multi-Vector Search (feature-block subspaces)
- Reciprocal Rank Fusion (RRF)
- Metadata Filtering
- Threshold Search
- Exact Pattern Matching
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable, Literal

import numpy as np

from .corpus import PatternCorpus
from .musical_search import (
    compare_musical,
    event_tokens,
    musical_topk,
    pitch_class_sequence,
)
from .results import SearchHit
from .vector_search import VectorMetric, vector_search_for_pattern, vector_topk

# Feature blocks from the 32-D layout
FEATURE_BLOCKS = {
    "rhythm": slice(0, 8),
    "melody": slice(8, 16),
    "harmony": slice(16, 28),
    "polyphony": slice(28, 32),
}


def metadata_filter(
    corpus: PatternCorpus,
    *,
    instrument: str | None = None,
    original_key: str | None = None,
    time_signature: str | None = None,
    bpm_min: float | None = None,
    bpm_max: float | None = None,
    pattern_ids: Iterable[str] | None = None,
) -> list[int]:
    """Return corpus row indices matching metadata constraints."""
    id_set = set(pattern_ids) if pattern_ids is not None else None
    kept: list[int] = []
    for i, meta in enumerate(corpus.metadata):
        if instrument is not None and meta.get("instrument") != instrument:
            continue
        if original_key is not None and meta.get("original_key") != original_key:
            continue
        if time_signature is not None and meta.get("time_signature") != time_signature:
            continue
        bpm = meta.get("bpm")
        if bpm_min is not None and (bpm is None or bpm < bpm_min):
            continue
        if bpm_max is not None and (bpm is None or bpm > bpm_max):
            continue
        if id_set is not None and meta.get("pattern_id") not in id_set:
            continue
        kept.append(i)
    return kept


def apply_threshold(
    hits: list[SearchHit],
    threshold: float,
) -> list[SearchHit]:
    """Keep hits that pass a similarity/distance threshold."""
    filtered: list[SearchHit] = []
    for hit in hits:
        ok = hit.score >= threshold if hit.higher_is_better else hit.score <= threshold
        if ok:
            filtered.append(
                SearchHit(
                    rank=len(filtered) + 1,
                    row=hit.row,
                    pattern_id=hit.pattern_id,
                    instrument=hit.instrument,
                    score=hit.score,
                    metric=f"threshold:{hit.metric}",
                    higher_is_better=hit.higher_is_better,
                    extra={**hit.extra, "threshold": threshold},
                )
            )
    return filtered


def exact_pattern_match(
    corpus: PatternCorpus,
    pattern_id: str,
    instrument: str | None = None,
    mode: Literal["tokens", "pitch_classes", "matrix"] = "tokens",
) -> list[SearchHit]:
    """Find patterns with identical musical structure."""
    ref = corpus.resolve(pattern_id, instrument)
    query_doc = corpus.events[ref.row]
    if query_doc is None:
        raise RuntimeError("Events not loaded")

    if mode == "tokens":
        target = event_tokens(query_doc)
    elif mode == "pitch_classes":
        target = pitch_class_sequence(query_doc)
    elif mode == "matrix":
        qmat = corpus.matrices[ref.row]
        if qmat is None:
            raise RuntimeError("Matrix not loaded")
        target_arr = np.round(qmat, 6)
    else:
        raise ValueError(mode)

    hits: list[SearchHit] = []
    for i, doc in enumerate(corpus.events):
        if i == ref.row or doc is None:
            continue
        match = False
        if mode == "tokens":
            match = event_tokens(doc) == target
        elif mode == "pitch_classes":
            match = pitch_class_sequence(doc) == target
        else:
            mat = corpus.matrices[i]
            match = mat is not None and mat.shape == target_arr.shape and np.allclose(
                np.round(mat, 6), target_arr
            )
        if match:
            r = corpus.index[i]
            hits.append(
                SearchHit(
                    rank=len(hits) + 1,
                    row=i,
                    pattern_id=r.pattern_id,
                    instrument=r.instrument,
                    score=1.0,
                    metric=f"exact:{mode}",
                    higher_is_better=True,
                )
            )
    return hits


def multi_vector_search(
    corpus: PatternCorpus,
    pattern_id: str,
    instrument: str | None = None,
    blocks: list[str] | None = None,
    k: int = 5,
) -> dict[str, list[SearchHit]]:
    """
    Search independently in feature subspaces (rhythm / melody / harmony / polyphony).

    Each block vector is L2-renormalised before cosine comparison.
    """
    ref = corpus.resolve(pattern_id, instrument)
    q_full = corpus.vector_at(ref.row)
    use_blocks = blocks or list(FEATURE_BLOCKS.keys())
    out: dict[str, list[SearchHit]] = {}

    for name in use_blocks:
        if name not in FEATURE_BLOCKS:
            raise KeyError(f"Unknown feature block: {name}")
        sl = FEATURE_BLOCKS[name]
        # Build block matrix
        block = corpus.vectors[:, sl].copy()
        q = q_full[sl].copy()
        # Renormalise rows
        bnorm = np.linalg.norm(block, axis=1, keepdims=True)
        bnorm[bnorm == 0] = 1.0
        block = block / bnorm
        qn = np.linalg.norm(q)
        q = q / qn if qn > 0 else q

        sims = block @ q
        order = np.argsort(-sims)
        hits: list[SearchHit] = []
        for row in order:
            row_i = int(row)
            if row_i == ref.row:
                continue
            r = corpus.index[row_i]
            hits.append(
                SearchHit(
                    rank=len(hits) + 1,
                    row=row_i,
                    pattern_id=r.pattern_id,
                    instrument=r.instrument,
                    score=float(sims[row_i]),
                    metric=f"multi:{name}",
                    higher_is_better=True,
                )
            )
            if len(hits) >= k:
                break
        out[name] = hits
    return out


def reciprocal_rank_fusion(
    rankings: list[list[SearchHit]],
    k_rrf: int = 60,
    top_n: int = 10,
) -> list[SearchHit]:
    """
    RRF: score(d) = Σ 1 / (k + rank_i(d))

    Combines heterogeneous rankings into one list.
    """
    scores: dict[int, float] = defaultdict(float)
    meta: dict[int, SearchHit] = {}
    for ranking in rankings:
        for hit in ranking:
            scores[hit.row] += 1.0 / (k_rrf + hit.rank)
            meta[hit.row] = hit

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    fused: list[SearchHit] = []
    for row_i, score in ordered[:top_n]:
        base = meta[row_i]
        fused.append(
            SearchHit(
                rank=len(fused) + 1,
                row=row_i,
                pattern_id=base.pattern_id,
                instrument=base.instrument,
                score=float(score),
                metric="rrf",
                higher_is_better=True,
                extra={"k_rrf": k_rrf},
            )
        )
    return fused


def weighted_similarity_search(
    corpus: PatternCorpus,
    pattern_id: str,
    instrument: str | None = None,
    weights: dict[str, float] | None = None,
    k: int = 5,
    vector_metric: VectorMetric = "cosine",
    musical_metric: str = "dtw",
) -> list[SearchHit]:
    """
    Combine vector + musical scores with configurable weights.

    Default weights: vector=0.6, musical=0.4
    Scores are assumed in [0, 1] / higher-is-better; distances are inverted softly.
    """
    w = weights or {"vector": 0.6, "musical": 0.4}
    w_sum = sum(w.values()) or 1.0
    w = {key: val / w_sum for key, val in w.items()}

    ref = corpus.resolve(pattern_id, instrument)
    # Pull generous candidate pools then fuse.
    pool = max(k * 5, 25)
    v_hits = vector_search_for_pattern(
        corpus, pattern_id, instrument, k=pool, metric=vector_metric
    )
    m_hits = musical_topk(
        corpus, pattern_id, instrument, k=pool, metric=musical_metric  # type: ignore[arg-type]
    )

    def _to_sim(hit: SearchHit) -> float:
        if hit.higher_is_better:
            return float(hit.score)
        # Convert distance → similarity
        return float(1.0 / (1.0 + hit.score))

    v_map = {h.row: _to_sim(h) for h in v_hits}
    m_map = {h.row: _to_sim(h) for h in m_hits}
    rows = set(v_map) | set(m_map)

    fused_scores: list[tuple[int, float]] = []
    for row in rows:
        score = w.get("vector", 0.0) * v_map.get(row, 0.0) + w.get(
            "musical", 0.0
        ) * m_map.get(row, 0.0)
        fused_scores.append((row, score))
    fused_scores.sort(key=lambda t: t[1], reverse=True)

    hits: list[SearchHit] = []
    for row_i, score in fused_scores[:k]:
        r = corpus.index[row_i]
        hits.append(
            SearchHit(
                rank=len(hits) + 1,
                row=row_i,
                pattern_id=r.pattern_id,
                instrument=r.instrument,
                score=float(score),
                metric="weighted",
                higher_is_better=True,
                extra={"weights": w},
            )
        )
    return hits


def hybrid_search(
    corpus: PatternCorpus,
    pattern_id: str,
    instrument: str | None = None,
    k: int = 5,
    *,
    vector_metric: VectorMetric = "cosine",
    musical_metric: str = "jaccard",
    use_rrf: bool = True,
    metadata: dict[str, Any] | None = None,
    threshold: float | None = None,
) -> list[SearchHit]:
    """
    Configurable hybrid pipeline:
    1) optional metadata pre-filter
    2) vector + musical rankings
    3) RRF fusion (or weighted)
    4) optional threshold
    """
    ref = corpus.resolve(pattern_id, instrument)
    allowed = None
    if metadata:
        allowed = set(metadata_filter(corpus, **metadata))
        if ref.row in allowed:
            allowed.remove(ref.row)

    v_hits = vector_search_for_pattern(
        corpus, pattern_id, instrument, k=max(k * 3, 15), metric=vector_metric
    )
    m_hits = musical_topk(
        corpus,
        pattern_id,
        instrument,
        k=max(k * 3, 15),
        metric=musical_metric,  # type: ignore[arg-type]
    )

    if allowed is not None:
        v_hits = [h for h in v_hits if h.row in allowed]
        m_hits = [h for h in m_hits if h.row in allowed]
        # re-rank locally
        v_hits = [
            SearchHit(i + 1, h.row, h.pattern_id, h.instrument, h.score, h.metric, h.higher_is_better)
            for i, h in enumerate(v_hits)
        ]
        m_hits = [
            SearchHit(i + 1, h.row, h.pattern_id, h.instrument, h.score, h.metric, h.higher_is_better)
            for i, h in enumerate(m_hits)
        ]

    if use_rrf:
        hits = reciprocal_rank_fusion([v_hits, m_hits], top_n=k)
    else:
        hits = weighted_similarity_search(
            corpus,
            pattern_id,
            instrument,
            k=k,
            vector_metric=vector_metric,
            musical_metric=musical_metric,
        )

    if threshold is not None:
        hits = apply_threshold(hits, threshold)
    return hits[:k]
