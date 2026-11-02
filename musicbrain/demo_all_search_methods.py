#!/usr/bin/env python3
"""Demo — toutes les méthodes de recherche MusicBrain sur 297249."""

from __future__ import annotations

from musicbrain.search_engine import (
    PatternCorpus,
    compare_musical,
    exact_pattern_match,
    hybrid_search,
    metadata_filter,
    multi_vector_search,
    musical_topk,
    radius_search,
    vector_search_for_pattern,
    weighted_similarity_search,
)


def _show(title: str, hits, limit: int = 3) -> None:
    print(f"\n{title}")
    for h in hits[:limit]:
        print(
            f"  #{h.rank:<2} {h.pattern_id:>8}  {h.instrument:<12}  "
            f"score={h.score:.4f}  [{h.metric}]"
        )


def main() -> None:
    corpus = PatternCorpus.load()
    pid = "297249"
    print(f"Corpus: {len(corpus)} patterns — query={pid}")

    print("\n=== 1. Recherche vectorielle (32D) ===")
    for metric in ("cosine", "dot", "euclidean", "manhattan", "minkowski", "chebyshev", "mahalanobis"):
        hits = vector_search_for_pattern(corpus, pid, k=3, metric=metric)
        _show(f"{metric}", hits)

    ref = corpus.resolve(pid)
    radius_hits = radius_search(
        corpus.vector_at(ref.row),
        corpus,
        radius=0.8,
        metric="cosine",
        exclude_row=ref.row,
        max_results=5,
    )
    _show("radius cosine ≥ 0.8", radius_hits, limit=5)

    print("\n=== 2. Recherche musicale (events) ===")
    pair = compare_musical(corpus, "297249", "297250")
    print("297249 vs 297250:")
    for k, v in pair.items():
        print(f"  {k:16s} {v:.4f}")
    for metric in ("dtw", "edit", "lcs", "ngram", "subsequence", "jaccard"):
        hits = musical_topk(corpus, pid, k=3, metric=metric)
        _show(f"musical:{metric}", hits)

    print("\n=== 3. Recherche hybride ===")
    n_lead = len(metadata_filter(corpus, instrument="Synth lead", bpm_min=130, bpm_max=140))
    print(f"Metadata filter Synth lead BPM 130–140: {n_lead} patterns")
    _show("weighted", weighted_similarity_search(corpus, pid, k=3))
    multi = multi_vector_search(corpus, pid, k=2)
    for block, hits in multi.items():
        _show(f"multi:{block}", hits, limit=2)
    _show("exact tokens", exact_pattern_match(corpus, pid, mode="tokens"), limit=5)
    _show(
        "hybrid RRF + Synth lead",
        hybrid_search(corpus, pid, k=5, metadata={"instrument": "Synth lead"}),
        limit=5,
    )


if __name__ == "__main__":
    main()
