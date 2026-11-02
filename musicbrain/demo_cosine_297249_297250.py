#!/usr/bin/env python3
"""Premier objectif MusicBrain : Cosine Similarity entre 297249 et 297250."""

from __future__ import annotations

import numpy as np

from musicbrain.search_engine import PatternCorpus, cosine_similarity, euclidean_distance
from musicbrain.search_engine.knn_search import knn_for_pattern
from musicbrain.search_engine.similarity_engine import explain_cosine


def main() -> None:
    corpus = PatternCorpus.load()
    a = corpus.get_vector("297249")
    b = corpus.get_vector("297250")

    steps = explain_cosine(a, b)
    sim = cosine_similarity(a, b)
    dist = euclidean_distance(a, b)

    print("=" * 60)
    print("MusicBrain — Cosine Similarity (297249 vs 297250)")
    print("=" * 60)
    print(f"||A||₂ (297249)     = {steps['norm_a']:.10f}")
    print(f"||B||₂ (297250)     = {steps['norm_b']:.10f}")
    print(f"A · B               = {steps['dot_product']:.10f}")
    print(f"S(A,B) = (A·B)/(||A||||B||) = {sim:.10f}")
    print(f"Angle               = {steps['angle_degrees']:.4f}°")
    print(f"Euclidean d(A,B)    = {dist:.10f}")
    print(f"Check: √(2-2S)      = {np.sqrt(2 - 2 * sim):.10f}")

    print("\nTop-5 KNN cosine depuis 297249 :")
    for n in knn_for_pattern(corpus, "297249", k=5, metric="cosine"):
        print(
            f"  #{n.rank}  {n.pattern_id:>8}  {n.instrument:<12}  "
            f"cos={n.score:.6f}"
        )


if __name__ == "__main__":
    main()
