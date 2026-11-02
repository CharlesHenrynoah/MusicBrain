"""Project 32-D normalised vectors to 3-D with PCA for the React scene."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from musicbrain.search_engine import PatternCorpus


@lru_cache(maxsize=1)
def get_pca_model() -> tuple[PCA, np.ndarray, float]:
    corpus = PatternCorpus.load()
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(corpus.vectors)
    scale = float(np.abs(coords).max() + 1e-9)
    coords = coords / scale * 8.0
    return pca, coords.astype(np.float64), scale


def build_scene_payload(corpus: PatternCorpus | None = None) -> dict[str, Any]:
    corpus = corpus or PatternCorpus.load()
    pca, coords, _scale = get_pca_model()
    if coords.shape[0] != len(corpus):
        get_pca_model.cache_clear()
        pca, coords, _scale = get_pca_model()

    points: list[dict[str, Any]] = []
    for i, ref in enumerate(corpus.index):
        meta = corpus.get_metadata(i)
        points.append(
            {
                "row": i,
                "pattern_id": ref.pattern_id,
                "instrument": ref.instrument,
                "bpm": meta.get("bpm"),
                "original_key": meta.get("original_key"),
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "z": float(coords[i, 2]),
            }
        )

    return {
        "method": "pca",
        "dims_source": 32,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "points": points,
        "instruments": sorted({p["instrument"] for p in points}),
    }


def project_rows(rows: list[int]) -> list[dict[str, float]]:
    _pca, coords, _scale = get_pca_model()
    out: list[dict[str, float]] = []
    for row in rows:
        out.append(
            {
                "row": row,
                "x": float(coords[row, 0]),
                "y": float(coords[row, 1]),
                "z": float(coords[row, 2]),
            }
        )
    return out
