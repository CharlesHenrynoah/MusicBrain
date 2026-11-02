"""Musical sequence search on events / event-matrices (no 32-D features).

Methods
-------
- DTW                  align pitch sequences with flexible timing
- Edit Distance        Levenshtein on pitch / token sequences
- LCS                  longest common pitch subsequence
- N-gram Search        Jaccard over successive n-grams
- Subsequence Matching contiguous or non-contiguous motif search
- Jaccard Similarity   set overlap of pitch classes / pitches
"""

from __future__ import annotations

from typing import Any, Hashable, Iterable, Literal, Sequence

import numpy as np

from .corpus import PatternCorpus
from .results import SearchHit

MusicalMetric = Literal[
    "dtw",
    "edit",
    "lcs",
    "ngram",
    "subsequence",
    "jaccard",
]


# ---------------------------------------------------------------------------
# Sequence extractors
# ---------------------------------------------------------------------------

def iter_attacks(events_doc: dict[str, Any]) -> list[dict[str, Any]]:
    attacks: list[dict[str, Any]] = []
    for measure in events_doc["measures"]:
        for event in measure["events"]:
            if event["type"] != "rest" and event.get("pitches"):
                attacks.append(event)
    return attacks


def pitch_sequence(events_doc: dict[str, Any], voice: str = "max") -> list[int]:
    """Lead pitch sequence (default: max pitch per attack)."""
    seq: list[int] = []
    for event in iter_attacks(events_doc):
        pitches = event["pitches"]
        if voice == "max":
            seq.append(int(max(pitches)))
        elif voice == "min":
            seq.append(int(min(pitches)))
        elif voice == "mean":
            seq.append(int(round(sum(pitches) / len(pitches))))
        else:
            raise ValueError(voice)
    return seq


def pitch_class_sequence(events_doc: dict[str, Any], voice: str = "max") -> list[int]:
    return [p % 12 for p in pitch_sequence(events_doc, voice=voice)]


def pitch_class_set(events_doc: dict[str, Any]) -> set[int]:
    pcs: set[int] = set()
    for event in iter_attacks(events_doc):
        for p in event["pitches"]:
            pcs.add(int(p) % 12)
    return pcs


def pitch_set(events_doc: dict[str, Any]) -> set[int]:
    out: set[int] = set()
    for event in iter_attacks(events_doc):
        out.update(int(p) for p in event["pitches"])
    return out


def event_tokens(events_doc: dict[str, Any]) -> list[tuple[Hashable, ...]]:
    """Discrete tokens for edit / LCS style matching."""
    tokens: list[tuple[Hashable, ...]] = []
    for event in iter_attacks(events_doc):
        pcs = tuple(sorted({int(p) % 12 for p in event["pitches"]}))
        dur = float(event["duration"])
        # Quantize duration lightly to absorb notation noise.
        dur_q = round(dur * 2) / 2.0
        tokens.append(("chord", pcs, dur_q))
    return tokens


def matrix_pitch_sequence(matrix: np.ndarray) -> list[float]:
    """Use pitch_max column (index 4) of chord rows (type==2)."""
    seq: list[float] = []
    for row in matrix:
        if int(row[0]) == 2:  # chord
            seq.append(float(row[4]))
    return seq


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------

def dtw_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Classic DTW on 1-D sequences (absolute local cost)."""
    if not a or not b:
        return float("inf")
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(float(a[i - 1]) - float(b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def edit_distance(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    """Levenshtein distance."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return int(prev[m])


def lcs_length(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def lcs_similarity(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return lcs_length(a, b) / float(max(len(a), len(b)))


def ngrams(seq: Sequence[Hashable], n: int) -> set[tuple[Hashable, ...]]:
    if n < 1:
        raise ValueError("n must be >= 1")
    if len(seq) < n:
        return set()
    return {tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)}


def jaccard(a: Iterable[Hashable], b: Iterable[Hashable]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def is_subsequence(query: Sequence[Hashable], target: Sequence[Hashable]) -> bool:
    """Non-contiguous subsequence match."""
    if not query:
        return True
    it = iter(target)
    return all(any(t == q for t in it) for q in query)


def is_contiguous_subsequence(
    query: Sequence[Hashable],
    target: Sequence[Hashable],
) -> bool:
    """Contiguous motif (substring) match."""
    n, m = len(query), len(target)
    if n == 0:
        return True
    if n > m:
        return False
    for i in range(m - n + 1):
        if list(target[i : i + n]) == list(query):
            return True
    return False


def edit_similarity(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    if not a and not b:
        return 1.0
    d = edit_distance(a, b)
    return 1.0 - d / float(max(len(a), len(b)))


def dtw_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Map DTW distance to (0, 1] similarity via soft normalisation."""
    if not a or not b:
        return 0.0
    d = dtw_distance(a, b)
    # Normalise by path length proxy max(len(a), len(b)) * scale
    denom = max(len(a), len(b)) * 12.0  # octave-ish scale in semitones
    return float(1.0 / (1.0 + d / max(denom, 1.0)))


# ---------------------------------------------------------------------------
# Corpus search
# ---------------------------------------------------------------------------

def _pair_score(
    query_doc: dict[str, Any],
    cand_doc: dict[str, Any],
    metric: MusicalMetric,
    ngram_n: int = 3,
    subsequence_mode: str = "contiguous",
    jaccard_on: str = "pitch_class",
) -> tuple[float, bool]:
    """Return (score, higher_is_better)."""
    q_pitches = pitch_sequence(query_doc)
    c_pitches = pitch_sequence(cand_doc)
    q_pcs = pitch_class_sequence(query_doc)
    c_pcs = pitch_class_sequence(cand_doc)
    q_tokens = event_tokens(query_doc)
    c_tokens = event_tokens(cand_doc)

    if metric == "dtw":
        return dtw_similarity(q_pitches, c_pitches), True
    if metric == "edit":
        return edit_similarity(q_tokens, c_tokens), True
    if metric == "lcs":
        return lcs_similarity(q_pcs, c_pcs), True
    if metric == "ngram":
        return jaccard(ngrams(q_pcs, ngram_n), ngrams(c_pcs, ngram_n)), True
    if metric == "subsequence":
        ok = (
            is_contiguous_subsequence(q_pcs, c_pcs)
            if subsequence_mode == "contiguous"
            else is_subsequence(q_pcs, c_pcs)
        )
        # Score: coverage if match else 0; soft score via LCS ratio always useful
        base = lcs_similarity(q_pcs, c_pcs)
        return (1.0 if ok else base * 0.5), True
    if metric == "jaccard":
        if jaccard_on == "pitch":
            return jaccard(pitch_set(query_doc), pitch_set(cand_doc)), True
        return jaccard(pitch_class_set(query_doc), pitch_class_set(cand_doc)), True
    raise ValueError(f"Unknown musical metric: {metric!r}")


def musical_topk(
    corpus: PatternCorpus,
    pattern_id: str,
    instrument: str | None = None,
    k: int = 5,
    metric: MusicalMetric = "dtw",
    ngram_n: int = 3,
    subsequence_mode: str = "contiguous",
    jaccard_on: str = "pitch_class",
) -> list[SearchHit]:
    """Top-K musical search using event sequences of a corpus pattern."""
    ref = corpus.resolve(pattern_id, instrument)
    query_doc = corpus.events[ref.row]
    if query_doc is None:
        raise RuntimeError("Events not loaded on corpus")

    scored: list[tuple[int, float]] = []
    for i, doc in enumerate(corpus.events):
        if i == ref.row or doc is None:
            continue
        score, _ = _pair_score(
            query_doc,
            doc,
            metric=metric,
            ngram_n=ngram_n,
            subsequence_mode=subsequence_mode,
            jaccard_on=jaccard_on,
        )
        scored.append((i, score))

    scored.sort(key=lambda t: t[1], reverse=True)
    hits: list[SearchHit] = []
    for row_i, score in scored[:k]:
        r = corpus.index[row_i]
        hits.append(
            SearchHit(
                rank=len(hits) + 1,
                row=row_i,
                pattern_id=r.pattern_id,
                instrument=r.instrument,
                score=float(score),
                metric=f"musical:{metric}",
                higher_is_better=True,
            )
        )
    return hits


def compare_musical(
    corpus: PatternCorpus,
    pattern_a: str,
    pattern_b: str,
    instrument_a: str | None = None,
    instrument_b: str | None = None,
) -> dict[str, float]:
    """Compute all musical similarities between two patterns."""
    a = corpus.get_events(pattern_a, instrument_a)
    b = corpus.get_events(pattern_b, instrument_b)
    out: dict[str, float] = {}
    for metric in ("dtw", "edit", "lcs", "ngram", "subsequence", "jaccard"):
        score, _ = _pair_score(a, b, metric=metric)  # type: ignore[arg-type]
        out[metric] = float(score)
    # Raw distances useful for teaching
    out["dtw_distance"] = dtw_distance(pitch_sequence(a), pitch_sequence(b))
    out["edit_distance"] = float(edit_distance(event_tokens(a), event_tokens(b)))
    out["lcs_length"] = float(lcs_length(pitch_class_sequence(a), pitch_class_sequence(b)))
    return out


MUSICAL_METHODS = [
    "dtw",
    "edit",
    "lcs",
    "ngram",
    "subsequence",
    "jaccard",
]
