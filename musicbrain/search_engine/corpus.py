"""Load the MusicBrain corpus: vectors, metadata, events, matrices."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VECTOR_DIR = PACKAGE_ROOT / "data" / "normalized_vectors"
DEFAULT_EVENTS_DIR = PROJECT_ROOT / "data" / "musical_events"
DEFAULT_MATRICES_DIR = PROJECT_ROOT / "data" / "event_matrices"


def _slug(instrument: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", instrument.lower()).strip("_")


@dataclass(frozen=True)
class PatternRef:
    row: int
    pattern_id: str
    instrument: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.pattern_id, self.instrument)

    @property
    def stem(self) -> str:
        return f"{self.pattern_id}__{_slug(self.instrument)}"


class PatternCorpus:
    """In-memory catalogue for vector + musical + hybrid search."""

    def __init__(
        self,
        vectors: np.ndarray,
        index: list[dict[str, Any]],
        feature_names: list[str] | None = None,
        vectors_zscore: np.ndarray | None = None,
        vectors_original: np.ndarray | None = None,
        events: list[dict[str, Any] | None] | None = None,
        matrices: list[np.ndarray | None] | None = None,
        cov_inv: np.ndarray | None = None,
    ) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != 32:
            raise ValueError(f"Expected (N, 32) vectors, got {vectors.shape}")
        if len(index) != vectors.shape[0]:
            raise ValueError("Index length must match number of vectors")

        self.vectors = np.asarray(vectors, dtype=np.float64)
        self.vectors_zscore = (
            np.asarray(vectors_zscore, dtype=np.float64)
            if vectors_zscore is not None
            else None
        )
        self.vectors_original = (
            np.asarray(vectors_original, dtype=np.float64)
            if vectors_original is not None
            else None
        )
        self.index = [
            PatternRef(
                row=int(item.get("row", i)),
                pattern_id=str(item["pattern_id"]),
                instrument=str(item["instrument"]),
            )
            for i, item in enumerate(index)
        ]
        self.feature_names = feature_names or []
        self.events = events or [None] * len(self.index)
        self.matrices = matrices or [None] * len(self.index)
        self.cov_inv = cov_inv

        self._by_id: dict[str, list[int]] = {}
        self._by_key: dict[tuple[str, str], int] = {}
        for i, ref in enumerate(self.index):
            self._by_id.setdefault(ref.pattern_id, []).append(i)
            self._by_key[ref.key] = i

        # Metadata cache from events JSON
        self.metadata: list[dict[str, Any]] = []
        for i, doc in enumerate(self.events):
            ref = self.index[i]
            if doc is None:
                self.metadata.append(
                    {
                        "pattern_id": ref.pattern_id,
                        "instrument": ref.instrument,
                        "bpm": None,
                        "original_key": None,
                        "time_signature": None,
                    }
                )
            else:
                self.metadata.append(
                    {
                        "pattern_id": doc["pattern_id"],
                        "instrument": doc["instrument"],
                        "bpm": doc.get("bpm"),
                        "original_key": doc.get("original_key"),
                        "time_signature": doc.get("time_signature"),
                    }
                )

    @classmethod
    def load(
        cls,
        data_dir: Path | str | None = None,
        events_dir: Path | str | None = None,
        matrices_dir: Path | str | None = None,
        load_events: bool = True,
    ) -> "PatternCorpus":
        root = Path(data_dir) if data_dir else DEFAULT_VECTOR_DIR
        ev_dir = Path(events_dir) if events_dir else DEFAULT_EVENTS_DIR
        mx_dir = Path(matrices_dir) if matrices_dir else DEFAULT_MATRICES_DIR

        vectors = np.load(root / "patterns_32d_normalized.npy")
        index = json.loads((root / "pattern_index.json").read_text())
        feature_names: list[str] = []
        scaler_path = root / "scaler_params.json"
        if scaler_path.exists():
            scaler = json.loads(scaler_path.read_text())
            feature_names = list(scaler.get("feature_names") or [])

        z_path = root / "patterns_32d_zscore.npy"
        o_path = root / "patterns_32d_original.npy"
        vectors_zscore = np.load(z_path) if z_path.exists() else None
        vectors_original = np.load(o_path) if o_path.exists() else None

        # Mahalanobis: inverse covariance from z-scored features (stable).
        cov_inv = None
        if vectors_zscore is not None:
            cov = np.cov(vectors_zscore, rowvar=False)
            # Ridge for numerical stability if needed
            cov = cov + np.eye(cov.shape[0]) * 1e-6
            cov_inv = np.linalg.pinv(cov)

        events: list[dict[str, Any] | None] = []
        matrices: list[np.ndarray | None] = []
        for item in index:
            ref = PatternRef(
                row=int(item.get("row", 0)),
                pattern_id=str(item["pattern_id"]),
                instrument=str(item["instrument"]),
            )
            if load_events:
                ep = ev_dir / f"{ref.stem}.json"
                mp = mx_dir / f"{ref.stem}.json"
                events.append(json.loads(ep.read_text()) if ep.exists() else None)
                if mp.exists():
                    mdoc = json.loads(mp.read_text())
                    matrices.append(np.asarray(mdoc["matrix"], dtype=np.float64))
                else:
                    matrices.append(None)
            else:
                events.append(None)
                matrices.append(None)

        return cls(
            vectors=vectors,
            index=index,
            feature_names=feature_names,
            vectors_zscore=vectors_zscore,
            vectors_original=vectors_original,
            events=events,
            matrices=matrices,
            cov_inv=cov_inv,
        )

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def vector_at(self, row: int) -> np.ndarray:
        return self.vectors[row]

    # When the same pattern_id exists on several instruments, prefer this order.
    INSTRUMENT_PRIORITY = ("Synth lead", "Keys", "Guitar", "Bass", "Bell")

    def resolve_all(self, pattern_id: str) -> list[PatternRef]:
        rows = self._by_id.get(pattern_id)
        if not rows:
            raise KeyError(f"Unknown pattern_id {pattern_id!r}")
        return [self.index[i] for i in rows]

    def resolve(
        self,
        pattern_id: str,
        instrument: str | None = None,
        *,
        strict: bool = False,
    ) -> PatternRef:
        """Resolve a pattern id to one corpus row.

        If ``instrument`` is omitted and several instruments share the id,
        pick by ``INSTRUMENT_PRIORITY`` unless ``strict=True``.
        """
        if instrument is not None:
            key = (pattern_id, instrument)
            if key not in self._by_key:
                # Soft match: case-insensitive / slug-friendly
                for cand_key, row in self._by_key.items():
                    if cand_key[0] == pattern_id and cand_key[1].lower() == instrument.lower():
                        return self.index[row]
                raise KeyError(f"Unknown pattern {pattern_id!r} / {instrument!r}")
            return self.index[self._by_key[key]]

        refs = self.resolve_all(pattern_id)
        if len(refs) == 1:
            return refs[0]

        if strict:
            instruments = [r.instrument for r in refs]
            raise KeyError(
                f"pattern_id {pattern_id!r} is ambiguous across {instruments}; "
                "pass instrument="
            )

        by_name = {r.instrument: r for r in refs}
        for preferred in self.INSTRUMENT_PRIORITY:
            if preferred in by_name:
                return by_name[preferred]
        return refs[0]

    def get_vector(
        self,
        pattern_id: str,
        instrument: str | None = None,
    ) -> np.ndarray:
        ref = self.resolve(pattern_id, instrument)
        return self.vector_at(ref.row)

    def get_events(
        self,
        pattern_id: str,
        instrument: str | None = None,
    ) -> dict[str, Any]:
        ref = self.resolve(pattern_id, instrument)
        doc = self.events[ref.row]
        if doc is None:
            raise KeyError(f"No events loaded for {ref.stem}")
        return doc

    def get_matrix(
        self,
        pattern_id: str,
        instrument: str | None = None,
    ) -> np.ndarray:
        ref = self.resolve(pattern_id, instrument)
        mat = self.matrices[ref.row]
        if mat is None:
            raise KeyError(f"No matrix loaded for {ref.stem}")
        return mat

    def get_metadata(self, row: int) -> dict[str, Any]:
        return self.metadata[row]
