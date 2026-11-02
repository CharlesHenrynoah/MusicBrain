#!/usr/bin/env python3
"""Embed each pattern's event sequence into a 32-D feature vector.

Source of truth: musical-event JSON (full pitch lists). The 5-column event
matrix is a lossy view; harmony / polyphony need the original pitches.

Layout
------
A. Rhythm   D1–D8
B. Melody   D9–D16   (melody voice = pitch max per attack, lead line)
C. Harmony  D17–D28  (12 pitch-class histogram, sums to 1)
D. Polyphony D29–D32
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

UNITS_PER_MEASURE = 8
N_MEASURES = 8
TIMELINE = UNITS_PER_MEASURE * N_MEASURES  # 64
VECTOR_DIM = 32

FEATURE_NAMES = [
    "n_events",
    "attacks_per_measure",
    "mean_duration",
    "median_duration",
    "silence_rate",
    "mean_inter_onset",
    "ioi_variability",
    "rhythmic_regularity",
    "mean_pitch",
    "min_pitch",
    "max_pitch",
    "pitch_range",
    "mean_interval",
    "mean_jump",
    "asc_motion_ratio",
    "desc_motion_ratio",
    *[f"pc_{i}" for i in range(12)],
    "mean_polyphony",
    "max_polyphony",
    "poly_event_ratio",
    "mean_chord_size",
]


def _slug(instrument: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", instrument.lower()).strip("_")


def _flatten_events(pattern: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for measure in pattern["measures"]:
        m_idx = int(measure["measure"])
        for event in measure["events"]:
            out.append(
                {
                    "type": event["type"],
                    "onset": float(event["onset"]),
                    "abs_onset": m_idx * UNITS_PER_MEASURE + float(event["onset"]),
                    "duration": float(event["duration"]),
                    "pitches": list(event.get("pitches") or []),
                }
            )
    return out


def embed_pattern(pattern: dict[str, Any]) -> np.ndarray:
    """Return shape (32,) float64 vector."""
    events = _flatten_events(pattern)
    attacks = [e for e in events if e["type"] != "rest" and e["pitches"]]
    rests = [e for e in events if e["type"] == "rest"]

    v = np.zeros(VECTOR_DIM, dtype=np.float64)

    # --- A. Rhythm ---------------------------------------------------------
    v[0] = float(len(events))
    v[1] = len(attacks) / float(N_MEASURES)

    durations = np.asarray([e["duration"] for e in events], dtype=np.float64)
    if len(durations):
        v[2] = float(durations.mean())
        v[3] = float(np.median(durations))

    rest_dur = float(sum(e["duration"] for e in rests))
    v[4] = rest_dur / float(TIMELINE)

    if len(attacks) >= 2:
        onsets = np.asarray([e["abs_onset"] for e in attacks], dtype=np.float64)
        ioi = np.diff(onsets)
        mean_ioi = float(ioi.mean())
        std_ioi = float(ioi.std(ddof=0))
        v[5] = mean_ioi
        v[6] = std_ioi
        v[7] = 1.0 / (1.0 + (std_ioi / mean_ioi if mean_ioi > 0 else 0.0))
    elif len(attacks) == 1:
        v[5] = 0.0
        v[6] = 0.0
        v[7] = 1.0

    # --- B. Melody (lead = max pitch of each attack) -----------------------
    if attacks:
        lead = np.asarray([max(e["pitches"]) for e in attacks], dtype=np.float64)
        all_pitches = np.asarray(
            [p for e in attacks for p in e["pitches"]], dtype=np.float64
        )
        v[8] = float(all_pitches.mean())
        v[9] = float(all_pitches.min())
        v[10] = float(all_pitches.max())
        v[11] = v[10] - v[9]

        if len(lead) >= 2:
            intervals = np.diff(lead)
            v[12] = float(intervals.mean())
            v[13] = float(np.abs(intervals).mean())
            n = float(len(intervals))
            v[14] = float(np.sum(intervals > 0) / n)
            v[15] = float(np.sum(intervals < 0) / n)

        # --- C. Harmony: pitch-class histogram ----------------------------
        pcs = np.mod(all_pitches.astype(np.int64), 12)
        hist = np.bincount(pcs, minlength=12).astype(np.float64)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum
        v[16:28] = hist

        # --- D. Polyphony -------------------------------------------------
        sizes = np.asarray([len(e["pitches"]) for e in attacks], dtype=np.float64)
        v[28] = float(sizes.mean())
        v[29] = float(sizes.max())
        v[30] = float(np.mean(sizes >= 2))
        chord_sizes = sizes[sizes >= 2]
        v[31] = float(chord_sizes.mean()) if len(chord_sizes) else 0.0

    return v


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/musical_events.json"),
        help="Original musical-event JSON (unchanged).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/vectors32"),
    )
    parser.add_argument(
        "--combined",
        type=Path,
        default=Path("data/vectors32.json"),
    )
    parser.add_argument(
        "--npy",
        type=Path,
        default=Path("data/vectors32.npy"),
        help="Stacked array of shape (N, 32).",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("data/vectors32_meta.json"),
    )
    args = parser.parse_args()

    patterns = json.loads(args.input.read_text())
    records: list[dict[str, Any]] = []
    matrix = np.zeros((len(patterns), VECTOR_DIM), dtype=np.float64)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob("*.json"):
        old.unlink()
    for old in args.out_dir.glob("*.npy"):
        old.unlink()

    for i, pattern in enumerate(patterns):
        vec = embed_pattern(pattern)
        matrix[i] = vec
        slug = _slug(pattern["instrument"])
        stem = f"{pattern['pattern_id']}__{slug}"
        record = {
            "pattern_id": pattern["pattern_id"],
            "instrument": pattern["instrument"],
            "bpm": pattern["bpm"],
            "original_key": pattern["original_key"],
            "time_signature": pattern["time_signature"],
            "feature_names": FEATURE_NAMES,
            "vector": vec.tolist(),
        }
        records.append(record)
        (args.out_dir / f"{stem}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n"
        )
        np.save(args.out_dir / f"{stem}.npy", vec)

    args.combined.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    np.save(args.npy, matrix)
    args.meta.write_text(
        json.dumps(
            {
                "dim": VECTOR_DIM,
                "feature_names": FEATURE_NAMES,
                "n_patterns": len(patterns),
                "units_per_measure": UNITS_PER_MEASURE,
                "n_measures": N_MEASURES,
                "melody_voice": "pitch_max",
                "type_codes_ref": {"rest": 0, "chord": 2},
                "index": [
                    {
                        "row": i,
                        "pattern_id": p["pattern_id"],
                        "instrument": p["instrument"],
                    }
                    for i, p in enumerate(patterns)
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    sample = next(
        r
        for r in records
        if r["pattern_id"] == "281581" and r["instrument"] == "Synth lead"
    )
    print(f"Embedded {len(records)} patterns -> {VECTOR_DIM}-D vectors")
    print(f"JSON  -> {args.out_dir}/ + {args.combined}")
    print(f"NPY   -> {args.npy}  shape={matrix.shape}")
    print("\n#281581 Synth lead:")
    for name, value in zip(FEATURE_NAMES, sample["vector"]):
        print(f"  {name:20s} {value:.6f}")


if __name__ == "__main__":
    main()
