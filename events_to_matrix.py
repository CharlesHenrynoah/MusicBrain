#!/usr/bin/env python3
"""Convert musical-event JSON patterns into event matrices + validity masks.

Convention (one row per event):
  [type, onset, duration, pitch_min, pitch_max]

Type codes:
  rest  -> 0
  chord -> 2

Pitch columns are padding (0) for rests; the mask marks them invalid.
Onset is absolute on the pattern timeline (measure_index * 8 + local onset),
with L:1/8 / 4/4 so each bar spans 8 units.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

TYPE_CODE = {"rest": 0, "chord": 2}
UNITS_PER_MEASURE = 8
FEATURE_DIM = 5  # type, onset, duration, pitch_min, pitch_max


def event_to_row(event: dict[str, Any], absolute_onset: float) -> tuple[list[float], list[int]]:
    """Return (feature_row, mask_row) for one musical event."""
    etype = event["type"]
    if etype not in TYPE_CODE:
        raise ValueError(f"Unknown event type: {etype!r}")

    duration = float(event["duration"])
    pitches = event.get("pitches") or []

    if etype == "rest":
        row = [float(TYPE_CODE["rest"]), float(absolute_onset), duration, 0.0, 0.0]
        mask = [1, 1, 1, 0, 0]
        return row, mask

    if not pitches:
        raise ValueError(f"Chord/note event without pitches: {event}")

    pitch_min = float(min(pitches))
    pitch_max = float(max(pitches))
    row = [float(TYPE_CODE["chord"]), float(absolute_onset), duration, pitch_min, pitch_max]
    mask = [1, 1, 1, 1, 1]
    return row, mask


def pattern_to_matrix(pattern: dict[str, Any]) -> dict[str, Any]:
    """Build full-pattern matrix/mask and per-measure matrices (relative onset)."""
    rows: list[list[float]] = []
    masks: list[list[int]] = []
    measures_out: list[dict[str, Any]] = []

    for measure in pattern["measures"]:
        m_idx = int(measure["measure"])
        m_rows: list[list[float]] = []
        m_masks: list[list[int]] = []

        for event in measure["events"]:
            local_onset = float(event["onset"])
            abs_onset = m_idx * UNITS_PER_MEASURE + local_onset

            abs_row, abs_mask = event_to_row(event, abs_onset)
            rows.append(abs_row)
            masks.append(abs_mask)

            # Per-measure view keeps the original local onset (as in the JSON example).
            local_row, local_mask = event_to_row(event, local_onset)
            m_rows.append(local_row)
            m_masks.append(local_mask)

        measures_out.append(
            {
                "measure": m_idx,
                "matrix": m_rows,
                "mask": m_masks,
            }
        )

    matrix = np.asarray(rows, dtype=np.float64)
    mask = np.asarray(masks, dtype=np.int8)

    return {
        "pattern_id": pattern["pattern_id"],
        "instrument": pattern["instrument"],
        "bpm": pattern["bpm"],
        "original_key": pattern["original_key"],
        "time_signature": pattern["time_signature"],
        "columns": ["type", "onset", "duration", "pitch_min", "pitch_max"],
        "type_codes": dict(TYPE_CODE),
        "units_per_measure": UNITS_PER_MEASURE,
        "onset_mode": "absolute",
        "n_events": int(matrix.shape[0]),
        "matrix": matrix.tolist(),
        "mask": mask.tolist(),
        "measures": measures_out,
    }


def instrument_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/musical_events.json"),
        help="Original musical-event JSON (left untouched).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/event_matrices"),
    )
    parser.add_argument(
        "--combined",
        type=Path,
        default=Path("data/event_matrices.json"),
    )
    parser.add_argument(
        "--npy-dir",
        type=Path,
        default=Path("data/event_matrices_npy"),
        help="Directory for .npy matrix/mask pairs.",
    )
    args = parser.parse_args()

    patterns = json.loads(args.input.read_text())
    converted = [pattern_to_matrix(p) for p in patterns]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob("*.json"):
        old.unlink()

    args.npy_dir.mkdir(parents=True, exist_ok=True)
    for old in args.npy_dir.glob("*.npy"):
        old.unlink()

    for item in converted:
        slug = instrument_slug(item["instrument"])
        stem = f"{item['pattern_id']}__{slug}"
        (args.out_dir / f"{stem}.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2) + "\n"
        )
        np.save(args.npy_dir / f"{stem}__matrix.npy", np.asarray(item["matrix"], dtype=np.float64))
        np.save(args.npy_dir / f"{stem}__mask.npy", np.asarray(item["mask"], dtype=np.int8))

    args.combined.parent.mkdir(parents=True, exist_ok=True)
    args.combined.write_text(json.dumps(converted, ensure_ascii=False, indent=2) + "\n")

    sample = next(
        p
        for p in converted
        if p["pattern_id"] == "281581" and p["instrument"] == "Synth lead"
    )
    m0 = sample["measures"][0]
    print(f"Converted {len(converted)} patterns (JSON originals untouched).")
    print(f"JSON -> {args.out_dir}/ + {args.combined}")
    print(f"NPY  -> {args.npy_dir}/")
    print("\nMeasure 0 matrix (local onset, matches the worked example):")
    print(np.asarray(m0["matrix"]))
    print("Mask:")
    print(np.asarray(m0["mask"]))


if __name__ == "__main__":
    main()
