#!/usr/bin/env python3
"""Convert NotaGen ABC patterns (from PDF / text dump) into musical event JSON."""

from __future__ import annotations

import argparse
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

NOTE_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
HEADER_RE = re.compile(
    r"^#(\d+)\s*·\s*(.+?)\s*·\s*(.+?)\s*·\s*(\d+)\s*BPM\s*$",
    re.MULTILINE,
)
MEASURE_START_RE = re.compile(r"\[r:(\d+)/7\]\[V:1\]")
# ABC durations: 4, 3/2, /, /2, // (=1/4). Bare "/" means 1/2.
DURATION_RE = re.compile(r"(\d+/\d+|/\d+|/+|\d+)")


def note_to_midi(token: str) -> int:
    """Map an ABC note token (e.g. ^A,, , =c') to MIDI pitch."""
    i = 0
    accidental = 0
    if token[i] == "^":
        accidental = 1
        i += 1
    elif token[i] == "_":
        accidental = -1
        i += 1
    elif token[i] == "=":
        accidental = 0
        i += 1

    letter = token[i]
    i += 1
    if letter not in NOTE_BASE and letter not in {c.lower() for c in NOTE_BASE}:
        raise ValueError(f"Invalid note letter in token: {token!r}")

    # ABC: C = MIDI 60 (C4); c = MIDI 72 (C5)
    if letter.isupper():
        midi = 60 + NOTE_BASE[letter]
    else:
        midi = 72 + NOTE_BASE[letter.upper()]

    while i < len(token) and token[i] == ",":
        midi -= 12
        i += 1
    while i < len(token) and token[i] == "'":
        midi += 12
        i += 1

    if i != len(token):
        raise ValueError(f"Unparsed trailing chars in note token: {token!r}")

    return midi + accidental


def parse_duration(body: str, pos: int) -> tuple[Fraction, int]:
    """Parse optional ABC duration multiplier starting at pos. Default = 1."""
    m = DURATION_RE.match(body, pos)
    if not m:
        return Fraction(1), pos
    raw = m.group(1)
    if re.fullmatch(r"/+", raw):
        # "/" => 1/2 ; "//" => 1/4 ; ...
        dur = Fraction(1, 2 ** len(raw))
    elif raw.startswith("/") and raw[1:].isdigit():
        dur = Fraction(1, int(raw[1:]))
    else:
        dur = Fraction(raw)
    return dur, m.end()


def parse_note_token(body: str, pos: int) -> tuple[str, int]:
    """Parse one ABC pitch token at pos (accidentals + letter + octaves)."""
    start = pos
    if body[pos] in "^_=":
        pos += 1
    if pos >= len(body) or body[pos].upper() not in NOTE_BASE:
        raise ValueError(f"Expected note at {pos} in {body!r}")
    pos += 1
    while pos < len(body) and body[pos] in ",'":
        pos += 1
    return body[start:pos], pos


def parse_measure_body(body: str) -> list[dict[str, Any]]:
    """
    Parse one measure body (without barline) into events.
    Units: L:1/8 (eighth notes). A 4/4 bar = 8 units.
    """
    body = body.strip().rstrip("|").strip()
    events: list[dict[str, Any]] = []
    onset = Fraction(0)
    pos = 0
    n = len(body)

    while pos < n:
        while pos < n and body[pos].isspace():
            pos += 1
        if pos >= n:
            break

        ch = body[pos]

        # Rest: z[duration]
        if ch == "z":
            pos += 1
            dur, pos = parse_duration(body, pos)
            events.append(
                {
                    "type": "rest",
                    "onset": _num(onset),
                    "duration": _num(dur),
                    "pitches": [],
                }
            )
            onset += dur
            continue

        # Chord: [notes][duration]
        if ch == "[":
            pos += 1
            pitches: list[int] = []
            while pos < n and body[pos] != "]":
                while pos < n and body[pos].isspace():
                    pos += 1
                if pos < n and body[pos] == "]":
                    break
                token, pos = parse_note_token(body, pos)
                pitches.append(note_to_midi(token))
            if pos >= n or body[pos] != "]":
                raise ValueError(f"Unclosed chord in {body!r}")
            pos += 1
            dur, pos = parse_duration(body, pos)
            events.append(
                {
                    "type": "chord",
                    "onset": _num(onset),
                    "duration": _num(dur),
                    "pitches": pitches,
                }
            )
            onset += dur
            continue

        # Single note (stored as chord with one pitch, per musical-event schema)
        if ch in "^_=" or ch.upper() in NOTE_BASE:
            token, pos = parse_note_token(body, pos)
            pitch = note_to_midi(token)
            dur, pos = parse_duration(body, pos)
            events.append(
                {
                    "type": "chord",
                    "onset": _num(onset),
                    "duration": _num(dur),
                    "pitches": [pitch],
                }
            )
            onset += dur
            continue

        raise ValueError(f"Unexpected char {ch!r} at {pos} in measure {body!r}")

    return events


def _num(value: Fraction) -> int | float:
    if value.denominator == 1:
        return int(value)
    return float(value)


def extract_measure_bodies(chunk: str) -> dict[int, str]:
    """
    Extract 8 measure bodies from a pattern chunk.

    PDF text wraps long ABC lines (sometimes mid-chord); consume from
    [r:k/7][V:1] until the next unbracketed barline '|'.
    """
    measure_map: dict[int, str] = {}
    for mm in MEASURE_START_RE.finditer(chunk):
        idx = int(mm.group(1))
        pos = mm.end()
        depth = 0
        chars: list[str] = []
        while pos < len(chunk):
            ch = chunk[pos]
            if ch == "[":
                depth += 1
                chars.append(ch)
            elif ch == "]":
                depth = max(0, depth - 1)
                chars.append(ch)
            elif ch == "|" and depth == 0:
                break
            elif ch in "\n\r":
                # Drop wraps so mid-token splits like "Ag\\nd'" / "3/\\n2" rejoin cleanly.
                pass
            else:
                chars.append(ch)
            pos += 1
        body = "".join(chars)
        body = re.sub(r"[ \t]+", " ", body).strip()
        measure_map[idx] = body
    return measure_map


def extract_patterns_from_text(text: str) -> list[dict[str, Any]]:
    headers = list(HEADER_RE.finditer(text))
    patterns: list[dict[str, Any]] = []

    for i, match in enumerate(headers):
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        chunk = text[start:end]

        pattern_id = match.group(1)
        instrument = match.group(2).strip()
        original_key = match.group(3).strip()
        bpm = int(match.group(4))

        measure_map = extract_measure_bodies(chunk)

        if len(measure_map) != 8:
            raise ValueError(
                f"Pattern {pattern_id}: expected 8 measures, got {sorted(measure_map)}"
            )

        measures = []
        for m_idx in range(8):
            body = measure_map[m_idx]
            events = parse_measure_body(body)
            total = sum(Fraction(str(e["duration"])) for e in events)
            if total != 8:
                raise ValueError(
                    f"Pattern {pattern_id} measure {m_idx}: duration sum {total} != 8 ({body})"
                )
            measures.append({"measure": m_idx, "events": events})

        patterns.append(
            {
                "pattern_id": pattern_id,
                "instrument": instrument,
                "bpm": bpm,
                "original_key": original_key,
                "time_signature": "4/4",
                "measures": measures,
            }
        )

    return patterns


def load_pdf_text(pdf_path: Path) -> str:
    import pymupdf as fitz

    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text("text") for page in doc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path(
            "/Users/cash/.cursor/projects/Users-cash-Desktop-VectorSearch/attachments/"
            "2cf9ad24-f556-4e86-a94a-d6cbc76e78ef/patternformatabc.pdf"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/musical_events"),
    )
    parser.add_argument(
        "--combined",
        type=Path,
        default=Path("data/musical_events.json"),
    )
    args = parser.parse_args()

    text = load_pdf_text(args.pdf)
    patterns = extract_patterns_from_text(text)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Clear previous single-file dumps (IDs collide across instruments).
    for old in args.out_dir.glob("*.json"):
        old.unlink()

    for pattern in patterns:
        slug = re.sub(r"[^a-z0-9]+", "_", pattern["instrument"].lower()).strip("_")
        out = args.out_dir / f"{pattern['pattern_id']}__{slug}.json"
        out.write_text(json.dumps(pattern, ensure_ascii=False, indent=2) + "\n")

    args.combined.parent.mkdir(parents=True, exist_ok=True)
    args.combined.write_text(json.dumps(patterns, ensure_ascii=False, indent=2) + "\n")

    sample = next(
        p
        for p in patterns
        if p["pattern_id"] == "281581" and p["instrument"] == "Synth lead"
    )
    print(
        f"Converted {len(patterns)} patterns "
        f"({len(list(args.out_dir.glob('*.json')))} files) "
        f"-> {args.out_dir}/ and {args.combined}"
    )
    print(json.dumps(sample, ensure_ascii=False, indent=2)[:1500])


if __name__ == "__main__":
    main()
