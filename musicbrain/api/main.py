"""FastAPI entrypoint — MusicBrain agent + 3D corpus API."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from musicbrain.api.viz3d import build_scene_payload, project_rows
from musicbrain.search_engine import PatternCorpus

app = FastAPI(title="MusicBrain API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_corpus() -> PatternCorpus:
    return PatternCorpus.load()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    api_key: str | None = Field(
        default=None,
        description="Inception API key (Mercury 3). Overrides INCEPTION_API_KEY when set.",
    )


class ChatResponse(BaseModel):
    message: str
    hits: list[dict[str, Any]] = Field(default_factory=list)
    query_pattern_id: str | None = None
    query_instrument: str | None = None
    metric: str | None = None
    tools_used: list[str] = Field(default_factory=list)
    highlight_coords: list[dict[str, Any]] = Field(default_factory=list)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/corpus/3d")
def corpus_3d() -> dict[str, Any]:
    """All patterns projected 32D → 3D (PCA) for the React scene."""
    return build_scene_payload(get_corpus())


@app.get("/api/patterns/{pattern_id}")
def pattern_detail(pattern_id: str, instrument: str | None = None) -> dict[str, Any]:
    corpus = get_corpus()
    try:
        ref = corpus.resolve(pattern_id, instrument)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    meta = corpus.get_metadata(ref.row)
    coords = project_rows([ref.row])[0]
    events_doc = corpus.events[ref.row]
    return {
        **meta,
        "row": ref.row,
        "coords": coords,
        "vector": corpus.vector_at(ref.row).tolist(),
        "measures": (events_doc or {}).get("measures", []),
        "time_signature": (events_doc or {}).get("time_signature", meta.get("time_signature")),
    }


@app.get("/api/patterns/{pattern_id}/playback")
def pattern_playback(pattern_id: str, instrument: str | None = None) -> dict[str, Any]:
    """Flattened note list for piano playback (onset/duration in seconds)."""
    corpus = get_corpus()
    try:
        ref = corpus.resolve(pattern_id, instrument)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    doc = corpus.events[ref.row]
    if doc is None:
        raise HTTPException(status_code=404, detail="No events for pattern")

    bpm = float(doc.get("bpm") or 120)
    # Musical events use L:1/8 units (eighth notes).
    seconds_per_unit = (60.0 / bpm) / 2.0
    notes: list[dict[str, Any]] = []
    for measure in doc.get("measures", []):
        m_idx = int(measure["measure"])
        for event in measure.get("events", []):
            if event.get("type") == "rest" or not event.get("pitches"):
                continue
            start = (m_idx * 8 + float(event["onset"])) * seconds_per_unit
            dur = max(float(event["duration"]) * seconds_per_unit, 0.05)
            for pitch in event["pitches"]:
                notes.append(
                    {
                        "midi": int(pitch),
                        "start": round(start, 5),
                        "duration": round(dur, 5),
                    }
                )

    notes.sort(key=lambda n: (n["start"], n["midi"]))
    total = notes[-1]["start"] + notes[-1]["duration"] if notes else 0.0
    return {
        "pattern_id": ref.pattern_id,
        "instrument": ref.instrument,
        "bpm": bpm,
        "original_key": doc.get("original_key"),
        "seconds_per_unit": seconds_per_unit,
        "duration_sec": round(total, 5),
        "notes": notes,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Mercury 3 agent turn — may call MusicBrain search tools."""
    try:
        from musicbrain.api.agent import run_agent
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        result = run_agent(
            req.message,
            history=[m.model_dump() for m in req.history],
            api_key=req.api_key,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent error: {exc}") from exc

    hits = result.get("hits") or []
    rows = [int(h["row"]) for h in hits if "row" in h]
    query_row = None
    corpus = get_corpus()
    qid = result.get("query_pattern_id")
    if qid:
        try:
            query_row = corpus.resolve(qid, result.get("query_instrument")).row
            rows = [query_row] + [r for r in rows if r != query_row]
        except KeyError:
            pass

    highlight = project_rows(rows) if rows else []
    if query_row is not None:
        for item in highlight:
            item["is_query"] = item["row"] == query_row
            item["is_hit"] = item["row"] != query_row
    else:
        for item in highlight:
            item["is_query"] = False
            item["is_hit"] = True

    return ChatResponse(
        message=result["message"],
        hits=hits,
        query_pattern_id=result.get("query_pattern_id"),
        query_instrument=result.get("query_instrument"),
        metric=result.get("metric"),
        tools_used=result.get("tools_used") or [],
        highlight_coords=highlight,
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "musicbrain.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
