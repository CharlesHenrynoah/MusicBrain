"""Mercury 3 (Inception) LangChain agent for MusicBrain search."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from musicbrain.api.tools import (
    LAST_SEARCH_CONTEXT,
    MUSICBRAIN_TOOLS,
    reset_search_context,
)

load_dotenv()

SYSTEM_PROMPT = """You are MusicBrain Agent, a musical assistant for the MusicBrain vector search engine.

You have tools to explore a corpus of 305 musical patterns represented as 32D vectors
(rhythm, melody, harmony, polyphony), plus their MIDI/ABC events.

Rules:
- Reply in English, concisely and musically.
- When the user asks for similar patterns, use the tools (vector_search, musical_search, hybrid_pattern_search).
- If no pattern_id is given, start with corpus_overview or filter_patterns, then propose concrete IDs.
- Some pattern_ids exist on multiple instruments. Prefer passing instrument when known
  (e.g. instrument="Synth lead"). If omitted, the engine auto-picks Synth lead when available.
- Always cite both pattern_id and instrument in results.
- Briefly explain the metric used (cosine, DTW, jaccard, RRF...).
- Do not invent pattern_ids outside the corpus.
"""


def build_llm(api_key: str | None = None) -> ChatOpenAI:
    key = (api_key or "").strip() or os.getenv("INCEPTION_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing Inception API key. Paste it in the UI or set INCEPTION_API_KEY."
        )
    return ChatOpenAI(
        model=os.getenv("INCEPTION_MODEL", "mercury-3"),
        temperature=float(os.getenv("INCEPTION_TEMPERATURE", "0.75")),
        api_key=key,
        base_url=os.getenv("INCEPTION_BASE_URL", "https://api.inceptionlabs.ai/v1"),
        max_tokens=int(os.getenv("INCEPTION_MAX_TOKENS", "4096")),
    )


def build_agent(api_key: str | None = None):
    llm = build_llm(api_key=api_key)
    return create_react_agent(llm, MUSICBRAIN_TOOLS, prompt=SYSTEM_PROMPT)


def run_agent(
    message: str,
    history: list[dict[str, str]] | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Execute one user turn.

    Returns assistant text + structured search context for the 3D UI.
    """
    reset_search_context()
    agent = build_agent(api_key=api_key)

    messages: list[Any] = []
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=message))

    result = agent.invoke({"messages": messages})
    out_messages = result.get("messages", [])
    answer = ""
    tool_trace: list[str] = []
    for msg in out_messages:
        if isinstance(msg, AIMessage):
            if msg.content:
                answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            if msg.tool_calls:
                for call in msg.tool_calls:
                    tool_trace.append(call.get("name", "tool"))
        name = getattr(msg, "name", None)
        if name:
            tool_trace.append(str(name))

    # Deduplicate tool names while preserving order
    seen: set[str] = set()
    tools_used: list[str] = []
    for name in tool_trace:
        if name not in seen:
            seen.add(name)
            tools_used.append(name)

    return {
        "message": answer or "I could not produce a response.",
        "hits": LAST_SEARCH_CONTEXT.get("hits", []),
        "query_pattern_id": LAST_SEARCH_CONTEXT.get("query_pattern_id"),
        "query_instrument": LAST_SEARCH_CONTEXT.get("query_instrument"),
        "metric": LAST_SEARCH_CONTEXT.get("metric"),
        "tools_used": tools_used,
    }
