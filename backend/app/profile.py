"""Document profiling — one cheap LLM pass over the whole document.

Returns a brief summary, the correct-as-of date, and key entities. This single
pass powers the contextual-enrichment blocks below; the same parent document is
prompt-cached and reused per chunk.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

from app.llm import LLM, LLMResult

_PROFILE_INSTRUCTION = (
    "You are profiling a document for a legal knowledge base. Read the document "
    "and respond with ONLY a JSON object with keys: "
    '"summary" (1-2 sentences), '
    '"effective_date" (the document\'s effective/execution/filing date as '
    'YYYY-MM-DD, or null if it states none), and '
    '"entities" (array of key parties/entities, max 8). No prose, JSON only.'
)


@dataclass
class DocProfile:
    summary: str
    effective_date: date | None
    entities: list[str] = field(default_factory=list)
    result: LLMResult | None = None  # carries token/cost metrics


def profile_document(llm: LLM, text: str, *, upload_date: date) -> DocProfile:
    # Cap the profiled text so a huge document doesn't blow the context window;
    # head+tail captures the parts that carry dates and parties.
    excerpt = _excerpt(text)
    res = llm.complete(
        instruction=_PROFILE_INSTRUCTION,
        user=excerpt,
        max_tokens=400,
    )
    data = _parse_json(res.text)
    eff = _parse_date(data.get("effective_date")) or upload_date
    entities = data.get("entities") or []
    if not isinstance(entities, list):
        entities = []
    return DocProfile(
        summary=str(data.get("summary") or "").strip() or "(no summary)",
        effective_date=eff,
        entities=[str(e) for e in entities][:8],
        result=res,
    )


def _excerpt(text: str, head: int = 6000, tail: int = 2000) -> str:
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}\n\n[...]\n\n{text[-tail:]}"


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _parse_date(value) -> date | None:
    if not value or not isinstance(value, str):
        return None
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", value)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None
