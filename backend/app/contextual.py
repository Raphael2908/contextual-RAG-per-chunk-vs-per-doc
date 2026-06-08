"""Contextual enrichment — the contextual-retrieval technique.

Three context modes, swept as a benchmark axis (see benchmark/matrix.py):

- ``none``        embed the bare chunk — the no-context control.
- ``per_chunk``   prepend a per-chunk situating sentence (one LLM call per chunk):

      [doc: <id> · as of <date> · <summary>]
      <situating sentence about where this chunk sits in the document>
      <chunk text>

- ``per_document`` prepend ONE longer document-level block (one LLM call per
  document, reused verbatim across every chunk), carrying the document's identity
  (id + filename) so an agent tool can fetch the whole document:

      [doc: <id> · as of <date> · file: <name>]
      <~100-150 word paragraph describing the whole document>
      <chunk text>

The static prefix is free. Both the per-chunk situating sentence and the
per-document paragraph are produced with the full document prompt-cached
(architecture.md §Contextual retrieval), which is what makes them affordable at
corpus scale. ``per_document`` trades N calls/doc for ~1.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.chunk import Chunk
from app.llm import LLM, LLMResult
from app.profile import DocProfile

_CONTEXT_INSTRUCTION = (
    "You situate a chunk within its parent document for retrieval. Given the whole "
    "document (cached) and one chunk, write a single short sentence (<= 30 words) "
    "that says what this chunk is about and how it relates to the document. Output "
    "only that sentence, no preamble."
)

_DOC_CONTEXT_INSTRUCTION = (
    "You are writing a retrieval context block for a whole document in a legal "
    "knowledge base. Given the full document (cached), write a single paragraph of "
    "~100-150 words describing it so that any clause from it is findable by meaning: "
    "name the parties, the instrument type and purpose, key defined terms, amounts, "
    "and effective/expiry dates. Output only the paragraph, no preamble or heading."
)


@dataclass
class EnrichedChunk:
    chunk: Chunk
    context_text: str
    result: LLMResult | None = None  # per-chunk LLM metrics (None in no-context mode)


def _prefix(document_id: str, profile: DocProfile, *, file_name: str | None = None) -> str:
    as_of = profile.effective_date.isoformat() if profile.effective_date else "unknown"
    file_part = f" · file: {file_name}" if file_name else ""
    return f"[doc: {document_id} · as of {as_of} · {profile.summary}{file_part}]"


def enrich_chunk(
    llm: LLM | None,
    *,
    document_id: str,
    profile: DocProfile,
    full_text: str,
    chunk: Chunk,
) -> EnrichedChunk:
    prefix = _prefix(document_id, profile)

    if llm is None:  # no-context control: embed the bare chunk
        return EnrichedChunk(chunk=chunk, context_text=chunk.text, result=None)

    res = llm.complete(
        instruction=_CONTEXT_INSTRUCTION,
        cached_context=full_text,
        user=f"Chunk (from {chunk.source_location}):\n{chunk.text}",
        max_tokens=80,
    )
    context_text = f"{prefix}\n{res.text}\n\n{chunk.text}"
    return EnrichedChunk(chunk=chunk, context_text=context_text, result=res)


def build_document_context(
    llm: LLM,
    *,
    document_id: str,
    profile: DocProfile,
    full_text: str,
    file_name: str,
) -> tuple[str, LLMResult]:
    """Per-document mode: ONE longer context block, reused across every chunk.

    Makes a single LLM call (full document prompt-cached, like enrich_chunk) for a
    ~100-150 word paragraph, then prepends a deterministic identity line carrying
    the document id + filename — the path is assembled here, never LLM-generated,
    so it is always accurate (an agent tool can fetch the document by it).
    """
    res = llm.complete(
        instruction=_DOC_CONTEXT_INSTRUCTION,
        cached_context=full_text,
        user="Write the retrieval context paragraph for the document above.",
        max_tokens=300,
    )
    prefix = _prefix(document_id, profile, file_name=file_name)
    block = f"{prefix}\n{res.text}"
    return block, res


def append_document_context(block: str, chunk: Chunk) -> EnrichedChunk:
    """Append the shared per-document block to one chunk. No per-chunk LLM call —
    the doc-context LLMResult is attributed once at the document level."""
    return EnrichedChunk(
        chunk=chunk, context_text=f"{block}\n\n{chunk.text}", result=None
    )
