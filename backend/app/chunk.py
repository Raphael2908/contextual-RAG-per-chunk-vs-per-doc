"""Structure-aware chunking.

Split on natural structural boundaries first (Markdown headings, numbered legal
sections/articles/clauses, then blank-line paragraphs), with a recursive
character fallback so no chunk exceeds the target size. Clause boundaries are the
natural unit of legal meaning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_CHARS = 1500
MAX_CHARS = 2400

# Markdown headings, or legal-style numbered headers: "1.", "1.1", "Article 4",
# "Section 12", "Clause 3", "SCHEDULE 2".
_BOUNDARY = re.compile(
    r"(?m)^(?:"
    r"#{1,6}\s+.*"  # markdown heading
    r"|(?:ARTICLE|Article|SECTION|Section|CLAUSE|Clause|SCHEDULE|Schedule)\s+[\w.]+.*"
    r"|\d+(?:\.\d+)*\.?\s+\S.*"  # 1.  1.2  3.4.5
    r")$"
)


@dataclass
class Chunk:
    index: int
    text: str
    source_location: str


def chunk_text(text: str) -> list[Chunk]:
    sections = _split_on_boundaries(text)
    chunks: list[Chunk] = []
    for heading, body in sections:
        for piece in _split_to_size(body):
            chunks.append(
                Chunk(index=len(chunks), text=piece, source_location=heading or "body")
            )
    if not chunks and text.strip():  # degenerate single-blob fallback
        for piece in _split_to_size(text.strip()):
            chunks.append(Chunk(index=len(chunks), text=piece, source_location="body"))
    return chunks


def _split_on_boundaries(text: str) -> list[tuple[str, str]]:
    matches = list(_BOUNDARY.finditer(text))
    if not matches:
        return [("", text)]
    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("preamble", preamble))
    for i, m in enumerate(matches):
        heading = m.group(0).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((heading[:120], f"{heading}\n{body}".strip()))
    return sections


def _split_to_size(body: str) -> list[str]:
    """Recursive paragraph/sentence packing so each piece <= MAX_CHARS."""
    body = body.strip()
    if not body:
        return []
    if len(body) <= MAX_CHARS:
        return [body]

    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    buf = ""
    for para in paras:
        if len(para) > MAX_CHARS:
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.extend(_split_sentences(para))
            continue
        if len(buf) + len(para) + 2 > TARGET_CHARS and buf:
            pieces.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        pieces.append(buf)
    return pieces


def _split_sentences(para: str) -> list[str]:
    sentences = re.split(r"(?<=[.;:])\s+", para)
    pieces: list[str] = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) + 1 > TARGET_CHARS and buf:
            pieces.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}" if buf else s
    if buf:
        # hard cut anything still oversized
        while len(buf) > MAX_CHARS:
            pieces.append(buf[:MAX_CHARS])
            buf = buf[MAX_CHARS:]
        pieces.append(buf)
    return pieces
