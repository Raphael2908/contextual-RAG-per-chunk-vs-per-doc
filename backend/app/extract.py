"""Text extraction per file type. PDF/DOCX are primary in a legal corpus."""

from __future__ import annotations

import io
import re


def file_type_of(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext or "txt"


def extract_text(data: bytes, file_type: str) -> str:
    ft = file_type.lower()
    if ft == "pdf":
        return _extract_pdf(data)
    if ft in ("docx",):
        return _extract_docx(data)
    if ft in ("html", "htm"):
        return _strip_html(data.decode("utf-8", errors="replace"))
    # md, txt, code, anything else: treat as plain text
    return data.decode("utf-8", errors="replace")


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n\n".join(parts).strip()


def _extract_docx(data: bytes) -> str:
    import docx  # python-docx

    document = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs).strip()


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\n{3,}")


def _strip_html(html: str) -> str:
    text = _TAG.sub("", html)
    return _WS.sub("\n\n", text).strip()
