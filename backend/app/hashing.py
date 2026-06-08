"""Cheap dedup hashes — byte-exact and normalized-content."""

from __future__ import annotations

import hashlib
import re


def byte_hash(data: bytes) -> str:
    """SHA-256 of raw bytes — catches exact re-uploads."""
    return hashlib.sha256(data).hexdigest()


_WS = re.compile(r"\s+")


def content_hash(text: str) -> str:
    """Normalized-content hash. Collapses whitespace and lowercases, but keeps
    meaning-bearing characters (punctuation, digits). Catches the same content
    re-exported to a different file."""
    normalized = _WS.sub(" ", text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
