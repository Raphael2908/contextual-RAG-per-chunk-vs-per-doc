"""The embedder, behind a Protocol with a real + fake impl.

Real impl routes through LiteLLM, which (unlike OpenRouter) covers Voyage
(`voyage/voyage-law-2`, `voyage/voyage-3-large`) as well as OpenAI, Cohere,
Gemini, and `openrouter/...` models — all by a single config string.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.litellm_util import cost_of, normalize_model, usage_of


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    input_tokens: int = 0
    cost_usd: float = 0.0
    dim: int = 0
    extra: dict = field(default_factory=dict)


@runtime_checkable
class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str], *, input_type: str | None = None) -> EmbedResult: ...


class LiteLLMEmbedder:
    def __init__(self, model: str, *, dim: int | None = None):
        self.model = normalize_model(model)
        self.dim = dim or 0

    def embed(self, texts: list[str], *, input_type: str | None = None) -> EmbedResult:
        import litellm

        kwargs: dict = {"model": self.model, "input": texts}
        if input_type:  # Voyage/Cohere use this to optimise document vs query vectors
            kwargs["input_type"] = input_type
        resp = litellm.embedding(**kwargs)
        # response.data is a list of {"embedding": [...], "index": i}
        rows = sorted(resp.data, key=lambda r: r.get("index", 0))
        vectors = [list(r["embedding"]) for r in rows]
        dim = len(vectors[0]) if vectors else self.dim
        self.dim = dim
        u = usage_of(resp)
        return EmbedResult(
            vectors=vectors,
            input_tokens=u.get("input_tokens", 0) or u.get("total_tokens", 0),
            cost_usd=cost_of(resp),
            dim=dim,
        )


_TOKEN = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Deterministic bag-of-words embedder — feature-hashes tokens into a fixed
    vector, then L2-normalizes. No network/cost, fully reproducible.

    Unlike a pure positional hash, this carries a real *lexical* signal: a query
    and a chunk that share words land in the same dimensions, so cosine is > 0
    and ranking is meaningful. That makes the retrieval-quality eval a genuine
    self-test on fakes (it can tell context vs no-context apart) rather than
    noise, while needing no keys. It is a deliberately crude stand-in for a
    semantic model, not a substitute for one — real recall needs a real embedder.
    """

    def __init__(self, model: str = "fake/hash", *, dim: int = 1024):
        self.model = model
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        out = [0.0] * self.dim
        for tok in _TOKEN.findall(text.lower()):
            h = int.from_bytes(hashlib.sha256(tok.encode("utf-8")).digest()[:8], "little")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0  # signed feature hashing
            out[idx] += sign
        norm = sum(v * v for v in out) ** 0.5 or 1.0
        return [v / norm for v in out]

    def embed(self, texts: list[str], *, input_type: str | None = None) -> EmbedResult:
        vectors = [self._vec(t) for t in texts]
        return EmbedResult(
            vectors=vectors,
            input_tokens=sum(len(t.split()) for t in texts),
            cost_usd=0.0,
            dim=self.dim,
        )
