"""The reranker, behind a Protocol with a real + fake impl.

architecture.md §Retrieval pipeline step 4: a cross-encoder reranks the merged
candidate set for precision before the LLM. Here the real impl routes through
LiteLLM's rerank API (Voyage `rerank-2`, Cohere `rerank-3.5`, etc.) so a reranker
is just a config string, mirroring the LLM/embedder Protocols. The fake is an
identity pass-through so the retrieval + eval harness runs with no keys/network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.litellm_util import cost_of, normalize_model


@dataclass
class RerankResult:
    # (item_index, score) sorted best-first, indexing into the input documents.
    ranking: list[tuple[int, float]]
    cost_usd: float = 0.0
    extra: dict = field(default_factory=dict)


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, documents: list[str], *, top_n: int) -> RerankResult: ...


class IdentityReranker:
    """No-op reranker — keeps the candidate order (descending dense score) and a
    monotonically decreasing pseudo-score. Lets the pipeline run identically
    whether or not a real reranker is wired."""

    name = "identity"

    def rerank(self, query: str, documents: list[str], *, top_n: int) -> RerankResult:
        ranking = [(i, 1.0 - i * 1e-6) for i in range(len(documents))][:top_n]
        return RerankResult(ranking=ranking)


class LiteLLMReranker:
    """LiteLLM-backed cross-encoder reranker. `voyage/rerank-2`,
    `cohere/rerank-3.5`, `together_ai/...`, etc. — any model LiteLLM's rerank
    API resolves. The architecture's default is a `bge-reranker-v2-m3`-class
    cross-encoder; pick the hosted equivalent per provider via `RAG_RERANK_MODEL`."""

    def __init__(self, model: str):
        self.model = normalize_model(model)
        self.name = self.model

    def rerank(self, query: str, documents: list[str], *, top_n: int) -> RerankResult:
        import litellm

        resp = litellm.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)) or 1,
        )
        results = getattr(resp, "results", None)
        if results is None and isinstance(resp, dict):
            results = resp.get("results", [])
        ranking: list[tuple[int, float]] = []
        for r in results or []:
            idx = r.get("index") if isinstance(r, dict) else getattr(r, "index", None)
            score = (
                r.get("relevance_score")
                if isinstance(r, dict)
                else getattr(r, "relevance_score", 0.0)
            )
            if idx is not None:
                ranking.append((int(idx), float(score or 0.0)))
        ranking.sort(key=lambda t: t[1], reverse=True)
        return RerankResult(ranking=ranking[:top_n], cost_usd=cost_of(resp))
