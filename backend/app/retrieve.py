"""RetrievalService — the read path: embed query → search → (rerank).

architecture.md §Retrieval pipeline, carved to what the slice supports today:
dense search (step 3, sparse/hybrid is the documented next step) → cross-encoder
rerank (step 4). The SAME service backs the `/query` API and the retrieval-quality
eval, so they exercise one path.

The query is embedded with the SAME embedder as the documents it searches, with
`input_type="query"` — Voyage (and Cohere) produce distinct query vs document
vectors, and mixing them quietly tanks recall.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.embeddings import Embedder
from app.reranker import Reranker
from app.vectorstore import Hit, VectorStore

# How many dense candidates to pull before reranking down to k. The reranker only
# improves precision within the candidate set, so it must be wider than k.
DEFAULT_RERANK_CANDIDATES = 30


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str | None
    score: float
    rank: int
    text: str | None = None


@dataclass
class SearchResult:
    """Ranked hits plus the AI cost of the read path. The query embedding is the
    only token spend on retrieval (dense, no rerank), so surface it for metrics —
    LiteLLM gives us the real tokens/cost on every query embed."""
    hits: list[RetrievedChunk]
    query_embed_tokens: int = 0
    query_embed_cost_usd: float = 0.0


class RetrievalService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker | None = None,
        hybrid: bool = False,
        rrf_k: int = 60,
    ):
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.hybrid = hybrid
        self.rrf_k = rrf_k

    def search(
        self,
        collection: str,
        query: str,
        k: int,
        *,
        rerank: bool = False,
        hybrid: bool | None = None,
        filter: str = "",
        candidate_k: int | None = None,
        text_resolver: Callable[[list[str]], list[str | None]] | None = None,
    ) -> SearchResult:
        qres = self.embedder.embed([query], input_type="query")
        qvec = qres.vectors[0]
        use_hybrid = self.hybrid if hybrid is None else hybrid

        do_rerank = rerank and self.reranker is not None and self.reranker.name != "identity"
        depth = (candidate_k or DEFAULT_RERANK_CANDIDATES) if do_rerank else k
        hits = self.store.search(
            collection, qvec, depth, filter=filter,
            query_text=query if use_hybrid else None,
            hybrid=use_hybrid, rrf_k=self.rrf_k,
        )

        # Resolve chunk text for the candidates (needed to rerank, useful for citations).
        texts: list[str | None] = [None] * len(hits)
        if text_resolver is not None and hits:
            texts = text_resolver([h.id for h in hits])

        if do_rerank and hits:
            docs = [t or "" for t in texts]
            ranking = self.reranker.rerank(query, docs, top_n=k).ranking
            ordered = [(hits[i], texts[i], score) for i, score in ranking if i < len(hits)]
        else:
            ordered = [(h, texts[i], h.score) for i, h in enumerate(hits[:k])]

        hits_out = [
            RetrievedChunk(
                chunk_id=hit.id,
                document_id=hit.document_id,
                score=score,
                rank=rank,
                text=text,
            )
            for rank, (hit, text, score) in enumerate(ordered)
        ]
        return SearchResult(
            hits=hits_out,
            query_embed_tokens=qres.input_tokens,
            query_embed_cost_usd=qres.cost_usd,
        )
