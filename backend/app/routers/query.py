"""Read path: POST /query — embed query → dense search → optional rerank.

The minimal slice of architecture.md §Retrieval pipeline (no org/matter/store
access filter yet — single-tenant slice). Returns ranked chunks with their text +
source location for citation. Hybrid (dense + sparse) is the documented next step.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.retrieve import RetrievalService
from app.security import Identity, require_identity
from app.vectorstore import collection_for

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    query: str
    k: int | None = None
    collection: str | None = None  # explicit collection name wins over context_mode
    # Which contextual-retrieval system to query (none | per_chunk | per_document) —
    # each lives in its own collection. Defaults to the deployed RAG_CONTEXT_MODE.
    context_mode: str | None = None
    mode: str | None = None        # "dense" | "hybrid"; defaults to RAG_RETRIEVAL_MODE
    rerank: bool = False


@router.post("/query")
def query(request: Request, body: QueryRequest, identity: Identity = Depends(require_identity)):
    settings = request.app.state.settings
    retrieval: RetrievalService = request.app.state.retrieval
    chunk_repo = request.app.state.chunk_repo

    mode = body.mode or settings.retrieval_mode
    hybrid = mode == "hybrid"
    collection = body.collection or collection_for(
        settings.embedding_model, settings.dense_dim,
        context_mode=body.context_mode or settings.context_mode, hybrid=hybrid,
    )
    k = body.k or settings.retrieval_top_k

    # Resolve candidate text from Postgres for reranking + citations.
    def resolver(ids: list[str]) -> list[str | None]:
        rows = chunk_repo.get_by_ids(ids)
        return [rows[i].context_text if i in rows else None for i in ids]

    result = retrieval.search(
        collection, body.query, k,
        rerank=body.rerank, hybrid=hybrid, candidate_k=settings.rerank_candidates,
        text_resolver=resolver,
    )
    hits = result.hits
    rows = chunk_repo.get_by_ids([h.chunk_id for h in hits])
    return {
        "query": body.query,
        "collection": collection,
        "mode": mode,
        "reranked": body.rerank and retrieval.reranker is not None
        and retrieval.reranker.name != "identity",
        "usage": {
            "query_embed_tokens": result.query_embed_tokens,
            "query_embed_cost_usd": round(result.query_embed_cost_usd, 8),
        },
        "results": [
            {
                "chunk_id": h.chunk_id,
                "document_id": h.document_id,
                "score": round(h.score, 6),
                "rank": h.rank,
                "source_location": rows[h.chunk_id].source_location if h.chunk_id in rows else None,
                "text": rows[h.chunk_id].text if h.chunk_id in rows else None,
            }
            for h in hits
        ],
    }
