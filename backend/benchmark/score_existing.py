"""Score retrieval accuracy of an ALREADY-INGESTED context-mode against a gold set.

The ingest-once companion to benchmark/eval.py. Because each context mode now has its
OWN Milvus collection (collection_for(..., context_mode=...)), a bulk run already left a
clean per-system store. This scorer queries that store directly — no re-ingest — and
computes recall@k / MRR / nDCG@k, persisting one `eval_runs` row so the result shows up in
the Quality tab next to the others.

Relevance is resolved from Postgres: the batch's documents (filename -> doc ids) and their
chunks (a chunk is relevant if its raw text contains a gold marker), reusing eval.py's
EvalQuery / metric helpers so labelling matches the rest of the harness.

    APP_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/company_brain \
    RAG_MILVUS_URI=http://localhost:19530 APP_USE_FAKES=0 PYTHONPATH=backend \
    python -m benchmark.score_existing --batch-id <id> --context-mode per_document \
        --eval-set backend/benchmark/eval_set_8k_eval.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app import factory
from app.config import get_settings
from app.db import make_pool, run_migrations
from app.repository import (
    EvalRun,
    PostgresBatchRepository,
    PostgresChunkRepository,
    PostgresEvalRepository,
)
from app.retrieve import RetrievalService
from app.vectorstore import collection_for
from benchmark.eval import EvalQuery, load_eval_set, mrr_at_k, ndcg_at_k, recall_at_k


def _name_to_doc_ids(batch_repo: PostgresBatchRepository, batch_id: str) -> dict[str, list[str]]:
    summary = batch_repo.summary(batch_id, include_docs=True)
    if summary is None:
        raise SystemExit(f"batch {batch_id} not found")
    out: dict[str, list[str]] = {}
    for d in summary.get("docs", []):
        if d["status"] == "ready":
            out.setdefault(d["name"], []).append(d["id"])
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-id", required=True, dest="batch_id")
    ap.add_argument("--context-mode", required=True, dest="context_mode",
                    choices=["none", "per_chunk", "per_document"])
    ap.add_argument("--eval-set", type=Path, default=Path("backend/benchmark/eval_set_8k_eval.jsonl"),
                    dest="eval_set")
    ap.add_argument("--eval-set-name", default="8k_eval", dest="eval_set_name")
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args(argv)

    settings = get_settings()
    settings.export_provider_keys()
    k = args.k or settings.eval_k
    mode = args.context_mode

    pool = make_pool(settings.database_url)
    run_migrations(pool)
    batch_repo = PostgresBatchRepository(pool)
    chunk_repo = PostgresChunkRepository(pool)
    eval_repo = PostgresEvalRepository(pool)

    embedder = factory.make_embedder(settings, settings.embedding_model)
    store = factory.make_vector_store(settings)
    reranker = factory.make_reranker(settings)
    retrieval = RetrievalService(embedder=embedder, store=store, reranker=reranker,
                                 hybrid=False, rrf_k=settings.rrf_k)
    collection = collection_for(embedder.model, settings.dense_dim, context_mode=mode)

    gold: list[EvalQuery] = load_eval_set(args.eval_set)
    if not gold:
        print(f"eval set {args.eval_set} is empty", file=sys.stderr)
        return 2

    name_map = _name_to_doc_ids(batch_repo, args.batch_id)

    # Cache each doc's chunks once (chunk_id -> raw text) to resolve relevance.
    chunks_by_doc: dict[str, dict[str, str]] = {}
    for doc_ids in name_map.values():
        for did in doc_ids:
            if did not in chunks_by_doc:
                chunks_by_doc[did] = {r.chunk_id: r.text for r in chunk_repo.list_for_document(did)}

    recalls, mrrs, ndcgs, details = [], [], [], []
    for q in gold:
        doc_ids = name_map.get(q.doc, [])
        relevant = {
            cid for did in doc_ids for cid, txt in chunks_by_doc.get(did, {}).items()
            if q.is_relevant(txt)
        }
        if not relevant:
            details.append({"query": q.query, "doc": q.doc, "note": "NO GOLD CHUNK MATCHED"})
            continue
        ranked = [h.chunk_id for h in retrieval.search(collection, q.query, k).hits]
        r, m, n = recall_at_k(ranked, relevant, k), mrr_at_k(ranked, relevant, k), ndcg_at_k(ranked, relevant, k)
        recalls.append(r); mrrs.append(m); ndcgs.append(n)
        details.append({"query": q.query, "doc": q.doc, "note": q.note,
                        "recall": round(r, 4), "mrr": round(m, 4), "ndcg": round(n, 4),
                        "relevant": len(relevant), "top": ranked[:k]})

    n_scored = len(recalls)
    run = EvalRun(
        eval_set=args.eval_set_name, label=f"8k {mode}", retrieval_mode="dense",
        llm_model=(settings.context_model if mode != "none" else None),
        embedding_model=embedder.model, embedding_dim=settings.dense_dim,
        with_context=(mode != "none"), context_mode=mode, reranked=False, rerank_model=None,
        k=k, query_count=n_scored,
        recall_at_k=(sum(recalls) / n_scored) if n_scored else 0.0,
        mrr=(sum(mrrs) / n_scored) if n_scored else 0.0,
        ndcg_at_k=(sum(ndcgs) / n_scored) if n_scored else 0.0,
        details=details,
    )
    run.id = eval_repo.insert(run)

    misses = sum(1 for d in details if d.get("note") == "NO GOLD CHUNK MATCHED")
    print(f"collection: {collection}")
    print(f"context_mode={mode}  queries={n_scored}  (no-gold-match: {misses})")
    print(f"  recall@{k}={run.recall_at_k:.3f}  MRR={run.mrr:.3f}  nDCG@{k}={run.ndcg_at_k:.3f}")
    pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
