"""Retrieval-quality eval endpoints: run the gold-set sweep, list results.

`POST /eval/run` scores every matrix combo against the server-side gold set
(`benchmark/eval_set.jsonl` over `samples/`) and persists `eval_runs`; the
frontend's Quality tab joins these with `benchmark_runs` cost/latency.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app import factory
from benchmark.eval import load_eval_set, run_eval
from benchmark.matrix import DEFAULT_MATRIX, FAKE_MATRIX

router = APIRouter(tags=["eval"])


@router.post("/eval/run")
def run(request: Request):
    settings = request.app.state.settings
    eval_set = load_eval_set()
    combos = FAKE_MATRIX if settings.use_fakes else DEFAULT_MATRIX
    runs = run_eval(
        settings=settings,
        combos=combos,
        eval_set=eval_set,
        samples_dir=request.app.state.samples_dir,
        doc_repo=request.app.state.doc_repo,
        chunk_repo=request.app.state.chunk_repo,
        eval_repo=request.app.state.eval_repo,
        reranker=factory.make_reranker(settings),
    )
    return {
        "eval_set": "samples",
        "k": settings.eval_k,
        "query_count": len(eval_set),
        "runs": [
            {
                "id": r.id, "label": r.label, "llm_model": r.llm_model,
                "embedding_model": r.embedding_model, "with_context": r.with_context,
                "reranked": r.reranked, "k": r.k, "query_count": r.query_count,
                "recall_at_k": round(r.recall_at_k, 4), "mrr": round(r.mrr, 4),
                "ndcg_at_k": round(r.ndcg_at_k, 4),
            }
            for r in runs
        ],
    }


@router.get("/eval/results")
def results(request: Request):
    return request.app.state.eval_repo.list()
