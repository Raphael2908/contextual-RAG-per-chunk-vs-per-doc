"""Benchmark endpoints: run a sweep over the matrix, list results."""

from __future__ import annotations

from fastapi import APIRouter, File, Request, UploadFile

from app import extract
from benchmark.matrix import DEFAULT_MATRIX, FAKE_MATRIX
from benchmark.runner import run_matrix

router = APIRouter(tags=["benchmark"])


@router.post("/benchmark")
async def run_benchmark(request: Request, file: UploadFile = File(...)):
    settings = request.app.state.settings
    data = await file.read()
    name = file.filename or "upload.bin"
    file_type = extract.file_type_of(name)

    matrix = FAKE_MATRIX if settings.use_fakes else DEFAULT_MATRIX
    runs = run_matrix(
        settings=settings,
        combos=matrix,
        data=data,
        name=name,
        file_type=file_type,
        doc_repo=request.app.state.doc_repo,
        chunk_repo=request.app.state.chunk_repo,
        bench_repo=request.app.state.bench_repo,
    )
    return {
        "document": name,
        "runs": [
            {
                "id": r.id,
                "label": r.label,
                "llm_model": r.llm_model,
                "embedding_model": r.embedding_model,
                "embedding_dim": r.embedding_dim,
                "with_context": r.with_context,
                "cache_active": r.cache_active,
                "chunk_count": r.chunk_count,
                "total_latency_ms": round(r.total_latency_ms, 1),
                "total_cost_usd": round(r.total_cost_usd, 6),
                "llm_input_tokens": r.llm_input_tokens,
                "llm_output_tokens": r.llm_output_tokens,
                "embed_tokens": r.embed_tokens,
            }
            for r in runs
        ],
    }


@router.get("/benchmark/results")
def benchmark_results(request: Request):
    return request.app.state.bench_repo.list()
