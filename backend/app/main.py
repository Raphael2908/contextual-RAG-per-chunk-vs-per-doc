"""FastAPI composition root for the minimal ingest + benchmark slice.

Wires Postgres + Milvus + blob volume + the default LLM/embedder, runs DB
migrations on startup, and mounts the documents + benchmark routers. No
Celery/Redis/Caddy/Supabase/auth — see the plan.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import factory
from app.config import get_settings
from app.db import make_pool, run_migrations
from app.ingest import IngestionService
from app.repository import (
    PostgresBatchRepository,
    PostgresBenchmarkRepository,
    PostgresChunkRepository,
    PostgresDocumentRepository,
    PostgresEvalRepository,
)
from app.retrieve import RetrievalService
from app.routers import batches as batches_router
from app.routers import benchmark as benchmark_router
from app.routers import documents as documents_router
from app.routers import e2e as e2e_router
from app.routers import eval as eval_router
from app.routers import query as query_router
from app.vectorstore import collection_for
from app.worker import IngestWorkerPool


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.export_provider_keys()

    pool = make_pool(settings.database_url)
    run_migrations(pool)

    doc_repo = PostgresDocumentRepository(pool)
    chunk_repo = PostgresChunkRepository(pool)
    bench_repo = PostgresBenchmarkRepository(pool)
    eval_repo = PostgresEvalRepository(pool)
    batch_repo = PostgresBatchRepository(pool)

    app.state.settings = settings
    app.state.pool = pool
    app.state.doc_repo = doc_repo
    app.state.chunk_repo = chunk_repo
    app.state.bench_repo = bench_repo
    app.state.eval_repo = eval_repo
    app.state.batch_repo = batch_repo
    app.state.samples_dir = Path(__file__).resolve().parents[2] / "samples"
    # Where scripts/e2e_bulk.py writes its JSON reports (resolved off the repo
    # root, not cwd, so the /e2e/results endpoint finds them regardless of how
    # the API was launched).
    app.state.e2e_results_dir = Path(__file__).resolve().parents[2] / "results"

    # Default models, shared by the upload endpoint and the read path so they
    # embed documents and queries with the same embedder.
    default_embedder = factory.make_embedder(settings, settings.embedding_model)
    default_store = factory.make_vector_store(settings)
    # One blob store shared by the single-upload ingest, the bulk upload endpoint
    # (writes the original), and the worker pool (reads it back). Sharing the
    # instance matters for the in-memory fake, which is process-local.
    default_blob = factory.make_blob(settings)
    app.state.blob = default_blob

    hybrid = settings.retrieval_mode == "hybrid"
    app.state.ingestion = IngestionService(
        embedder=default_embedder,
        store=default_store,
        blob=default_blob,
        doc_repo=doc_repo,
        chunk_repo=chunk_repo,
        llm=factory.make_llm(settings, settings.context_model),
        hybrid=hybrid,
    )
    app.state.retrieval = RetrievalService(
        embedder=default_embedder,
        store=default_store,
        reranker=factory.make_reranker(settings),
        hybrid=hybrid,
        rrf_k=settings.rrf_k,
    )

    # Pre-warm the default collection so bulk worker threads always hit the
    # ensure_collection fast path (avoids a first-batch create race).
    default_collection = collection_for(
        default_embedder.model, default_embedder.dim or settings.dense_dim,
        context_mode=settings.context_mode, hybrid=hybrid,
    )
    default_store.ensure_collection(
        default_collection, default_embedder.dim or settings.dense_dim, hybrid=hybrid
    )

    # Bulk-upload ingest worker pool (decoupled upload → async ingest).
    worker_pool = IngestWorkerPool(
        ingestion=app.state.ingestion,
        blob=default_blob,
        doc_repo=doc_repo,
        concurrency=settings.bulk_ingest_concurrency,
        queue_max=settings.bulk_ingest_queue_max,
    )
    await worker_pool.start()
    app.state.worker_pool = worker_pool

    try:
        yield
    finally:
        await worker_pool.drain_and_stop()
        pool.close()


app = FastAPI(title="Company Brain — ingest + benchmark slice", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local-only slice
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router.router)
app.include_router(batches_router.router)
app.include_router(benchmark_router.router)
app.include_router(query_router.router)
app.include_router(eval_router.router)
app.include_router(e2e_router.router)


@app.get("/health")
def health():
    return {"status": "ok"}
