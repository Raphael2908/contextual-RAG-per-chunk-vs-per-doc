"""Shared benchmark logic — run one document through a set of combos.

Used by both the CLI (`benchmark/run.py`) and the API (`POST /benchmark`). Each
combo gets its own document row + Milvus collection so vectors stay isolated.
"""

from __future__ import annotations

from app import factory
from app.config import Settings
from app.ingest import IngestionService
from app.repository import (
    BenchmarkRun,
    DocumentRepository,
    PostgresBenchmarkRepository,
    PostgresChunkRepository,
)
from benchmark.matrix import Combo


def run_combo(
    *,
    settings: Settings,
    combo: Combo,
    data: bytes,
    name: str,
    file_type: str,
    doc_repo: DocumentRepository,
    chunk_repo: PostgresChunkRepository | None,
) -> BenchmarkRun:
    llm = factory.make_llm(settings, combo.llm_model)
    embedder = factory.make_embedder(settings, combo.embedding_model, dim=combo.embedding_dim)
    store = factory.make_vector_store(settings)
    blob = factory.make_blob(settings)

    service = IngestionService(
        embedder=embedder, store=store, blob=blob,
        doc_repo=doc_repo, chunk_repo=chunk_repo, llm=llm,
    )
    result = service.ingest(
        data,
        f"{name} [{combo.label}]",
        file_type,
        # The control (no LLM) is "none"; otherwise honour the combo's context_mode
        # so per_chunk vs per_document is a real cost/speed axis here too.
        context_mode="none" if combo.llm_model is None else combo.context_mode,
        enforce_dedup=False,        # same file across combos must not dedup
        persist_chunks=True,
    )
    return BenchmarkRun(
        document_id=result.document_id if result.status != "duplicate" else None,
        document_name=name,
        label=combo.label,
        llm_model=combo.llm_model,
        embedding_model=combo.embedding_model,
        embedding_dim=result.embedding_dim,
        with_context=result.with_context,
        cache_active=result.cache_active,
        chunk_count=result.chunk_count,
        stages=result.stage_dicts(),
        total_latency_ms=result.total_latency_ms,
        total_cost_usd=result.total_cost_usd,
        llm_input_tokens=result.llm_input_tokens,
        llm_output_tokens=result.llm_output_tokens,
        embed_tokens=result.embed_tokens,
    )


def run_matrix(
    *,
    settings: Settings,
    combos: list[Combo],
    data: bytes,
    name: str,
    file_type: str,
    doc_repo: DocumentRepository,
    chunk_repo: PostgresChunkRepository | None,
    bench_repo: PostgresBenchmarkRepository | None,
) -> list[BenchmarkRun]:
    runs: list[BenchmarkRun] = []
    for combo in combos:
        run = run_combo(
            settings=settings, combo=combo, data=data, name=name,
            file_type=file_type, doc_repo=doc_repo, chunk_repo=chunk_repo,
        )
        if bench_repo is not None:
            run.id = bench_repo.insert(run)
        runs.append(run)
    return runs
