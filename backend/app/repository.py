"""Postgres repositories for documents, chunks, and benchmark runs.

Each has an in-memory double so the pipeline can run on dependency-free fakes
(architecture.md convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # avoid importing psycopg on the dependency-free fake path
    from psycopg_pool import ConnectionPool


@dataclass
class DocumentRow:
    id: str
    name: str
    file_type: str
    storage_path: str
    file_hash: str
    content_hash: str | None = None
    summary: str | None = None
    effective_date: date | None = None
    size: int = 0
    status: str = "queued"
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None    # terminal-transition time (migration 005); feeds batch elapsed
    batch_id: str | None = None           # set for bulk-uploaded docs (migration 003)
    chunk_count: int = 0                  # persisted on completion (batch totals)
    total_cost_usd: float = 0.0           # persisted on completion (batch totals)
    llm_input_tokens: int = 0             # persisted on completion (migration 004)
    llm_output_tokens: int = 0            # persisted on completion (migration 004)
    embed_tokens: int = 0                 # persisted on completion (migration 004)


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
class DocumentRepository(Protocol):
    def create(self, row: DocumentRow) -> str: ...
    def set_status(self, doc_id: str, status: str, error: str | None = None) -> None: ...
    def set_result(
        self, doc_id: str, status: str, *, chunk_count: int, total_cost_usd: float,
        llm_input_tokens: int = 0, llm_output_tokens: int = 0, embed_tokens: int = 0,
        error: str | None = None,
    ) -> None: ...
    def update_profile(
        self, doc_id: str, *, summary: str, effective_date: date | None, content_hash: str
    ) -> None: ...
    def get(self, doc_id: str) -> DocumentRow | None: ...
    def list(self, limit: int = 100) -> list[DocumentRow]: ...
    def find_by_file_hash(self, file_hash: str) -> DocumentRow | None: ...


class PostgresDocumentRepository:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def create(self, row: DocumentRow) -> str:
        with self.pool.connection() as conn:
            res = conn.execute(
                """INSERT INTO documents
                   (name, file_type, storage_path, file_hash, content_hash,
                    summary, effective_date, size, status, batch_id)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (
                    row.name, row.file_type, row.storage_path, row.file_hash,
                    row.content_hash, row.summary, row.effective_date, row.size,
                    row.status, row.batch_id,
                ),
            ).fetchone()
            conn.commit()
            return str(res[0])

    def set_status(self, doc_id: str, status: str, error: str | None = None) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE documents SET status=%s, error=%s, updated_at=now() WHERE id=%s",
                (status, error, doc_id),
            )
            conn.commit()

    def set_result(
        self, doc_id: str, status: str, *, chunk_count: int, total_cost_usd: float,
        llm_input_tokens: int = 0, llm_output_tokens: int = 0, embed_tokens: int = 0,
        error: str | None = None,
    ) -> None:
        """Final write after ingest: status + the metrics that feed batch totals."""
        with self.pool.connection() as conn:
            conn.execute(
                """UPDATE documents
                   SET status=%s, chunk_count=%s, total_cost_usd=%s,
                       llm_input_tokens=%s, llm_output_tokens=%s, embed_tokens=%s,
                       error=%s, updated_at=now()
                   WHERE id=%s""",
                (status, chunk_count, total_cost_usd, llm_input_tokens,
                 llm_output_tokens, embed_tokens, error, doc_id),
            )
            conn.commit()

    def update_profile(
        self, doc_id: str, *, summary: str, effective_date: date | None, content_hash: str
    ) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE documents SET summary=%s, effective_date=%s, content_hash=%s WHERE id=%s",
                (summary, effective_date, content_hash, doc_id),
            )
            conn.commit()

    def get(self, doc_id: str) -> DocumentRow | None:
        with self.pool.connection() as conn:
            r = conn.execute(_DOC_COLS + " WHERE id=%s", (doc_id,)).fetchone()
        return _to_doc(r) if r else None

    def list(self, limit: int = 100) -> list[DocumentRow]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                _DOC_COLS + " ORDER BY created_at DESC LIMIT %s", (limit,)
            ).fetchall()
        return [_to_doc(r) for r in rows]

    def find_by_file_hash(self, file_hash: str) -> DocumentRow | None:
        with self.pool.connection() as conn:
            r = conn.execute(
                _DOC_COLS + " WHERE file_hash=%s ORDER BY created_at LIMIT 1",
                (file_hash,),
            ).fetchone()
        return _to_doc(r) if r else None


_DOC_COLS = (
    "SELECT id, name, file_type, storage_path, file_hash, content_hash, summary, "
    "effective_date, size, status, error, created_at, batch_id, chunk_count, "
    "total_cost_usd, llm_input_tokens, llm_output_tokens, embed_tokens FROM documents"
)


def _to_doc(r: tuple) -> DocumentRow:
    return DocumentRow(
        id=str(r[0]), name=r[1], file_type=r[2], storage_path=r[3], file_hash=r[4],
        content_hash=r[5], summary=r[6], effective_date=r[7], size=r[8],
        status=r[9], error=r[10], created_at=r[11],
        batch_id=str(r[12]) if r[12] is not None else None,
        chunk_count=r[13] or 0, total_cost_usd=float(r[14] or 0),
        llm_input_tokens=r[15] or 0, llm_output_tokens=r[16] or 0,
        embed_tokens=r[17] or 0,
    )


# --------------------------------------------------------------------------- #
# Batches (bulk upload)
# --------------------------------------------------------------------------- #
class BatchRepository(Protocol):
    def create(self, label: str | None, with_context: bool, context_mode: str = "per_chunk") -> str: ...
    def get(self, batch_id: str) -> dict[str, Any] | None: ...
    def bump_total(self, batch_id: str, n: int = 1) -> None: ...
    def summary(self, batch_id: str, *, include_docs: bool = True) -> dict[str, Any] | None: ...


class PostgresBatchRepository:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def create(self, label: str | None, with_context: bool, context_mode: str = "per_chunk") -> str:
        with self.pool.connection() as conn:
            res = conn.execute(
                "INSERT INTO batches (label, with_context, context_mode) "
                "VALUES (%s,%s,%s) RETURNING id",
                (label, with_context, context_mode),
            ).fetchone()
            conn.commit()
            return str(res[0])

    def get(self, batch_id: str) -> dict[str, Any] | None:
        with self.pool.connection() as conn:
            r = conn.execute(
                "SELECT id, label, with_context, total, created_at, context_mode "
                "FROM batches WHERE id=%s",
                (batch_id,),
            ).fetchone()
        if not r:
            return None
        return {
            "id": str(r[0]), "label": r[1], "with_context": r[2], "total": r[3],
            "created_at": r[4].isoformat() if r[4] else None, "context_mode": r[5],
        }

    def bump_total(self, batch_id: str, n: int = 1) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE batches SET total = total + %s WHERE id=%s", (n, batch_id)
            )
            conn.commit()

    def summary(self, batch_id: str, *, include_docs: bool = True) -> dict[str, Any] | None:
        batch = self.get(batch_id)
        if batch is None:
            return None
        with self.pool.connection() as conn:
            grouped = conn.execute(
                """SELECT status, count(*), coalesce(sum(chunk_count),0),
                          coalesce(sum(total_cost_usd),0),
                          coalesce(sum(llm_input_tokens),0),
                          coalesce(sum(llm_output_tokens),0),
                          coalesce(sum(embed_tokens),0),
                          max(coalesce(updated_at, created_at))
                   FROM documents WHERE batch_id=%s GROUP BY status""",
                (batch_id,),
            ).fetchall()
            docs = []
            if include_docs:
                rows = conn.execute(
                    """SELECT id, name, file_type, status, chunk_count, total_cost_usd,
                              llm_input_tokens, llm_output_tokens, embed_tokens, error
                       FROM documents WHERE batch_id=%s ORDER BY created_at""",
                    (batch_id,),
                ).fetchall()
                docs = [
                    {
                        "id": str(d[0]), "name": d[1], "file_type": d[2], "status": d[3],
                        "chunk_count": d[4] or 0, "total_cost_usd": float(d[5] or 0),
                        "llm_input_tokens": d[6] or 0, "llm_output_tokens": d[7] or 0,
                        "embed_tokens": d[8] or 0, "error": d[9],
                    }
                    for d in rows
                ]
        return _batch_summary(batch, grouped, docs, include_docs)


def _batch_summary(batch, grouped, docs, include_docs) -> dict[str, Any]:
    counts: dict[str, int] = {}
    chunk_total = 0
    cost_total = 0.0
    llm_in_total = 0
    llm_out_total = 0
    embed_total = 0
    last_at: datetime | None = None  # latest doc-status transition across the batch
    for row in grouped:
        status, n, chunks, cost, llm_in, llm_out, embed = row[:7]
        row_last = row[7] if len(row) > 7 else None
        counts[status] = int(n)
        chunk_total += int(chunks)
        cost_total += float(cost)
        llm_in_total += int(llm_in)
        llm_out_total += int(llm_out)
        embed_total += int(embed)
        if row_last is not None and (last_at is None or row_last > last_at):
            last_at = row_last
    out = {
        "batch_id": batch["id"], "label": batch["label"],
        "with_context": batch["with_context"],
        "context_mode": batch.get("context_mode", "per_chunk"),
        "total": batch["total"],
        "created_at": batch.get("created_at"),
        "elapsed_seconds": _elapsed_seconds(batch.get("created_at"), counts, batch["total"], last_at),
        "counts": counts,
        "totals": {
            "chunk_count": chunk_total,
            "total_cost_usd": round(cost_total, 6),
            "llm_input_tokens": llm_in_total,
            "llm_output_tokens": llm_out_total,
            "embed_tokens": embed_total,
        },
    }
    if include_docs:
        out["docs"] = docs
    return out


def _elapsed_seconds(created_at, counts, total, last_at) -> float | None:
    """Batch wall-time: (last status transition if finished, else now) − batch start.

    Persisted per-doc `updated_at` is the source of truth (max gathered as `last_at`).
    A still-running batch keeps growing (now − start); a finished one is pinned to its
    last transition, so it's stable on every re-fetch.
    """
    start = created_at
    if isinstance(start, str):
        start = datetime.fromisoformat(start)
    if start is None:
        return None
    terminal = (counts.get("ready", 0) + counts.get("failed", 0) + counts.get("duplicate", 0))
    finished = total > 0 and terminal >= total
    end = last_at if (finished and last_at is not None) else datetime.now(timezone.utc)
    return round(max(0.0, (end - start).total_seconds()), 1)


# --------------------------------------------------------------------------- #
# Chunks
# --------------------------------------------------------------------------- #
@dataclass
class ChunkRow:
    document_id: str
    chunk_index: int
    source_location: str
    text: str
    context_text: str
    token_count: int = 0

    @property
    def chunk_id(self) -> str:
        """The id this chunk's vector is stored under (see ingest.py)."""
        return f"{self.document_id}:{self.chunk_index}"


def _to_chunk(r: tuple) -> ChunkRow:
    return ChunkRow(
        document_id=str(r[0]), chunk_index=r[1], source_location=r[2],
        text=r[3], context_text=r[4], token_count=r[5],
    )


def split_chunk_id(chunk_id: str) -> tuple[str, int]:
    """Vector-store chunk ids are `{document_id}:{chunk_index}` (see ingest.py)."""
    doc_id, _, idx = chunk_id.rpartition(":")
    return doc_id, int(idx)


class PostgresChunkRepository:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def list_for_document(self, document_id: str) -> list[ChunkRow]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                """SELECT document_id, chunk_index, source_location, text,
                          context_text, token_count
                   FROM chunks WHERE document_id=%s ORDER BY chunk_index""",
                (document_id,),
            ).fetchall()
        return [_to_chunk(r) for r in rows]

    def get_by_ids(self, chunk_ids: list[str]) -> dict[str, ChunkRow]:
        """Resolve `{document_id}:{chunk_index}` ids to their rows (for rerank
        text + citations on the read path)."""
        out: dict[str, ChunkRow] = {}
        with self.pool.connection() as conn:
            for cid in chunk_ids:
                doc_id, idx = split_chunk_id(cid)
                r = conn.execute(
                    """SELECT document_id, chunk_index, source_location, text,
                              context_text, token_count
                       FROM chunks WHERE document_id=%s AND chunk_index=%s""",
                    (doc_id, idx),
                ).fetchone()
                if r:
                    out[cid] = _to_chunk(r)
        return out

    def insert_many(self, rows: list[ChunkRow]) -> list[str]:
        ids: list[str] = []
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                for row in rows:
                    cur.execute(
                        """INSERT INTO chunks
                           (document_id, chunk_index, source_location, text, context_text, token_count)
                           VALUES (%s,%s,%s,%s,%s,%s)
                           ON CONFLICT (document_id, chunk_index)
                           DO UPDATE SET context_text=EXCLUDED.context_text,
                                         text=EXCLUDED.text
                           RETURNING id""",
                        (row.document_id, row.chunk_index, row.source_location,
                         row.text, row.context_text, row.token_count),
                    )
                    ids.append(str(cur.fetchone()[0]))
            conn.commit()
        return ids


# --------------------------------------------------------------------------- #
# Benchmark runs
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkRun:
    document_id: str | None
    document_name: str
    label: str
    llm_model: str | None
    embedding_model: str
    embedding_dim: int
    with_context: bool
    cache_active: bool
    chunk_count: int
    stages: list[dict] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    embed_tokens: int = 0
    id: str | None = None
    created_at: datetime | None = None


class PostgresBenchmarkRepository:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def insert(self, run: BenchmarkRun) -> str:
        from psycopg.types.json import Json

        with self.pool.connection() as conn:
            res = conn.execute(
                """INSERT INTO benchmark_runs
                   (document_id, document_name, label, llm_model, embedding_model,
                    embedding_dim, with_context, cache_active, chunk_count, stages,
                    total_latency_ms, total_cost_usd, llm_input_tokens,
                    llm_output_tokens, embed_tokens)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (
                    run.document_id, run.document_name, run.label, run.llm_model,
                    run.embedding_model, run.embedding_dim, run.with_context,
                    run.cache_active, run.chunk_count, Json(run.stages),
                    run.total_latency_ms, run.total_cost_usd, run.llm_input_tokens,
                    run.llm_output_tokens, run.embed_tokens,
                ),
            ).fetchone()
            conn.commit()
            return str(res[0])

    def list(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                """SELECT id, document_name, label, llm_model, embedding_model,
                          embedding_dim, with_context, cache_active, chunk_count,
                          stages, total_latency_ms, total_cost_usd, llm_input_tokens,
                          llm_output_tokens, embed_tokens, created_at
                   FROM benchmark_runs ORDER BY created_at DESC LIMIT %s""",
                (limit,),
            ).fetchall()
        keys = [
            "id", "document_name", "label", "llm_model", "embedding_model",
            "embedding_dim", "with_context", "cache_active", "chunk_count", "stages",
            "total_latency_ms", "total_cost_usd", "llm_input_tokens",
            "llm_output_tokens", "embed_tokens", "created_at",
        ]
        out = []
        for r in rows:
            d = dict(zip(keys, r))
            d["id"] = str(d["id"])
            d["total_cost_usd"] = float(d["total_cost_usd"])
            d["created_at"] = d["created_at"].isoformat() if d["created_at"] else None
            out.append(d)
        return out


# --------------------------------------------------------------------------- #
# In-memory doubles (dependency-free; used by fakes / tests / no-DB smoke runs)
# --------------------------------------------------------------------------- #
class InMemoryDocumentRepository:
    def __init__(self) -> None:
        self._docs: dict[str, DocumentRow] = {}
        self._seq = 0

    def create(self, row: DocumentRow) -> str:
        self._seq += 1
        doc_id = f"doc-{self._seq}"
        stored = DocumentRow(
            id=doc_id, name=row.name, file_type=row.file_type,
            storage_path=row.storage_path, file_hash=row.file_hash,
            content_hash=row.content_hash, summary=row.summary,
            effective_date=row.effective_date, size=row.size, status=row.status,
            created_at=datetime.now(), batch_id=row.batch_id,
        )
        self._docs[doc_id] = stored
        return doc_id

    def set_status(self, doc_id: str, status: str, error: str | None = None) -> None:
        if doc_id in self._docs:
            self._docs[doc_id].status = status
            self._docs[doc_id].error = error
            self._docs[doc_id].updated_at = datetime.now(timezone.utc)

    def set_result(
        self, doc_id: str, status: str, *, chunk_count: int, total_cost_usd: float,
        llm_input_tokens: int = 0, llm_output_tokens: int = 0, embed_tokens: int = 0,
        error: str | None = None,
    ) -> None:
        d = self._docs.get(doc_id)
        if d is not None:
            d.status = status
            d.chunk_count = chunk_count
            d.total_cost_usd = total_cost_usd
            d.llm_input_tokens = llm_input_tokens
            d.llm_output_tokens = llm_output_tokens
            d.embed_tokens = embed_tokens
            d.error = error
            d.updated_at = datetime.now(timezone.utc)

    def update_profile(
        self, doc_id: str, *, summary: str, effective_date: date | None, content_hash: str
    ) -> None:
        if doc_id in self._docs:
            self._docs[doc_id].summary = summary
            self._docs[doc_id].effective_date = effective_date
            self._docs[doc_id].content_hash = content_hash

    def get(self, doc_id: str) -> DocumentRow | None:
        return self._docs.get(doc_id)

    def list(self, limit: int = 100) -> list[DocumentRow]:
        return list(reversed(list(self._docs.values())))[:limit]

    def find_by_file_hash(self, file_hash: str) -> DocumentRow | None:
        for d in self._docs.values():
            if d.file_hash == file_hash:
                return d
        return None


class InMemoryBatchRepository:
    """Reads document state straight from the in-memory document repo, so batch
    progress mirrors the Postgres GROUP BY aggregate without a separate store."""

    def __init__(self, doc_repo: InMemoryDocumentRepository) -> None:
        self.doc_repo = doc_repo
        self._batches: dict[str, dict[str, Any]] = {}
        self._seq = 0

    def create(self, label: str | None, with_context: bool, context_mode: str = "per_chunk") -> str:
        self._seq += 1
        batch_id = f"batch-{self._seq}"
        self._batches[batch_id] = {
            "id": batch_id, "label": label, "with_context": with_context,
            "context_mode": context_mode,
            "total": 0, "created_at": datetime.now(timezone.utc),
        }
        return batch_id

    def get(self, batch_id: str) -> dict[str, Any] | None:
        return self._batches.get(batch_id)

    def bump_total(self, batch_id: str, n: int = 1) -> None:
        b = self._batches.get(batch_id)
        if b is not None:
            b["total"] += n

    def summary(self, batch_id: str, *, include_docs: bool = True) -> dict[str, Any] | None:
        batch = self._batches.get(batch_id)
        if batch is None:
            return None
        members = [d for d in self.doc_repo._docs.values() if d.batch_id == batch_id]
        started = batch["created_at"]  # fallback finish time for docs not yet terminal
        grouped: dict[str, list] = {}
        for d in members:
            g = grouped.setdefault(d.status, [d.status, 0, 0, 0.0, 0, 0, 0, None])
            g[1] += 1
            g[2] += d.chunk_count
            g[3] += d.total_cost_usd
            g[4] += d.llm_input_tokens
            g[5] += d.llm_output_tokens
            g[6] += d.embed_tokens
            d_last = d.updated_at or started
            if d_last is not None and (g[7] is None or d_last > g[7]):
                g[7] = d_last
        docs = [
            {
                "id": d.id, "name": d.name, "file_type": d.file_type, "status": d.status,
                "chunk_count": d.chunk_count, "total_cost_usd": d.total_cost_usd,
                "llm_input_tokens": d.llm_input_tokens,
                "llm_output_tokens": d.llm_output_tokens,
                "embed_tokens": d.embed_tokens, "error": d.error,
            }
            for d in members
        ] if include_docs else []
        return _batch_summary(batch, list(grouped.values()), docs, include_docs)


class InMemoryChunkRepository:
    def __init__(self) -> None:
        self.rows: list[ChunkRow] = []

    def insert_many(self, rows: list[ChunkRow]) -> list[str]:
        start = len(self.rows)
        self.rows.extend(rows)
        return [f"chunk-{start + i}" for i in range(len(rows))]

    def list_for_document(self, document_id: str) -> list[ChunkRow]:
        return sorted(
            (r for r in self.rows if r.document_id == document_id),
            key=lambda r: r.chunk_index,
        )

    def get_by_ids(self, chunk_ids: list[str]) -> dict[str, ChunkRow]:
        wanted = set(chunk_ids)
        return {r.chunk_id: r for r in self.rows if r.chunk_id in wanted}


class InMemoryBenchmarkRepository:
    def __init__(self) -> None:
        self.runs: list[BenchmarkRun] = []

    def insert(self, run: BenchmarkRun) -> str:
        run.id = f"run-{len(self.runs)}"
        self.runs.append(run)
        return run.id

    def list(self, limit: int = 200) -> list[dict[str, Any]]:
        return [
            {
                "id": r.id, "document_name": r.document_name, "label": r.label,
                "llm_model": r.llm_model, "embedding_model": r.embedding_model,
                "embedding_dim": r.embedding_dim, "with_context": r.with_context,
                "cache_active": r.cache_active, "chunk_count": r.chunk_count,
                "stages": r.stages, "total_latency_ms": r.total_latency_ms,
                "total_cost_usd": r.total_cost_usd,
                "llm_input_tokens": r.llm_input_tokens,
                "llm_output_tokens": r.llm_output_tokens, "embed_tokens": r.embed_tokens,
                "created_at": None,
            }
            for r in reversed(self.runs[-limit:])
        ]


# --------------------------------------------------------------------------- #
# Eval runs (retrieval-quality benchmark: recall@k / MRR / nDCG@k per combo)
# --------------------------------------------------------------------------- #
@dataclass
class EvalRun:
    eval_set: str
    label: str
    llm_model: str | None
    embedding_model: str
    embedding_dim: int
    with_context: bool
    context_mode: str              # "none" | "per_chunk" | "per_document"
    retrieval_mode: str            # "dense" | "hybrid" (hybrid deferred)
    reranked: bool
    rerank_model: str | None
    k: int
    query_count: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    details: list[dict] = field(default_factory=list)  # per-query breakdown
    id: str | None = None
    created_at: datetime | None = None


_EVAL_COLS = [
    "id", "eval_set", "label", "llm_model", "embedding_model", "embedding_dim",
    "with_context", "context_mode", "retrieval_mode", "reranked", "rerank_model",
    "k", "query_count", "recall_at_k", "mrr", "ndcg_at_k", "details", "created_at",
]


class PostgresEvalRepository:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def insert(self, run: EvalRun) -> str:
        from psycopg.types.json import Json

        with self.pool.connection() as conn:
            res = conn.execute(
                """INSERT INTO eval_runs
                   (eval_set, label, llm_model, embedding_model, embedding_dim,
                    with_context, context_mode, retrieval_mode, reranked, rerank_model,
                    k, query_count, recall_at_k, mrr, ndcg_at_k, details)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (
                    run.eval_set, run.label, run.llm_model, run.embedding_model,
                    run.embedding_dim, run.with_context, run.context_mode,
                    run.retrieval_mode, run.reranked, run.rerank_model, run.k,
                    run.query_count, run.recall_at_k, run.mrr, run.ndcg_at_k,
                    Json(run.details),
                ),
            ).fetchone()
            conn.commit()
            return str(res[0])

    def list(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                """SELECT id, eval_set, label, llm_model, embedding_model,
                          embedding_dim, with_context, context_mode, retrieval_mode,
                          reranked, rerank_model, k, query_count, recall_at_k, mrr,
                          ndcg_at_k, details, created_at
                   FROM eval_runs ORDER BY created_at DESC LIMIT %s""",
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(zip(_EVAL_COLS, r))
            d["id"] = str(d["id"])
            for f in ("recall_at_k", "mrr", "ndcg_at_k"):
                d[f] = float(d[f])
            d["created_at"] = d["created_at"].isoformat() if d["created_at"] else None
            out.append(d)
        return out


class InMemoryEvalRepository:
    def __init__(self) -> None:
        self.runs: list[EvalRun] = []

    def insert(self, run: EvalRun) -> str:
        run.id = f"eval-{len(self.runs)}"
        self.runs.append(run)
        return run.id

    def list(self, limit: int = 200) -> list[dict[str, Any]]:
        return [
            {
                "id": r.id, "eval_set": r.eval_set, "label": r.label,
                "llm_model": r.llm_model, "embedding_model": r.embedding_model,
                "embedding_dim": r.embedding_dim, "with_context": r.with_context,
                "context_mode": r.context_mode,
                "retrieval_mode": r.retrieval_mode, "reranked": r.reranked,
                "rerank_model": r.rerank_model, "k": r.k,
                "query_count": r.query_count, "recall_at_k": r.recall_at_k,
                "mrr": r.mrr, "ndcg_at_k": r.ndcg_at_k, "details": r.details,
                "created_at": None,
            }
            for r in reversed(self.runs[-limit:])
        ]


# --------------------------------------------------------------------------- #
# Users (beta auth: email allow-list + password set on first login)
# --------------------------------------------------------------------------- #
@dataclass
class UserRow:
    email: str
    name: str | None = None
    password_hash: str | None = None
    is_active: bool = True
    id: str | None = None
    created_at: datetime | None = None
    last_login_at: datetime | None = None


class UserRepository(Protocol):
    def get_by_email(self, email: str) -> UserRow | None: ...
    def create(self, email: str, name: str | None = None) -> UserRow: ...
    def set_password(self, email: str, password_hash: str) -> None: ...
    def mark_login(self, email: str) -> None: ...


def _to_user(r: tuple) -> UserRow:
    return UserRow(
        id=str(r[0]), email=r[1], name=r[2], password_hash=r[3],
        is_active=r[4], created_at=r[5], last_login_at=r[6],
    )


_USER_COLS = (
    "SELECT id, email, name, password_hash, is_active, created_at, last_login_at "
    "FROM users"
)


class PostgresUserRepository:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def get_by_email(self, email: str) -> UserRow | None:
        with self.pool.connection() as conn:
            r = conn.execute(_USER_COLS + " WHERE email=%s", (email.lower(),)).fetchone()
        return _to_user(r) if r else None

    def create(self, email: str, name: str | None = None) -> UserRow:
        with self.pool.connection() as conn:
            r = conn.execute(
                "INSERT INTO users (email, name) VALUES (%s,%s) "
                "ON CONFLICT (email) DO UPDATE SET name=COALESCE(users.name, EXCLUDED.name) "
                "RETURNING id, email, name, password_hash, is_active, created_at, last_login_at",
                (email.lower(), name),
            ).fetchone()
            conn.commit()
        return _to_user(r)

    def set_password(self, email: str, password_hash: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE users SET password_hash=%s WHERE email=%s",
                (password_hash, email.lower()),
            )
            conn.commit()

    def mark_login(self, email: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE users SET last_login_at=now() WHERE email=%s", (email.lower(),)
            )
            conn.commit()


class InMemoryUserRepository:
    def __init__(self) -> None:
        self._users: dict[str, UserRow] = {}
        self._seq = 0

    def get_by_email(self, email: str) -> UserRow | None:
        return self._users.get(email.lower())

    def create(self, email: str, name: str | None = None) -> UserRow:
        key = email.lower()
        existing = self._users.get(key)
        if existing is not None:
            if name and not existing.name:
                existing.name = name
            return existing
        self._seq += 1
        row = UserRow(id=f"user-{self._seq}", email=key, name=name,
                      created_at=datetime.now(timezone.utc))
        self._users[key] = row
        return row

    def set_password(self, email: str, password_hash: str) -> None:
        u = self._users.get(email.lower())
        if u is not None:
            u.password_hash = password_hash

    def mark_login(self, email: str) -> None:
        u = self._users.get(email.lower())
        if u is not None:
            u.last_login_at = datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Actions (proactive/approval flow: proposed outbound actions await approval)
# --------------------------------------------------------------------------- #
@dataclass
class ActionRow:
    kind: str
    payload: str
    target: str | None = None
    summary: str | None = None
    status: str = "pending"
    injection_flag: bool = False
    proposed_by: str | None = None
    decided_by: str | None = None
    error: str | None = None
    id: str | None = None
    created_at: datetime | None = None
    decided_at: datetime | None = None


class ActionRepository(Protocol):
    def create(self, row: ActionRow) -> str: ...
    def get(self, action_id: str) -> ActionRow | None: ...
    def list_pending(self, limit: int = 100) -> list[ActionRow]: ...
    def set_status(
        self, action_id: str, status: str, *,
        decided_by: str | None = None, error: str | None = None,
    ) -> None: ...


_ACTION_COLS = (
    "SELECT id, kind, status, target, summary, payload, injection_flag, "
    "proposed_by, decided_by, error, created_at, decided_at FROM actions"
)


def _to_action(r: tuple) -> ActionRow:
    return ActionRow(
        id=str(r[0]), kind=r[1], status=r[2], target=r[3], summary=r[4],
        payload=r[5], injection_flag=r[6], proposed_by=r[7], decided_by=r[8],
        error=r[9], created_at=r[10], decided_at=r[11],
    )


class PostgresActionRepository:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def create(self, row: ActionRow) -> str:
        with self.pool.connection() as conn:
            res = conn.execute(
                """INSERT INTO actions
                   (kind, status, target, summary, payload, injection_flag, proposed_by)
                   VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (row.kind, row.status, row.target, row.summary, row.payload,
                 row.injection_flag, row.proposed_by),
            ).fetchone()
            conn.commit()
            return str(res[0])

    def get(self, action_id: str) -> ActionRow | None:
        with self.pool.connection() as conn:
            r = conn.execute(_ACTION_COLS + " WHERE id=%s", (action_id,)).fetchone()
        return _to_action(r) if r else None

    def list_pending(self, limit: int = 100) -> list[ActionRow]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                _ACTION_COLS + " WHERE status='pending' ORDER BY created_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [_to_action(r) for r in rows]

    def set_status(
        self, action_id: str, status: str, *,
        decided_by: str | None = None, error: str | None = None,
    ) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE actions SET status=%s, decided_by=COALESCE(%s, decided_by), "
                "error=%s, decided_at=now() WHERE id=%s",
                (status, decided_by, error, action_id),
            )
            conn.commit()


class InMemoryActionRepository:
    def __init__(self) -> None:
        self._actions: dict[str, ActionRow] = {}
        self._seq = 0

    def create(self, row: ActionRow) -> str:
        self._seq += 1
        action_id = f"action-{self._seq}"
        self._actions[action_id] = ActionRow(
            id=action_id, kind=row.kind, status=row.status, target=row.target,
            summary=row.summary, payload=row.payload, injection_flag=row.injection_flag,
            proposed_by=row.proposed_by, created_at=datetime.now(timezone.utc),
        )
        return action_id

    def get(self, action_id: str) -> ActionRow | None:
        return self._actions.get(action_id)

    def list_pending(self, limit: int = 100) -> list[ActionRow]:
        pending = [a for a in self._actions.values() if a.status == "pending"]
        return list(reversed(pending))[:limit]

    def set_status(
        self, action_id: str, status: str, *,
        decided_by: str | None = None, error: str | None = None,
    ) -> None:
        a = self._actions.get(action_id)
        if a is not None:
            a.status = status
            if decided_by is not None:
                a.decided_by = decided_by
            a.error = error
            a.decided_at = datetime.now(timezone.utc)
