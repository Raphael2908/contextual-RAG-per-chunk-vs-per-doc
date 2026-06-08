"""IngestionService — orchestrates the (instrumented) ingestion pipeline.

Steps (cheapest first), each timed and costed:
  1. byte-hash dedup → 2. blob store → 3. text extraction → 4. content-hash
  → 5. document profiling (1 LLM pass) → 6. structure-aware chunking
  → 7. contextual enrichment (context_mode: none | per_chunk | per_document)
  → 8. embed + upsert to the vector store + persist chunks.

The SAME service backs both the API (default models) and the benchmark (any
injected LLM/Embedder). `context_mode="none"` (or `llm=None`) runs the no-context
control; `per_chunk` enriches each chunk; `per_document` writes one shared block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from app import contextual, extract, hashing
from app.blob import BlobStore
from app.chunk import chunk_text
from app.embeddings import Embedder
from app.llm import LLM
from app.metrics import StageMetric, timed
from app.profile import DocProfile, profile_document
from app.repository import (
    ChunkRow,
    DocumentRepository,
    DocumentRow,
    PostgresChunkRepository,
)
from app.vectorstore import VectorStore, collection_for

EMBED_BATCH = 64

CONTEXT_MODES = ("none", "per_chunk", "per_document")


def resolve_context_mode(context_mode: str | None, with_context: bool) -> str:
    """Effective mode from the new knob, falling back to the legacy boolean
    (False → "none", True → "per_chunk"). Unknown strings fall back too."""
    if context_mode in CONTEXT_MODES:
        return context_mode
    return "per_chunk" if with_context else "none"


@dataclass
class IngestResult:
    document_id: str
    status: str
    chunk_count: int
    collection: str
    embedding_dim: int
    with_context: bool
    cache_active: bool
    context_mode: str = "per_chunk"
    stages: list[StageMetric] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    embed_tokens: int = 0
    error: str | None = None

    def stage_dicts(self) -> list[dict]:
        return [s.to_dict() for s in self.stages]


class IngestionService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        blob: BlobStore,
        doc_repo: DocumentRepository,
        chunk_repo: PostgresChunkRepository | None,
        llm: LLM | None,
        collection_suffix: str = "",
        hybrid: bool = False,
    ):
        self.embedder = embedder
        self.store = store
        self.blob = blob
        self.doc_repo = doc_repo
        self.chunk_repo = chunk_repo
        self.llm = llm
        # Disambiguates collections that share an embedding model (e.g. the
        # context vs no-context combos in the quality eval), so their vectors
        # never mix in one collection. "" → the default per-model collection.
        self.collection_suffix = collection_suffix
        # Hybrid (dense + sparse/BM25) collection layout vs dense-only. Hybrid
        # collections carry the chunk text + a BM25 function (a distinct schema,
        # hence a distinct name via collection_for(..., hybrid=True)).
        self.hybrid = hybrid

    def ingest(
        self,
        data: bytes,
        name: str,
        file_type: str,
        *,
        with_context: bool = True,
        context_mode: str | None = None,
        enforce_dedup: bool = True,
        persist_chunks: bool = True,
    ) -> IngestResult:
        stages: list[StageMetric] = []
        upload_date = datetime.now(timezone.utc).date()
        mode = resolve_context_mode(context_mode, with_context)

        # 1. byte-hash dedup
        with timed("byte_hash", stages):
            fhash = hashing.byte_hash(data)
        if enforce_dedup:
            existing = self.doc_repo.find_by_file_hash(fhash)
            if existing is not None:
                return IngestResult(
                    document_id=existing.id, status="duplicate", chunk_count=0,
                    collection="", embedding_dim=0, with_context=mode != "none",
                    cache_active=False, context_mode=mode, stages=stages,
                )

        # 2. store original in the blob volume + create the document row
        with timed("blob_put", stages):
            storage_path = self.blob.put(f"{fhash}_{name}", data)
        doc_id = self.doc_repo.create(
            DocumentRow(
                id="", name=name, file_type=file_type, storage_path=storage_path,
                file_hash=fhash, size=len(data), status="processing",
            )
        )

        try:
            return self._process(
                doc_id, data, file_type, fhash, upload_date, mode,
                persist_chunks, stages,
            )
        except Exception as exc:  # noqa: BLE001 — surface as a failed document
            self.doc_repo.set_status(doc_id, "failed", str(exc))
            return IngestResult(
                document_id=doc_id, status="failed", chunk_count=0, collection="",
                embedding_dim=0, with_context=mode != "none", cache_active=False,
                context_mode=mode, stages=stages, error=str(exc),
            )

    def process_existing(
        self,
        doc_id: str,
        data: bytes,
        file_type: str,
        *,
        with_context: bool = True,
        context_mode: str | None = None,
        persist_chunks: bool = True,
    ) -> IngestResult:
        """Ingest an already-created `documents` row (byte-hash dedup, blob put and
        row create having happened at upload time — see routers/batches.py).

        Runs only steps 3–8 of the pipeline against `doc_id`. Used by the bulk
        worker pool so upload (fast I/O) and ingest (slow LLM/embeddings) are
        decoupled. Mirrors `ingest()`'s error handling: a failure marks the row
        `failed` and returns a failed IngestResult rather than raising, so one
        poison document never kills a worker.
        """
        stages: list[StageMetric] = []
        upload_date = datetime.now(timezone.utc).date()
        mode = resolve_context_mode(context_mode, with_context)
        self.doc_repo.set_status(doc_id, "processing")
        try:
            return self._process(
                doc_id, data, file_type, None, upload_date, mode,
                persist_chunks, stages,
            )
        except Exception as exc:  # noqa: BLE001 — surface as a failed document
            self.doc_repo.set_status(doc_id, "failed", str(exc))
            return IngestResult(
                document_id=doc_id, status="failed", chunk_count=0, collection="",
                embedding_dim=0, with_context=mode != "none", cache_active=False,
                context_mode=mode, stages=stages, error=str(exc),
            )

    def _process(
        self, doc_id, data, file_type, fhash, upload_date, mode,
        persist_chunks, stages,
    ) -> IngestResult:
        use_llm = self.llm if mode != "none" else None
        # 3. text extraction
        with timed("extract", stages) as m:
            text = extract.extract_text(data, file_type)
            m.extra["chars"] = len(text)
        if not text.strip():
            raise ValueError("no extractable text")

        # 4. content-hash
        with timed("content_hash", stages):
            chash = hashing.content_hash(text)

        # 5. document profiling (one LLM pass; cheap model)
        if use_llm is not None:
            with timed("profile", stages) as m:
                profile = profile_document(use_llm, text, upload_date=upload_date)
                _fold_llm(m, profile.result)
        else:
            profile = DocProfile(summary="(no-context control)", effective_date=upload_date)
        self.doc_repo.update_profile(
            doc_id, summary=profile.summary, effective_date=profile.effective_date,
            content_hash=chash,
        )

        # 6. structure-aware chunking
        with timed("chunk", stages) as m:
            chunks = chunk_text(text)
            m.extra["chunks"] = len(chunks)

        # 7. contextual enrichment (none | per_chunk | per_document)
        cache_active = False
        enriched = []
        with timed("contextual_enrichment", stages) as m:
            if mode == "per_document" and use_llm is not None:
                # One longer block per document, reused across every chunk.
                row = self.doc_repo.get(doc_id)
                file_name = row.name if row is not None else doc_id
                block, res = contextual.build_document_context(
                    use_llm, document_id=doc_id, profile=profile,
                    full_text=text, file_name=file_name,
                )
                _fold_llm(m, res)
                cache_active = res.cache_active
                enriched = [contextual.append_document_context(block, ch) for ch in chunks]
            else:
                # per_chunk (one LLM call/chunk) or none (use_llm is None → bare chunk).
                for ch in chunks:
                    ec = contextual.enrich_chunk(
                        use_llm, document_id=doc_id, profile=profile,
                        full_text=text, chunk=ch,
                    )
                    enriched.append(ec)
                    if ec.result is not None:
                        _fold_llm(m, ec.result)
                        cache_active = cache_active or ec.result.cache_active

        # 8. embed + upsert + persist
        embed_texts = [ec.context_text for ec in enriched]
        embed_tokens = 0
        embed_cost = 0.0
        vectors: list[list[float]] = []
        dim = self.embedder.dim
        with timed("embed", stages) as m:
            for i in range(0, len(embed_texts), EMBED_BATCH):
                batch = embed_texts[i : i + EMBED_BATCH]
                res = self.embedder.embed(batch, input_type="document")
                vectors.extend(res.vectors)
                embed_tokens += res.input_tokens
                embed_cost += res.cost_usd
                dim = res.dim or dim
            m.input_tokens = embed_tokens
            m.cost_usd = embed_cost

        collection = (
            collection_for(
                self.embedder.model, dim or 1, context_mode=mode, hybrid=self.hybrid
            )
            + self.collection_suffix
        )
        with timed("vector_upsert", stages) as m:
            self.store.ensure_collection(collection, dim or 1, hybrid=self.hybrid)
            chunk_ids = [f"{doc_id}:{ec.chunk.index}" for ec in enriched]
            self.store.upsert(
                collection, chunk_ids, vectors, [doc_id] * len(chunk_ids),
                texts=embed_texts, hybrid=self.hybrid,
            )
            m.extra["collection"] = collection

        if persist_chunks and self.chunk_repo is not None:
            with timed("persist_chunks", stages):
                self.chunk_repo.insert_many(
                    [
                        ChunkRow(
                            document_id=doc_id, chunk_index=ec.chunk.index,
                            source_location=ec.chunk.source_location,
                            text=ec.chunk.text, context_text=ec.context_text,
                            token_count=len(ec.context_text.split()),
                        )
                        for ec in enriched
                    ]
                )

        self.doc_repo.set_status(doc_id, "ready")

        llm_in = sum(s.input_tokens for s in stages if s.stage in ("profile", "contextual_enrichment"))
        llm_out = sum(s.output_tokens for s in stages if s.stage in ("profile", "contextual_enrichment"))
        return IngestResult(
            document_id=doc_id, status="ready", chunk_count=len(enriched),
            collection=collection, embedding_dim=dim or 0, with_context=mode != "none",
            cache_active=cache_active, context_mode=mode, stages=stages,
            total_latency_ms=sum(s.latency_ms for s in stages),
            total_cost_usd=sum(s.cost_usd for s in stages),
            llm_input_tokens=llm_in, llm_output_tokens=llm_out,
            embed_tokens=embed_tokens,
        )


def _fold_llm(metric: StageMetric, result) -> None:
    if result is None:
        return
    metric.input_tokens += result.input_tokens
    metric.output_tokens += result.output_tokens
    metric.cache_read_tokens += result.cache_read_tokens
    metric.cache_creation_tokens += result.cache_creation_tokens
    metric.cost_usd += result.cost_usd
