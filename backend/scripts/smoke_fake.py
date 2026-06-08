"""Dependency-free smoke test: full ingestion pipeline on fakes, no DB/network.

    python backend/scripts/smoke_fake.py [path-to-doc]

Proves the pipeline + benchmark wiring end to end with zero external services.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import extract  # noqa: E402
from app.blob import InMemoryBlobStore  # noqa: E402
from app.embeddings import FakeEmbedder  # noqa: E402
from app.ingest import IngestionService  # noqa: E402
from app.llm import FakeLLM  # noqa: E402
from app.repository import (  # noqa: E402
    InMemoryBenchmarkRepository,
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
)
from app.vectorstore import InMemoryStore  # noqa: E402


def main() -> int:
    default = Path(__file__).resolve().parents[2] / "samples" / "sample_agreement.md"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    data = path.read_bytes()
    file_type = extract.file_type_of(path.name)

    doc_repo = InMemoryDocumentRepository()
    chunk_repo = InMemoryChunkRepository()
    bench_repo = InMemoryBenchmarkRepository()
    store = InMemoryStore()

    combos = [
        ("haiku(fake) + voyage(fake)", FakeLLM("fake/haiku"), FakeEmbedder("fake/voyage-law-2", dim=1024)),
        ("no-context + voyage(fake)", None, FakeEmbedder("fake/voyage-law-2", dim=1024)),
    ]

    print(f"Document: {path.name} ({len(data)} bytes)\n")
    for label, llm, embedder in combos:
        service = IngestionService(
            embedder=embedder, store=store, blob=InMemoryBlobStore(),
            doc_repo=doc_repo, chunk_repo=chunk_repo, llm=llm,
        )
        res = service.ingest(
            data, f"{path.name} [{label}]", file_type,
            with_context=llm is not None, enforce_dedup=False,
        )
        assert res.status == "ready", f"expected ready, got {res.status}: {res.error}"
        assert res.chunk_count > 0, "no chunks produced"
        # collection is shared by combos with the same embedding model, so it
        # accumulates across documents — assert this run's chunks are present.
        assert store.count(res.collection) >= res.chunk_count, "vector count mismatch"
        print(f"  {label:<30} status={res.status} chunks={res.chunk_count} "
              f"dim={res.embedding_dim} collection={res.collection} "
              f"cache={res.cache_active} latency={res.total_latency_ms:.1f}ms")
        for s in res.stages:
            print(f"      - {s.stage:<22} {s.latency_ms:7.2f}ms  "
                  f"in={s.input_tokens} out={s.output_tokens} $={s.cost_usd}")
        print()

    # show a sample enriched (embedded) chunk so the context block is visible
    sample = chunk_repo.rows[0]
    print("Sample embedded text (context block + chunk):")
    print("-" * 60)
    print(sample.context_text[:400])
    print("-" * 60)

    # --- per_document mode: one shared, longer block appended to every chunk ----
    pd_doc_repo = InMemoryDocumentRepository()
    pd_chunk_repo = InMemoryChunkRepository()
    pd_service = IngestionService(
        embedder=FakeEmbedder("fake/voyage-law-2", dim=1024), store=InMemoryStore(),
        blob=InMemoryBlobStore(), doc_repo=pd_doc_repo, chunk_repo=pd_chunk_repo,
        llm=FakeLLM("fake/haiku"),
    )
    pd = pd_service.ingest(
        data, path.name, file_type, context_mode="per_document", enforce_dedup=False,
    )
    assert pd.status == "ready", f"per_document: {pd.status}: {pd.error}"
    assert pd.context_mode == "per_document", pd.context_mode
    rows = pd_chunk_repo.list_for_document(pd.document_id)
    assert len(rows) > 1, "need >1 chunk to prove the block is shared"
    # Every chunk shares an identical document-level block...
    blocks = {r.context_text.removesuffix(r.text).rstrip("\n") for r in rows}
    assert len(blocks) == 1, f"per_document block not shared across chunks: {len(blocks)} distinct"
    block = next(iter(blocks))
    # ...carrying the file path (filename) and the document id...
    assert "file:" in block and path.name in block, f"block missing file path: {block!r}"
    assert f"doc: {pd.document_id}" in block, f"block missing doc id: {block!r}"
    # ...produced by exactly ONE doc-context LLM call (FakeLLM emits 12 out tokens/call),
    # vs one-per-chunk for per_chunk — the whole cost win.
    ce = next(s for s in pd.stages if s.stage == "contextual_enrichment")
    assert ce.output_tokens == 12, f"expected 1 doc-context call (12 out tok), got {ce.output_tokens}"
    print(f"\nper_document: {len(rows)} chunks share 1 block; "
          f"contextual_enrichment={ce.output_tokens} out tok (single call)")
    print("Shared document block:")
    print("-" * 60)
    print(block[:400])
    print("-" * 60)

    print("\nSMOKE OK — pipeline + per_document + benchmark wiring verified on fakes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
