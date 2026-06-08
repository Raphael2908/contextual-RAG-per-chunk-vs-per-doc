"""Dependency-free smoke test: bulk upload + async worker pool on fakes.

    python backend/scripts/smoke_bulk.py [n_docs]

Exercises the decoupled bulk path end to end with zero external services:
create a batch → run the cheap upload prefix (byte-hash dedup → blob put →
queued row) → enqueue onto the worker pool → drain → assert every doc reached
`ready` (plus an injected exact-duplicate is caught), and the batch summary
counts/totals add up.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import extract, hashing  # noqa: E402
from app.blob import InMemoryBlobStore  # noqa: E402
from app.embeddings import FakeEmbedder  # noqa: E402
from app.ingest import IngestionService  # noqa: E402
from app.llm import FakeLLM  # noqa: E402
from app.repository import (  # noqa: E402
    DocumentRow,
    InMemoryBatchRepository,
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
)
from app.vectorstore import InMemoryStore  # noqa: E402
from app.worker import IngestJob, IngestWorkerPool  # noqa: E402


def _make_docs(n: int) -> list[tuple[str, bytes]]:
    """N distinct synthetic docs + one exact duplicate of the first."""
    base = (
        "MUTUAL NON-DISCLOSURE AGREEMENT\n\n"
        "1. Definitions. Confidential Information means any data disclosed by "
        "the Disclosing Party to the Receiving Party.\n\n"
        "2. Term. This Agreement is effective as of {date} between {a} and {b}.\n\n"
        "3. Obligations. The Receiving Party shall hold the Confidential "
        "Information in strict confidence for a period of {years} years.\n"
    )
    docs: list[tuple[str, bytes]] = []
    for i in range(n):
        text = base.format(
            date=f"2026-0{(i % 9) + 1}-15",
            a=f"Acme Holdings {i}", b=f"Beta Industries {i}", years=(i % 5) + 1,
        )
        docs.append((f"nda_{i:04d}.txt", text.encode("utf-8")))
    if docs:
        docs.append(("nda_dup.txt", docs[0][1]))  # exact duplicate of doc 0
    return docs


async def _run(n: int) -> int:
    doc_repo = InMemoryDocumentRepository()
    chunk_repo = InMemoryChunkRepository()
    batch_repo = InMemoryBatchRepository(doc_repo)
    blob = InMemoryBlobStore()
    store = InMemoryStore()

    ingestion = IngestionService(
        embedder=FakeEmbedder("fake/voyage-law-2", dim=1024),
        store=store, blob=blob, doc_repo=doc_repo, chunk_repo=chunk_repo,
        llm=FakeLLM("fake/haiku"),
    )
    pool = IngestWorkerPool(
        ingestion=ingestion, blob=blob, doc_repo=doc_repo,
        concurrency=4, queue_max=1000,
    )
    await pool.start()

    batch_id = batch_repo.create("smoke", with_context=True)
    docs = _make_docs(n)
    print(f"Batch {batch_id}: uploading {len(docs)} files ({n} unique + 1 dup)\n")

    duplicates = 0
    for name, data in docs:
        file_type = extract.file_type_of(name)
        fhash = hashing.byte_hash(data)
        existing = doc_repo.find_by_file_hash(fhash)
        if existing is not None:
            doc_repo.create(DocumentRow(
                id="", name=name, file_type=file_type, storage_path=existing.storage_path,
                file_hash=fhash, size=len(data), status="duplicate", batch_id=batch_id,
            ))
            batch_repo.bump_total(batch_id)
            duplicates += 1
            continue
        storage_path = blob.put(f"{fhash}_{name}", data)
        doc_id = doc_repo.create(DocumentRow(
            id="", name=name, file_type=file_type, storage_path=storage_path,
            file_hash=fhash, size=len(data), status="queued", batch_id=batch_id,
        ))
        batch_repo.bump_total(batch_id)
        await pool.enqueue(IngestJob(
            doc_id=doc_id, storage_path=storage_path, file_type=file_type,
            with_context=True, batch_id=batch_id,
        ))

    await pool.drain_and_stop()

    summary = batch_repo.summary(batch_id)
    counts = summary["counts"]
    print(f"  counts:  {counts}")
    print(f"  totals:  {summary['totals']}")

    assert summary["total"] == len(docs), f"total mismatch: {summary['total']} != {len(docs)}"
    assert counts.get("ready", 0) == n, f"expected {n} ready, got {counts.get('ready', 0)}"
    assert counts.get("duplicate", 0) == duplicates, "duplicate not caught"
    assert counts.get("failed", 0) == 0, f"unexpected failures: {counts.get('failed')}"
    assert sum(counts.values()) == len(docs), "counts don't sum to total"
    assert summary["totals"]["chunk_count"] > 0, "no chunks across the batch"

    print("\nSMOKE OK — bulk upload + async worker pool verified on fakes.")
    return 0


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    return asyncio.run(_run(n))


if __name__ == "__main__":
    raise SystemExit(main())
