"""Infra smoke test: real Postgres + real Milvus, fake models (no API keys).

Validates migrations, the Postgres repositories, and Milvus collection
create/upsert/count against the live docker stack — the parts the dependency-free
smoke test can't reach. Run after `docker compose up -d postgres milvus`.

    PG_URL=postgresql://postgres:postgres@localhost:5432/company_brain \\
    MILVUS_URI=http://localhost:19530 \\
    backend/.venv/bin/python backend/scripts/smoke_infra.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import extract  # noqa: E402
from app.blob import FilesystemBlobStore  # noqa: E402
from app.db import make_pool, run_migrations  # noqa: E402
from app.embeddings import FakeEmbedder  # noqa: E402
from app.ingest import IngestionService  # noqa: E402
from app.llm import FakeLLM  # noqa: E402
from app.repository import (  # noqa: E402
    PostgresChunkRepository,
    PostgresDocumentRepository,
)
from app.vectorstore import MilvusStore  # noqa: E402


def main() -> int:
    pg_url = os.getenv("PG_URL", "postgresql://postgres:postgres@localhost:5432/company_brain")
    milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")

    pool = make_pool(pg_url)
    run_migrations(pool)
    print("migrations applied")

    doc_repo = PostgresDocumentRepository(pool)
    chunk_repo = PostgresChunkRepository(pool)
    store = MilvusStore(milvus_uri)
    blob = FilesystemBlobStore("./data/blobs")

    service = IngestionService(
        embedder=FakeEmbedder("fake/voyage-law-2", dim=1024),
        store=store, blob=blob, doc_repo=doc_repo, chunk_repo=chunk_repo,
        llm=FakeLLM("fake/haiku"),
    )

    path = Path(__file__).resolve().parents[2] / "samples" / "sample_agreement.md"
    data = path.read_bytes()
    res = service.ingest(
        data, "sample_agreement.md", extract.file_type_of(path.name),
        with_context=True, enforce_dedup=False,
    )
    print(f"ingest: status={res.status} chunks={res.chunk_count} collection={res.collection}")
    assert res.status == "ready", res.error

    # Postgres assertions
    doc = doc_repo.get(res.document_id)
    assert doc is not None and doc.status == "ready", "document row not ready"
    print(f"postgres: document {doc.id} status={doc.status} summary={doc.summary!r}")

    # Milvus assertions
    count = store.count(res.collection)
    print(f"milvus: collection '{res.collection}' vector count={count}")
    assert count >= res.chunk_count, "milvus vector count mismatch"

    pool.close()
    print("\nINFRA SMOKE OK — Postgres + Milvus verified with the real datastores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
