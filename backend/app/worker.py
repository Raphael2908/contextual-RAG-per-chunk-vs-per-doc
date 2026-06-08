"""In-process async ingest worker pool for bulk upload (no Celery/Redis).

Bulk upload decouples a fast upload step (read bytes → byte-hash dedup → blob put
→ create a `queued` documents row) from the slow ingest step (profile → chunk →
enrich → embed → upsert). The upload endpoint enqueues a tiny `IngestJob`
(doc_id + storage_path, NOT the bytes), and this pool ingests them with bounded
concurrency.

`IngestionService.process_existing()` is synchronous and blocking (LiteLLM +
Milvus + psycopg), so each job runs in a ThreadPoolExecutor via
`loop.run_in_executor` — never on the event loop. The number of asyncio worker
tasks equals the executor size (one in-flight blocking job per thread).

Lifecycle: `start()` in the FastAPI lifespan, `drain_and_stop()` in its finally
(joins the queue so in-flight + queued work finishes before shutdown). A hard
process kill still leaves rows `queued` — a requeue-on-startup sweep is a noted
follow-up, deliberately out of scope for this slice.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from app.blob import BlobStore
from app.ingest import IngestionService
from app.repository import DocumentRepository

log = logging.getLogger("app.worker")


@dataclass
class IngestJob:
    doc_id: str
    storage_path: str
    file_type: str
    with_context: bool
    batch_id: str | None = None
    context_mode: str | None = None  # None → derived from with_context


class IngestWorkerPool:
    def __init__(
        self,
        *,
        ingestion: IngestionService,
        blob: BlobStore,
        doc_repo: DocumentRepository,
        concurrency: int,
        queue_max: int,
    ):
        self.ingestion = ingestion
        self.blob = blob
        self.doc_repo = doc_repo
        self.concurrency = max(1, concurrency)
        self.queue: asyncio.Queue[IngestJob] = asyncio.Queue(maxsize=queue_max)
        self.executor = ThreadPoolExecutor(
            max_workers=self.concurrency, thread_name_prefix="ingest"
        )
        self._workers: list[asyncio.Task] = []

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._workers = [
            loop.create_task(self._worker(i), name=f"ingest-worker-{i}")
            for i in range(self.concurrency)
        ]
        log.info("ingest worker pool started: %d workers", self.concurrency)

    async def enqueue(self, job: IngestJob) -> None:
        # Awaits when the queue is full → backpressure on the upload handler.
        await self.queue.put(job)

    async def _worker(self, idx: int) -> None:
        while True:
            job = await self.queue.get()
            try:
                await self._loop.run_in_executor(self.executor, self._run_blocking, job)
            except Exception:  # noqa: BLE001 — never let a worker die on one job
                log.exception("ingest worker %d failed on doc %s", idx, job.doc_id)
            finally:
                self.queue.task_done()

    def _run_blocking(self, job: IngestJob) -> None:
        # Runs in a worker thread. process_existing() swallows its own errors into
        # a failed IngestResult, so this is defensive belt-and-braces.
        data = self.blob.get(_blob_key(job.storage_path))
        result = self.ingestion.process_existing(
            job.doc_id, data, job.file_type,
            with_context=job.with_context, context_mode=job.context_mode,
        )
        self.doc_repo.set_result(
            job.doc_id, result.status,
            chunk_count=result.chunk_count,
            total_cost_usd=round(result.total_cost_usd, 6),
            llm_input_tokens=result.llm_input_tokens,
            llm_output_tokens=result.llm_output_tokens,
            embed_tokens=result.embed_tokens,
            error=result.error,
        )

    async def drain_and_stop(self) -> None:
        """Finish queued + in-flight work, then cancel workers and the executor."""
        try:
            await self.queue.join()
        finally:
            for w in self._workers:
                w.cancel()
            for w in self._workers:
                try:
                    await w
                except asyncio.CancelledError:
                    pass
            self.executor.shutdown(wait=True)
            log.info("ingest worker pool stopped")


def _blob_key(storage_path: str) -> str:
    """The blob store keys by the `{fhash}_{name}` filename. FilesystemBlobStore
    returns an absolute path from put(); InMemoryBlobStore returns `mem://{key}`.
    Recover the key from either so the worker can read the bytes back."""
    if storage_path.startswith("mem://"):
        return storage_path[len("mem://"):]
    return os.path.basename(storage_path)
