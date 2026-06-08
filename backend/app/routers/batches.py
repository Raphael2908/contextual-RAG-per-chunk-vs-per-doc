"""Bulk upload endpoints: create a batch, stream files into it, poll progress.

The architecture's deferred "~1000 PDFs at once" path, fleshed out within the
slice's constraints (no Celery/Redis). Upload is decoupled from ingest: each
file POST does only the cheap, deterministic prefix (read → byte-hash dedup →
blob put → create a `queued` documents row tagged with batch_id) and enqueues a
small job onto the in-process worker pool (see app/worker.py). The client streams
files through with bounded concurrency; the queue's maxsize provides backpressure.
Progress is a SQL aggregate over documents.batch_id, polled via GET /batches/{id}.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

from app import extract, hashing
from app.ingest import resolve_context_mode
from app.repository import DocumentRow
from app.security import Identity, require_identity
from app.worker import IngestJob

router = APIRouter(tags=["batches"])


class CreateBatch(BaseModel):
    label: str | None = None
    with_context: bool = True
    # none | per_chunk | per_document. None → fall back to with_context (and the
    # server default RAG_CONTEXT_MODE when with_context is left at its True default).
    context_mode: str | None = None


@router.post("/batches")
def create_batch(
    request: Request, body: CreateBatch, identity: Identity = Depends(require_identity)
):
    default_mode = request.app.state.settings.context_mode
    # Explicit context_mode wins; else derive from with_context, but honour the
    # server default when the caller left with_context at its True default.
    mode = resolve_context_mode(
        body.context_mode, body.with_context
    ) if (body.context_mode is not None or not body.with_context) else default_mode
    batch_id = request.app.state.batch_repo.create(
        body.label, mode != "none", context_mode=mode
    )
    return {
        "batch_id": batch_id, "with_context": mode != "none",
        "context_mode": mode, "label": body.label,
    }


@router.post("/batches/{batch_id}/documents")
async def add_document(
    request: Request,
    batch_id: str,
    file: UploadFile = File(...),
    dedup: bool = Query(True, description="byte-hash dedup; pass false for a benchmark "
                        "re-ingest of identical bytes under a different context_mode"),
    identity: Identity = Depends(require_identity),
):
    state = request.app.state
    batch = state.batch_repo.get(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="batch not found")

    data = await file.read()
    name = file.filename or "upload.bin"
    file_type = extract.file_type_of(name)
    fhash = hashing.byte_hash(data)

    # Cheap prefix (mirrors IngestionService.ingest's dedup + blob + row create),
    # so the heavy pipeline can run later on the worker pool.
    existing = state.doc_repo.find_by_file_hash(fhash) if dedup else None
    if existing is not None:
        # Record the duplicate against this batch so the counts stay honest,
        # but don't re-ingest.
        doc_id = state.doc_repo.create(
            DocumentRow(
                id="", name=name, file_type=file_type, storage_path=existing.storage_path,
                file_hash=fhash, size=len(data), status="duplicate", batch_id=batch_id,
            )
        )
        state.batch_repo.bump_total(batch_id)
        return {"document_id": doc_id, "status": "duplicate"}

    storage_path = state.blob.put(f"{fhash}_{name}", data)
    doc_id = state.doc_repo.create(
        DocumentRow(
            id="", name=name, file_type=file_type, storage_path=storage_path,
            file_hash=fhash, size=len(data), status="queued", batch_id=batch_id,
        )
    )
    state.batch_repo.bump_total(batch_id)
    await state.worker_pool.enqueue(
        IngestJob(
            doc_id=doc_id, storage_path=storage_path, file_type=file_type,
            with_context=batch["with_context"], batch_id=batch_id,
            context_mode=batch.get("context_mode"),
        )
    )
    return {"document_id": doc_id, "status": "queued"}


@router.get("/batches/{batch_id}")
def get_batch(
    request: Request,
    batch_id: str,
    summary_only: bool = Query(False, description="omit the per-doc list (lighter polls)"),
    identity: Identity = Depends(require_identity),
):
    summary = request.app.state.batch_repo.summary(batch_id, include_docs=not summary_only)
    if summary is None:
        raise HTTPException(status_code=404, detail="batch not found")
    return summary
