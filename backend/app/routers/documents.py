"""Document endpoints: upload (synchronous ingest), list, status.

Ingestion runs inline (no Celery/Redis in this slice), so the upload response
already carries the ingest summary + per-stage metrics.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Request, UploadFile

from app import extract
from app.ingest import IngestionService
from app.security import Identity, require_identity

router = APIRouter(tags=["documents"])


@router.post("/documents")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    identity: Identity = Depends(require_identity),
):
    data = await file.read()
    name = file.filename or "upload.bin"
    file_type = extract.file_type_of(name)

    service: IngestionService = request.app.state.ingestion
    result = service.ingest(
        data, name, file_type, context_mode=request.app.state.settings.context_mode
    )

    return {
        "document_id": result.document_id,
        "status": result.status,
        "chunk_count": result.chunk_count,
        "collection": result.collection,
        "embedding_dim": result.embedding_dim,
        "cache_active": result.cache_active,
        "context_mode": result.context_mode,
        "total_latency_ms": round(result.total_latency_ms, 1),
        "total_cost_usd": round(result.total_cost_usd, 6),
        "stages": result.stage_dicts(),
        "error": result.error,
    }


@router.get("/documents")
def list_documents(request: Request, identity: Identity = Depends(require_identity)):
    repo = request.app.state.doc_repo
    return [
        {
            "id": d.id, "name": d.name, "file_type": d.file_type, "status": d.status,
            "size": d.size, "summary": d.summary,
            "effective_date": d.effective_date.isoformat() if d.effective_date else None,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in repo.list()
    ]


@router.get("/documents/{document_id}/status")
def document_status(
    request: Request, document_id: str, identity: Identity = Depends(require_identity)
):
    repo = request.app.state.doc_repo
    doc = repo.get(document_id)
    if doc is None:
        return {"status": "not_found"}
    return {"id": doc.id, "status": doc.status, "error": doc.error}
