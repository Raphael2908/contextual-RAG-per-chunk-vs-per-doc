-- Bulk upload: a `batch` groups the documents from one mass-upload (~1000 PDFs).
-- The upload endpoint creates a batch, tags each document with batch_id, and an
-- in-process async worker pool ingests them (no Celery/Redis in this slice).
-- Batch progress is a pure SQL aggregate over documents.batch_id, so per-doc
-- chunk_count + cost are persisted here too (IngestResult carries them at runtime
-- but the original slice never stored them).

CREATE TABLE IF NOT EXISTS batches (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    label        text,
    with_context boolean NOT NULL DEFAULT true,    -- false = cheap no-context throughput run
    total        int     NOT NULL DEFAULT 0,       -- documents accepted into the batch
    created_at   timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE documents ADD COLUMN IF NOT EXISTS batch_id uuid
    REFERENCES batches (id) ON DELETE SET NULL;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunk_count int NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS total_cost_usd numeric(12, 6) NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS documents_batch_id_idx ON documents (batch_id);
