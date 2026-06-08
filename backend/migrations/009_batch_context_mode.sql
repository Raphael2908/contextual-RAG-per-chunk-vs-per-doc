-- Lets a bulk batch pick its contextual-retrieval strategy (none | per_chunk |
-- per_document), so a 1000-PDF run can compare per-chunk vs per-document INGEST
-- COST + THROUGHPUT at scale (where the eval's 3-doc corpus has no signal).
-- `with_context` (migration 003) stays as the coarse boolean; this is the precise axis.

ALTER TABLE batches
    ADD COLUMN IF NOT EXISTS context_mode text NOT NULL DEFAULT 'per_chunk';
