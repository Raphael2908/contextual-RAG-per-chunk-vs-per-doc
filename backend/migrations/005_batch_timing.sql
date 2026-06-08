-- Per-document terminal-transition time, so a batch's elapsed wall-time is a
-- pure SQL aggregate: max(coalesce(updated_at, created_at)) - batches.created_at.
-- Nullable on purpose: pre-feature rows and still-queued docs read through the
-- COALESCE to created_at, and the migration stays idempotent (no backfill UPDATE,
-- since run_migrations re-executes every *.sql on each startup).
ALTER TABLE documents ADD COLUMN IF NOT EXISTS updated_at timestamptz;
