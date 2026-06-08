-- Records which contextual-retrieval strategy an eval run used, so the quality
-- sweep can compare per_chunk vs per_document vs the no-context control directly
-- (not just via the free-text label). See backend/benchmark/eval.py + contextual.py.
-- `with_context` (migration 002) stays as the coarse boolean; this is the precise axis.

ALTER TABLE eval_runs
    ADD COLUMN IF NOT EXISTS context_mode text NOT NULL DEFAULT 'per_chunk';
