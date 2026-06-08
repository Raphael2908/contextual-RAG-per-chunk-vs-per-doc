-- Per-document AI token breakdown, so batch totals can report the full token
-- spend (LLM input/output for profiling + contextual enrichment, plus embedding
-- tokens) — not just the rolled-up cost_usd. IngestResult carries these at
-- runtime; 003 only persisted chunk_count + total_cost_usd, dropping the rest.
-- Defaults to 0, so failed/duplicate docs (written via set_status) read as 0.

ALTER TABLE documents ADD COLUMN IF NOT EXISTS llm_input_tokens  bigint NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS llm_output_tokens bigint NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS embed_tokens      bigint NOT NULL DEFAULT 0;
