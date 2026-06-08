-- Minimal ingestion + benchmark slice schema.
-- Single implicit tenant, company-level "open" store. No orgs/matters/RLS/auth
-- (deliberately dropped vs architecture.md for this slice — see the plan).

CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

CREATE TABLE IF NOT EXISTS documents (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name          text        NOT NULL,
    file_type     text        NOT NULL,
    storage_path  text        NOT NULL,
    file_hash     text        NOT NULL,             -- sha256 of raw bytes
    content_hash  text,                             -- normalized-content hash
    summary       text,
    effective_date date,                            -- "correct-as-of" date
    size          bigint      NOT NULL DEFAULT 0,
    status        text        NOT NULL DEFAULT 'queued',
                  -- queued | processing | ready | failed | duplicate
    error         text,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_file_hash_idx ON documents (file_hash);

CREATE TABLE IF NOT EXISTS chunks (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index     int  NOT NULL,
    source_location text,                           -- e.g. "heading > section"
    text            text NOT NULL,                  -- raw chunk text
    context_text    text NOT NULL,                  -- the text actually embedded (context block + chunk)
    token_count     int  NOT NULL DEFAULT 0,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);

-- The benchmark's core output: one row per (document × LLM × embedding model) run.
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id      uuid REFERENCES documents (id) ON DELETE SET NULL,
    document_name    text NOT NULL,
    label            text NOT NULL,                 -- human label for the combo
    llm_model        text,                          -- NULL = no-context control
    embedding_model  text NOT NULL,
    embedding_dim    int  NOT NULL,
    with_context     boolean NOT NULL,
    cache_active     boolean NOT NULL DEFAULT false,
    chunk_count      int  NOT NULL DEFAULT 0,
    stages           jsonb NOT NULL DEFAULT '[]'::jsonb,  -- per-stage latency + tokens + cost
    total_latency_ms double precision NOT NULL DEFAULT 0,
    total_cost_usd   numeric(12, 6) NOT NULL DEFAULT 0,
    llm_input_tokens   bigint NOT NULL DEFAULT 0,
    llm_output_tokens  bigint NOT NULL DEFAULT 0,
    embed_tokens       bigint NOT NULL DEFAULT 0,
    created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS benchmark_runs_created_idx ON benchmark_runs (created_at DESC);
