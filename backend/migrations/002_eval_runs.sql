-- Retrieval-quality benchmark output: one row per (eval set × combo [× rerank]).
-- Complements benchmark_runs (cost/speed) with a quality axis — recall@k / MRR /
-- nDCG@k — so combos can be ranked on what they actually retrieve, joined with
-- cost/latency into quality-per-dollar. See backend/benchmark/eval.py.

CREATE TABLE IF NOT EXISTS eval_runs (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_set        text NOT NULL,                 -- gold set / corpus name
    label           text NOT NULL,                 -- combo label (+ "rerank" variant)
    llm_model       text,                          -- NULL = no-context control
    embedding_model text NOT NULL,
    embedding_dim   int  NOT NULL,
    with_context    boolean NOT NULL,
    retrieval_mode  text NOT NULL DEFAULT 'dense', -- dense | hybrid (hybrid deferred)
    reranked        boolean NOT NULL DEFAULT false,
    rerank_model    text,
    k               int  NOT NULL,                 -- recall@k / nDCG@k cutoff
    query_count     int  NOT NULL DEFAULT 0,
    recall_at_k     double precision NOT NULL DEFAULT 0,
    mrr             double precision NOT NULL DEFAULT 0,
    ndcg_at_k       double precision NOT NULL DEFAULT 0,
    details         jsonb NOT NULL DEFAULT '[]'::jsonb,  -- per-query breakdown
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS eval_runs_created_idx ON eval_runs (created_at DESC);
