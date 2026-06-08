# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reproducible benchmark comparing three contextual-retrieval strategies (`none`, `per_chunk`, `per_document`) on cost, speed, and retrieval accuracy. The experiment is based on Anthropic's [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) writeup. It is the trimmed ingest + benchmark core extracted from a larger RAG system; answer-generation, full auth, and the frontend are intentionally absent. See `README.md` for the experiment writeup and `results/RESULTS.md` for numbers.

## Commands

All Python lives under `backend/`. The package is installed editable as `company-brain-backend` and exposes two import roots: `app` and `benchmark`.

```bash
# Install (editable, with dev extras)
pip install -e 'backend[dev]'

# Keyless wiring checks — full pipeline + eval on dependency-free fakes (no DB, no network, no keys)
python backend/scripts/smoke_fake.py    # ingestion incl. per_document
python backend/scripts/smoke_eval.py    # recall/nDCG eval on samples/ corpus

# Real stack (needs API keys in .env + Docker)
docker compose up -d                     # postgres + milvus (+ etcd/minio) + api on :8000
curl localhost:8000/health               # wait for {"status":"ok"}

# Benchmark sweep over documents (matrix in benchmark/matrix.py)
PYTHONPATH=backend python -m benchmark.run path/to/doc.pdf
APP_USE_FAKES=1 PYTHONPATH=backend python -m benchmark.run sample.txt   # no keys

# Score an already-ingested batch's accuracy (ingest-once; no re-ingest)
APP_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/company_brain \
RAG_MILVUS_URI=http://localhost:19530 PYTHONPATH=backend \
python -m benchmark.score_existing --batch-id <ID> --context-mode <MODE> \
    --eval-set backend/benchmark/eval_set_8k_eval.jsonl
```

The full 3-mode experiment (corpus generation → bulk ingest per mode → scoring) is documented step-by-step in `README.md` ("Replicate it — Option B"). `pytest` is a declared dev dependency but there is no test suite in this slice; the `scripts/smoke_*.py` files are the executable correctness checks.

## Architecture

**The pipeline is mode-parameterized, not mode-branched.** One `IngestionService.ingest()` (in `app/ingest.py`) runs every context mode; `context_mode ∈ {none, per_chunk, per_document}` selects behavior at the contextual-enrichment step. The eight pipeline stages (byte-hash dedup → blob → extract → content-hash → profile → chunk → contextual enrich → embed+upsert) are each timed and costed via `app/metrics.py`, which is the entire point — the benchmark *is* this instrumentation.

**The three context modes** (`app/contextual.py`) all embed text built from the same template `[prefix]\n<context>\n\n<chunk text>`:
- `none` — bare chunk, no LLM (the `llm=None` control).
- `per_chunk` — one LLM call **per chunk** writing a situating sentence (the default; expensive).
- `per_document` — one LLM call **per document** writing a ~100–150 word block, reused verbatim across every chunk (`build_document_context` + `append_document_context`). The doc-level LLM cost is attributed once, not per chunk.

Both LLM modes prompt-cache the full document; that caching is what makes them affordable at scale.

**Each mode lives in its own vector collection.** `collection_for(model, dim, context_mode=, hybrid=)` in `app/vectorstore.py` produces `chunks_<model>_<dim>_<mode>[_hybrid]`, so the three systems never commingle vectors and can be queried/scored head-to-head from a single ingest run. `score_existing.py` relies on this — it queries the per-mode collection directly instead of re-ingesting.

**Real/fake providers swap in one place.** `app/factory.py` is the only composition point: `make_llm`/`make_embedder`/`make_vector_store`/`make_blob`/`make_reranker` return real (LiteLLM/Milvus/filesystem) or fake (in-memory, lexical) implementations based on `settings.use_fakes`. The fakes are lexical so recall metrics stay meaningful without keys. The **same** `IngestionService` and `RetrievalService` back both the FastAPI app (`app/main.py`) and the benchmark CLIs — injection differs, code does not.

**Models are LiteLLM strings everywhere.** Any model on the market (Anthropic, OpenAI, Gemini, Cohere, `openrouter/...`) works by editing `.env` (`RAG_CONTEXT_MODEL`, `RAG_EMBEDDING_MODEL`, `RAG_RERANK_MODEL`) or `benchmark/matrix.py`. `app/litellm_util.normalize_model` accepts both provider-prefixed (`anthropic/claude-haiku-4-5`) and bare (`claude-haiku-4-5`) forms. Provider keys are read from `RAG_*` env names and mirrored into the `ANTHROPIC_API_KEY`/`VOYAGE_API_KEY`/etc. that LiteLLM expects via `Settings.export_provider_keys()`.

**Config** is `app/config.py` (`pydantic-settings`, env-driven, `.env`). Note the legacy `with_context` boolean still resolves through `resolve_context_mode` (False→`none`, True→`per_chunk`) for back-compat.

**Bulk ingest** decouples fast upload from slow LLM/embedding work: the upload endpoint (`routers/batches.py`) does byte-hash + blob + row creation, then an in-process async worker pool (`app/worker.py`, no Celery/Redis) calls `process_existing()` (pipeline steps 3–8) with bounded concurrency. Both `ingest()` and `process_existing()` mark a document `failed` and return a failed result rather than raising, so one poison doc never kills a worker.

**Eval methodology** (`benchmark/eval.py`, `benchmark/score_existing.py`): the gold corpus generator (`scripts/gen_eval_corpus.py`) buries a globally-unique `marker` string in each document and self-verifies uniqueness; a chunk is relevant iff its raw text contains the marker. Metrics are `recall@k`, `MRR`, `nDCG@k`. Be aware of the documented caveat: gold queries anchor on document-level facts, which favors `per_document`.

## Layout

```
backend/app/         ingest + read pipeline, providers (LLM/Embedder/VectorStore/Reranker), FastAPI, config, metrics
backend/benchmark/   eval.py (metrics), score_existing.py (accuracy), matrix.py (sweep axes), run.py (CLI), eval sets
backend/scripts/     gen_eval_corpus.py, e2e_bulk.py (ingest driver), smoke_*.py (keyless checks)
backend/migrations/  Postgres schema (run automatically on app/CLI startup via app/db.run_migrations)
samples/             3 hand-written labeled contracts (keyless smoke eval corpus)
results/             this experiment's reports + RESULTS.md
```

Migrations in `backend/migrations/` are applied automatically by `run_migrations()` at startup of the API and the scoring CLIs — no separate migrate step.
