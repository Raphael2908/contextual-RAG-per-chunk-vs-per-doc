# Contextual RAG: per-chunk vs per-document context

A small, reproducible benchmark of two contextual-retrieval strategies, on **cost, speed, and
retrieval accuracy**. Based on Anthropic's
[Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) writeup.

**Contextual retrieval** ([Anthropic, 2024](https://www.anthropic.com/news/contextual-retrieval))
improves RAG recall by prepending a short context note to each chunk before embedding it — so a
clause that reads *"the cap is $2,000,000"* becomes findable as *"the indemnification cap in the
Acme deal is $2,000,000."* The catch: it's **one LLM call per chunk**.

This repo tests a cheaper variant: write **one** richer context block **per document** and reuse it
across that document's chunks (one call per *doc*), and measures the tradeoff against the per-chunk
version and a no-context baseline.

## Results (TL;DR)

![Per-chunk vs per-document contextual retrieval: ingest time, cost per run, and recall@5](results/stats.jpg)

On 10 distinguishable ~8,000-token synthetic contracts with 50 hand-labeled gold queries:

| system        | ingest | cost    | recall@5 | nDCG@5 |
|---------------|--------|---------|----------|--------|
| none          | 8 s    | $0.009  | 0.280    | 0.123  |
| per_chunk     | 147 s  | $0.414  | 0.980    | 0.930  |
| per_document  | 26 s   | $0.138  | 1.000    | 1.000  |

per_document was **~3× cheaper, ~22× fewer LLM tokens, ~5.6× faster**, and matched/edged per_chunk
on accuracy. Full numbers + per-run reports: [`results/`](results/RESULTS.md).

**Caveat worth reading:** the gold queries are anchored on document-level facts (the parties), which
favors the per-document block. See the caveat in [`results/RESULTS.md`](results/RESULTS.md) — the
harder, unrun test is telling near-identical clauses apart *within* one document.

## How it works

Each context mode embeds chunks differently, then lives in its **own** vector store
(`chunks_<model>_<dim>_<mode>`), so the three systems are compared head-to-head:

- `none` — the bare chunk (control).
- `per_chunk` — `[doc · date · summary]` + an LLM-written situating sentence, per chunk
  (`backend/app/contextual.py: enrich_chunk`).
- `per_document` — one longer LLM-written block per document (`build_document_context`), reused on
  every chunk.

Pipeline: extract → chunk → profile (1 LLM pass) → contextual enrich → embed (Voyage) → upsert
(Milvus). See `backend/app/ingest.py`.

## Replicate it

### Option A — keyless wiring check (no API keys, no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e 'backend[dev]'
python backend/scripts/smoke_fake.py   # ingestion pipeline incl. per_document, on fakes
python backend/scripts/smoke_eval.py   # recall/nDCG eval on the labeled samples/ corpus, on fakes
```

These run the whole pipeline + benchmark on a dependency-free fake embedder/LLM (lexical, so recall
is meaningful), proving the harness end to end without keys.

### Option B — the real experiment (Anthropic + Voyage keys)

```bash
cp .env.example .env          # fill RAG_ANTHROPIC_API_KEY and RAG_EMBEDDING_API_KEY (Voyage)
docker compose up -d          # postgres + milvus + api on :8000 (wait for: curl localhost:8000/health)

# 1. Generate the distinguishable, self-labeled corpus (deterministic; markers verified unique):
python backend/scripts/gen_eval_corpus.py --count 10 --target-tokens 8000 \
    --out sample_pdfs_eval --eval-out backend/benchmark/eval_set_8k_eval.jsonl --seed 7

# 2. Ingest each system once → cost/speed report per mode (header auth, no token needed):
for MODE in per_chunk per_document none; do
  python backend/scripts/e2e_bulk.py --api http://localhost:8000 \
      --corpus sample_pdfs_eval --context-mode "$MODE" --no-dedup --mode dense --k 5 \
      --label "eval8k-$MODE" --out results
done   # each prints its batch id — note them for step 3

# 3. Score accuracy against each mode's collection (no re-ingest):
APP_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/company_brain \
RAG_MILVUS_URI=http://localhost:19530 PYTHONPATH=backend \
python -m benchmark.score_existing --batch-id <BATCH_ID> --context-mode <MODE> \
    --eval-set backend/benchmark/eval_set_8k_eval.jsonl
```

Swap models/embedders freely — every model is a LiteLLM string in `.env` (`RAG_CONTEXT_MODEL`,
`RAG_EMBEDDING_MODEL`).

## Methodology

- **Gold set** (`backend/scripts/gen_eval_corpus.py`): each document gets globally-unique facts
  (party pair, matter reference, liability cap, governing state, notice period…) buried in
  collision-free filler. 5 queries per document, each tagged with a `marker` string that is the
  unique answer. The generator self-checks that every marker appears in exactly one document.
- **Relevance** (`benchmark/score_existing.py`): a chunk is relevant if its raw text contains the
  marker. Usually exactly one relevant chunk per query.
- **Metrics** (`benchmark/eval.py`): `recall@k = |top-k ∩ relevant| / |relevant|`, plus MRR and
  nDCG@k, averaged over the 50 queries (k = 5).

## Layout

```
backend/app/         ingest pipeline + providers (Milvus/Voyage/Anthropic via LiteLLM) + FastAPI
backend/benchmark/   eval.py (metrics), score_existing.py (accuracy), matrix.py, eval sets
backend/scripts/     gen_eval_corpus.py (labeled corpus), e2e_bulk.py (ingest driver), smoke_*.py
backend/migrations/  Postgres schema
samples/             3 hand-written labeled contracts (used by the keyless smoke eval)
results/             this experiment's reports + RESULTS.md
```

This is the trimmed ingest + benchmark core of a larger RAG system, extracted to make the experiment
reproducible. The answer-generation, auth, and frontend layers are intentionally not included.

## License

MIT — see [LICENSE](LICENSE).
