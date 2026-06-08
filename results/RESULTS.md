# Results

Per-chunk vs per-document contextual retrieval, over **10 distinguishable ~8,000-token
synthetic contracts** with **50 hand-labeled gold queries** (5 per document). Dense
retrieval, Voyage `voyage-law-2` embeddings, Anthropic `claude-haiku-4-5` writing the
context. Raw per-run reports are the `eval8k-*.json` / `.md` files in this folder.

## Cost & speed (ingesting all 10 documents)

| system        | ingest wall | cost     | LLM input tokens | chunks |
|---------------|-------------|----------|------------------|--------|
| per_chunk     | 147 s       | $0.4137  | 1,899,965        | 268    |
| per_document  | 26 s        | $0.1378  | 85,252           | 268    |
| none          | 8 s         | $0.0095  | 0                | 268    |

per_document vs per_chunk: **~3× cheaper, ~22× fewer LLM input tokens, ~5.6× faster** —
one context call per document instead of one per chunk.

## Accuracy (recall@5 / MRR / nDCG@5, averaged over 50 gold queries)

| system        | recall@5 | MRR   | nDCG@5 |
|---------------|----------|-------|--------|
| none          | 0.280    | 0.073 | 0.123  |
| per_chunk     | 0.980    | 0.913 | 0.930  |
| per_document  | 1.000    | 1.000 | 1.000  |

## How to read this

- **Contextual retrieval clearly matters**: bare chunks (`none`) only hit 0.28 recall, because
  a clause like *"the cap is $2,000,000"* never names the parties the query asks about.
- **per_document matched/edged per_chunk here, far cheaper.** Its shared block carries the
  parties on every chunk, which is exactly what these queries need.

## Honest caveat

Every gold query is anchored on a **document-level** fact (which parties signed it), and the
per-document block stamps that on every chunk — so this corpus favors per_document. It shows
per_document is *at least as good* at document-level disambiguation, not that it is strictly
better everywhere. The harder, unrun test is a document with **several near-identical clauses**,
where you must tell chunks *within the same document* apart — that is where a shared block could
blur things and per-chunk context could pull ahead.
