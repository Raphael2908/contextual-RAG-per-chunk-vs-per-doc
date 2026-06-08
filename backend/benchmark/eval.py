"""Retrieval-quality benchmark — the quality axis the cost/speed benchmark lacks.

For each (LLM × embedding × context-on/off) combo, ingest the gold corpus into an
isolated collection, run every gold query through the read path (embed query →
dense search → optional rerank), and score the ranking against the labelled
relevant chunks with **recall@k, MRR, nDCG@k**. Persist one `eval_runs` row per
combo so quality can be joined with `benchmark_runs` cost/latency into
quality-per-dollar — answering "does Voyage + Haiku + context actually retrieve
better than the no-context control (or another embedder)?"

Runnable on `APP_USE_FAKES=1` end to end (fake embedder carries a lexical signal,
identity reranker) so the harness is verifiable without keys; run live for real
numbers.

    APP_USE_FAKES=1 PYTHONPATH=backend python -m benchmark.eval   # fakes, needs Postgres
    PYTHONPATH=backend python -m benchmark.eval                   # live (set keys)

Hybrid (dense + sparse/BM25) retrieval is the documented next step (architecture.md
§"Search layer: Milvus only"); `retrieval_mode` is recorded as "dense" until it lands.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _ws(s: str) -> str:
    """Collapse runs of whitespace to single spaces (line-wrap-insensitive match)."""
    return re.sub(r"\s+", " ", s).strip()

from app import extract, factory
from app.config import Settings, get_settings
from app.ingest import IngestionService
from app.repository import EvalRun
from app.reranker import Reranker
from app.retrieve import RetrievalService
from app.vectorstore import collection_for
from benchmark.matrix import Combo, DEFAULT_MATRIX, FAKE_MATRIX

# Repo-root samples/ holds the gold corpus; eval_set.jsonl sits next to this file.
_SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples"
_EVAL_SET = Path(__file__).resolve().parent / "eval_set.jsonl"


# --------------------------------------------------------------------------- #
# Gold set
# --------------------------------------------------------------------------- #
@dataclass
class EvalQuery:
    query: str
    doc: str                       # filename in samples/
    markers: list[str]             # a chunk is relevant if its text contains any
    note: str = ""

    def is_relevant(self, chunk_text: str) -> bool:
        # Normalize whitespace so a marker still matches across the source's line
        # wraps (e.g. "five (5)\nyears" vs the marker's "five (5) years").
        low = _ws(chunk_text.lower())
        return any(_ws(m.lower()) in low for m in self.markers)


def load_eval_set(path: Path = _EVAL_SET) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        d = json.loads(line)
        queries.append(
            EvalQuery(
                query=d["query"], doc=d["doc"],
                markers=d.get("markers") or d.get("answer_contains") or [],
                note=d.get("note", ""),
            )
        )
    return queries


# --------------------------------------------------------------------------- #
# Metrics (binary relevance)
# --------------------------------------------------------------------------- #
def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    for i, cid in enumerate(ranked[:k]):
        if cid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(ranked[:k]) if cid in relevant)
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg else 0.0


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
@dataclass
class _IngestedDoc:
    doc_id: str
    # chunk_id -> raw chunk text, for relevance labelling + rerank text resolution
    chunks: dict[str, str] = field(default_factory=dict)


def _combo_mode(combo: Combo) -> str:
    """Effective context mode: the control (no LLM) is always 'none'."""
    return "none" if combo.llm_model is None else combo.context_mode


def _ingest_corpus(
    *, service: IngestionService, combo: Combo, docs: list[str], samples_dir: Path,
    chunk_repo,
) -> tuple[dict[str, _IngestedDoc], str]:
    """Ingest every gold doc through the combo into its (isolated) collection.
    Returns {filename: _IngestedDoc} and the collection name they share."""
    by_doc: dict[str, _IngestedDoc] = {}
    collection = ""
    mode = _combo_mode(combo)
    for fname in docs:
        path = samples_dir / fname
        data = path.read_bytes()
        res = service.ingest(
            data, fname, extract.file_type_of(fname),
            context_mode=mode,
            enforce_dedup=False, persist_chunks=True,
        )
        if res.status != "ready":
            raise RuntimeError(f"ingest of {fname} failed: {res.error or res.status}")
        collection = res.collection
        rows = chunk_repo.list_for_document(res.document_id)
        by_doc[fname] = _IngestedDoc(
            doc_id=res.document_id,
            chunks={r.chunk_id: r.text for r in rows},
        )
    return by_doc, collection


def _score_combo(
    *, settings: Settings, combo: Combo, suffix: str, eval_set: list[EvalQuery],
    samples_dir: Path, doc_repo, chunk_repo, reranker: Reranker, k: int,
    reranked: bool, mode: str, eval_set_name: str,
) -> EvalRun:
    hybrid = mode == "hybrid"
    store = factory.make_vector_store(settings)
    embedder = factory.make_embedder(settings, combo.embedding_model, dim=combo.embedding_dim)
    llm = factory.make_llm(settings, combo.llm_model)

    # Idempotent re-runs: drop this combo+mode's eval collection before re-ingesting.
    expected = (
        collection_for(
            embedder.model, combo.embedding_dim or settings.dense_dim,
            context_mode=_combo_mode(combo), hybrid=hybrid,
        )
        + suffix
    )
    store.drop(expected)

    service = IngestionService(
        embedder=embedder, store=store, blob=factory.make_blob(settings),
        doc_repo=doc_repo, chunk_repo=chunk_repo, llm=llm, collection_suffix=suffix,
        hybrid=hybrid,
    )
    docs = sorted({q.doc for q in eval_set})
    by_doc, collection = _ingest_corpus(
        service=service, combo=combo, docs=docs, samples_dir=samples_dir, chunk_repo=chunk_repo,
    )

    # One combined map across the corpus so rerank can resolve any candidate's text.
    all_chunks: dict[str, str] = {}
    for d in by_doc.values():
        all_chunks.update(d.chunks)

    retrieval = RetrievalService(
        embedder=embedder, store=store, reranker=reranker,
        hybrid=hybrid, rrf_k=settings.rrf_k,
    )

    recalls, mrrs, ndcgs, details = [], [], [], []
    for q in eval_set:
        doc = by_doc.get(q.doc)
        if doc is None:
            continue
        relevant = {cid for cid, txt in doc.chunks.items() if q.is_relevant(txt)}
        if not relevant:
            details.append({"query": q.query, "doc": q.doc, "note": "NO GOLD CHUNK MATCHED — check markers"})
            continue
        retrieved = retrieval.search(
            collection, q.query, k, rerank=reranked,
            candidate_k=settings.rerank_candidates,
            text_resolver=lambda ids: [all_chunks.get(i) for i in ids],
        )
        ranked = [r.chunk_id for r in retrieved.hits]
        r_at_k = recall_at_k(ranked, relevant, k)
        m = mrr_at_k(ranked, relevant, k)
        n = ndcg_at_k(ranked, relevant, k)
        recalls.append(r_at_k); mrrs.append(m); ndcgs.append(n)
        details.append({
            "query": q.query, "doc": q.doc, "note": q.note,
            "recall": round(r_at_k, 4), "mrr": round(m, 4), "ndcg": round(n, 4),
            "relevant": len(relevant), "top": ranked[:k],
        })

    n_scored = len(recalls)
    ctx_mode = _combo_mode(combo)
    label = combo.label + (f" [{mode} · {ctx_mode}]") + (" +rerank" if reranked else "")
    return EvalRun(
        eval_set=eval_set_name, label=label, llm_model=combo.llm_model,
        embedding_model=combo.embedding_model,
        embedding_dim=combo.embedding_dim or settings.dense_dim,
        with_context=ctx_mode != "none", context_mode=ctx_mode, retrieval_mode=mode,
        reranked=reranked, rerank_model=(reranker.name if reranked else None), k=k,
        query_count=n_scored,
        recall_at_k=(sum(recalls) / n_scored) if n_scored else 0.0,
        mrr=(sum(mrrs) / n_scored) if n_scored else 0.0,
        ndcg_at_k=(sum(ndcgs) / n_scored) if n_scored else 0.0,
        details=details,
    )


def run_eval(
    *, settings: Settings, combos: list[Combo], eval_set: list[EvalQuery],
    samples_dir: Path, doc_repo, chunk_repo, eval_repo, reranker: Reranker,
    k: int | None = None, eval_set_name: str = "samples", modes: list[str] | None = None,
) -> list[EvalRun]:
    k = k or settings.eval_k
    # Sweep retrieval mode (dense vs hybrid) as an axis — the A/B for "does
    # sparse/BM25 recover exact terms dense misses?". A real reranker adds a
    # second (reranked) variant per combo; the identity fake is a no-op.
    modes = modes or ["dense", "hybrid"]
    rerank_variants = [False] if reranker.name == "identity" else [False, True]
    runs: list[EvalRun] = []
    for i, combo in enumerate(combos):
        for mode in modes:
            for reranked in rerank_variants:
                try:
                    run = _score_combo(
                        settings=settings, combo=combo, suffix=f"_eval_{i}",
                        eval_set=eval_set, samples_dir=samples_dir, doc_repo=doc_repo,
                        chunk_repo=chunk_repo, reranker=reranker, k=k, reranked=reranked,
                        mode=mode, eval_set_name=eval_set_name,
                    )
                except Exception as exc:  # noqa: BLE001 — one mode failing (e.g. hybrid on a
                    # pre-2.5 Milvus) must not abort the whole sweep.
                    print(
                        f"  ! skipped {combo.label} [{mode}]"
                        f"{' +rerank' if reranked else ''}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                if eval_repo is not None:
                    run.id = eval_repo.insert(run)
                runs.append(run)
    return runs


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_table(runs: list[EvalRun], k: int) -> None:
    header = (
        f"{'combo':<52} {'mode':>7} {'context':>13} {f'recall@{k}':>10} {'mrr':>7} "
        f"{f'ndcg@{k}':>9} {'queries':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in runs:
        print(
            f"{r.label:<52} {r.retrieval_mode:>7} {r.context_mode:>13} "
            f"{r.recall_at_k:>10.3f} {r.mrr:>7.3f} {r.ndcg_at_k:>9.3f} {r.query_count:>8}"
        )
    print()


def main(argv: list[str]) -> int:
    from app.db import make_pool, run_migrations
    from app.repository import (
        PostgresChunkRepository,
        PostgresDocumentRepository,
        PostgresEvalRepository,
    )

    settings = get_settings()
    settings.export_provider_keys()

    eval_set = load_eval_set()
    if not eval_set:
        print("eval_set.jsonl is empty", file=sys.stderr)
        return 2

    pool = make_pool(settings.database_url)
    run_migrations(pool)
    doc_repo = PostgresDocumentRepository(pool)
    chunk_repo = PostgresChunkRepository(pool)
    eval_repo = PostgresEvalRepository(pool)
    reranker = factory.make_reranker(settings)

    combos = FAKE_MATRIX if settings.use_fakes else DEFAULT_MATRIX
    k = settings.eval_k
    print(
        f"Scoring {len(combos)} combo(s) over {len(eval_set)} gold queries "
        f"(use_fakes={settings.use_fakes}, reranker={reranker.name}, k={k})\n"
    )
    runs = run_eval(
        settings=settings, combos=combos, eval_set=eval_set, samples_dir=_SAMPLES_DIR,
        doc_repo=doc_repo, chunk_repo=chunk_repo, eval_repo=eval_repo,
        reranker=reranker, k=k,
    )
    _print_table(runs, k)
    pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
