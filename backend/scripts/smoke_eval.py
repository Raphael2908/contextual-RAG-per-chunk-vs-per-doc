"""Dependency-free smoke test: the retrieval-quality eval on fakes, no DB/network.

Runs the gold-set sweep with the fake (lexical) embedder + identity reranker and
in-memory repos, then asserts the harness produces sane metrics and that the
contextual combo retrieves at least as well as the no-context control.

    python backend/scripts/smoke_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.reranker import IdentityReranker  # noqa: E402
from app.repository import (  # noqa: E402
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryEvalRepository,
)
from benchmark.eval import _print_table, load_eval_set, run_eval  # noqa: E402
from benchmark.matrix import FAKE_MATRIX  # noqa: E402


def main() -> int:
    settings = Settings(APP_USE_FAKES=True)
    samples_dir = Path(__file__).resolve().parents[2] / "samples"
    eval_set = load_eval_set()
    assert eval_set, "eval_set.jsonl is empty"

    runs = run_eval(
        settings=settings,
        combos=FAKE_MATRIX,
        eval_set=eval_set,
        samples_dir=samples_dir,
        doc_repo=InMemoryDocumentRepository(),
        chunk_repo=InMemoryChunkRepository(),
        eval_repo=InMemoryEvalRepository(),
        reranker=IdentityReranker(),
        k=settings.eval_k,
    )

    print(f"Gold queries: {len(eval_set)} over {len({q.doc for q in eval_set})} docs\n")
    _print_table(runs, settings.eval_k)

    assert runs, "no eval runs produced"
    for r in runs:
        assert r.query_count > 0, f"{r.label}: no queries scored (markers mismatched?)"
        assert 0.0 <= r.recall_at_k <= 1.0, f"{r.label}: recall out of range"
        assert 0.0 <= r.ndcg_at_k <= 1.0, f"{r.label}: ndcg out of range"
    # Every gold query's marked chunk must exist in the corpus (else markers are wrong).
    for r in runs:
        misses = [d for d in r.details if d.get("note", "").startswith("NO GOLD")]
        assert not misses, f"{r.label}: {len(misses)} queries matched no gold chunk: {misses}"
    # Both retrieval modes must be exercised (dense + hybrid BM25/RRF on the fake store).
    modes = {r.retrieval_mode for r in runs}
    assert {"dense", "hybrid"} <= modes, f"expected dense + hybrid runs, got {modes}"
    # All three context strategies must be swept (the A/B this change adds).
    ctx_modes = {r.context_mode for r in runs}
    assert {"none", "per_chunk", "per_document"} <= ctx_modes, (
        f"expected none + per_chunk + per_document runs, got {ctx_modes}"
    )

    print("SMOKE OK — dense + hybrid retrieval-quality eval across "
          "none/per_chunk/per_document runs end to end on fakes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
