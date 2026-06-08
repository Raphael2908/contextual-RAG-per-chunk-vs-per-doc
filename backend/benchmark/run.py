"""CLI: sweep the benchmark matrix over one or more documents.

    uv run python -m benchmark.run path/to/contract.pdf [more.pdf ...]
    APP_USE_FAKES=1 uv run python -m benchmark.run sample.txt   # no keys/network

Prints a comparison table (latency, tokens, $/doc, $/1k-chunks, cache) and writes
a benchmark_runs row per (document × combo).
"""

from __future__ import annotations

import sys
from pathlib import Path

from app import extract
from app.config import get_settings
from app.db import make_pool, run_migrations
from app.repository import (
    PostgresBenchmarkRepository,
    PostgresChunkRepository,
    PostgresDocumentRepository,
)
from benchmark.matrix import DEFAULT_MATRIX, FAKE_MATRIX
from benchmark.runner import run_matrix


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    settings = get_settings()
    settings.export_provider_keys()

    pool = make_pool(settings.database_url)
    run_migrations(pool)
    doc_repo = PostgresDocumentRepository(pool)
    chunk_repo = PostgresChunkRepository(pool)
    bench_repo = PostgresBenchmarkRepository(pool)

    matrix = FAKE_MATRIX if settings.use_fakes else DEFAULT_MATRIX
    print(f"Sweeping {len(matrix)} combos over {len(argv)} document(s) "
          f"(use_fakes={settings.use_fakes})\n")

    all_runs = []
    for path_str in argv:
        path = Path(path_str)
        data = path.read_bytes()
        file_type = extract.file_type_of(path.name)
        runs = run_matrix(
            settings=settings, combos=matrix, data=data, name=path.name,
            file_type=file_type, doc_repo=doc_repo, chunk_repo=chunk_repo,
            bench_repo=bench_repo,
        )
        _print_table(path.name, runs)
        all_runs.extend(runs)

    pool.close()
    return 0


def _print_table(doc_name: str, runs) -> None:
    print(f"=== {doc_name} ===")
    header = (
        f"{'combo':<34} {'chunks':>6} {'latency_s':>10} {'$/doc':>10} "
        f"{'$/1k-chunks':>12} {'llm_tok':>9} {'embed_tok':>10} {'cache':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in runs:
        per_1k = (r.total_cost_usd / r.chunk_count * 1000) if r.chunk_count else 0.0
        print(
            f"{r.label:<34} {r.chunk_count:>6} "
            f"{r.total_latency_ms / 1000:>10.2f} "
            f"{r.total_cost_usd:>10.5f} {per_1k:>12.5f} "
            f"{r.llm_input_tokens + r.llm_output_tokens:>9} "
            f"{r.embed_tokens:>10} {('yes' if r.cache_active else 'no'):>6}"
        )
    print()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
