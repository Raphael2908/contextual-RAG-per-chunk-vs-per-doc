"""E2E bulk-test report endpoint: list the JSON reports scripts/e2e_bulk.py writes.

The harness records ingest cost/tokens AND retrieval latency to JSON files under
upload-benchmarks/results/ (cost-bearing, gitignored). The dashboard reads ingest
metrics live from the batch summary, but retrieval latency lives only in these
reports — this serves them so the Bulk tab can show a run's retrieval panel,
matched to the loaded batch by `run.batch_id`.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request

router = APIRouter(tags=["e2e"])


@router.get("/e2e/results")
def results(request: Request):
    results_dir: Path = request.app.state.e2e_results_dir
    if not results_dir.exists():
        return {"results": []}
    reports = []
    for f in sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            reports.append(json.loads(f.read_text()))
        except (OSError, json.JSONDecodeError):
            continue  # skip a half-written or malformed report rather than 500
    return {"results": reports}
