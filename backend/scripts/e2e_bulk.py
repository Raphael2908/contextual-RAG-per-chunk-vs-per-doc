"""End-to-end bulk-upload test driver — over HTTP, against a live API.

    python backend/scripts/e2e_bulk.py --gen --count 100 --label e2e-100-real

Unlike scripts/smoke_bulk.py (in-process, fakes), this drives the REAL running
API over HTTP, so it exercises the full stack with real keys (APP_USE_FAKES=0):
Postgres, Milvus, Anthropic (profiling + contextual enrichment), Voyage
(embeddings). It measures the metrics that matter for the company brain:

  - cost in USD (ingest, per-doc + total) and the grand total incl. retrieval
  - wall-clock time for upload and for ingest (decoupled server-side)
  - retrieval latency (p50/p95) measured client-side
  - full AI token breakdown: LLM input/output tokens + embedding tokens

Flow: health ping -> (optionally) generate a synthetic PDF corpus via
scripts/gen_pdfs.py -> create a batch -> stream files in with bounded concurrency
(timed) -> poll GET /batches/{id} until terminal -> pull batch totals -> run a
fixed dense query set (timed) -> write a JSON + Markdown report.

Requires the `httpx` dev dependency:  pip install -e 'backend[dev]'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

# Dense queries drawn from the synthetic corpus vocabulary (see scripts/gen_pdfs.py:
# party names, document types, and clause phrasings) so they actually retrieve.
QUERY_SET = [
    "What is the confidentiality obligation of the receiving party?",
    "definition of confidential information",
    "governing law State of Delaware",
    "termination on thirty days written notice",
    "limitation of liability cap on fees paid",
    "indemnification cap survives termination for two years",
    "force majeure acts of God and war",
    "disputes resolved by binding arbitration AAA rules",
    "what is the consideration amount in USD",
    "matter reference number for the agreement",
    "mutual non-disclosure agreement between the parties",
    "master services agreement payment terms",
    "commercial lease agreement base rent",
    "share purchase agreement closing",
    "employment agreement obligations",
    "software license agreement scope",
    "Acme Holdings disclosing party",
    "effective date of the agreement",
]

TERMINAL_STATUSES = ("ready", "duplicate", "failed")


def _pct(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (no numpy). p in [0,100]."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (p / 100) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _gen_corpus(out: Path, count: int, seed: int, dup_rate: float) -> None:
    """Reuse scripts/gen_pdfs.py to fabricate the corpus (don't reimplement it)."""
    gen = Path(__file__).resolve().parent / "gen_pdfs.py"
    cmd = [
        sys.executable, str(gen),
        "--count", str(count), "--out", str(out), "--seed", str(seed),
        "--dup-rate", str(dup_rate),
    ]
    print(f"Generating corpus: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


async def _upload_one(
    client: httpx.AsyncClient, base: str, batch_id: str, path: Path,
    sem: asyncio.Semaphore, dedup: bool = True,
) -> dict:
    async with sem:
        data = path.read_bytes()
        files = {"file": (path.name, data, "application/pdf")}
        r = await client.post(
            f"{base}/batches/{batch_id}/documents", files=files,
            params={"dedup": str(dedup).lower()},
        )
        r.raise_for_status()
        return r.json()


async def _run(args: argparse.Namespace) -> int:
    base = args.api.rstrip("/")
    corpus = Path(args.corpus)

    if args.gen:
        _gen_corpus(corpus, args.count, args.seed, args.dup_rate)
    files = sorted(corpus.glob("*.pdf"))
    if not files:
        print(f"No PDFs in {corpus} (use --gen to fabricate a corpus).", file=sys.stderr)
        return 2
    print(f"Corpus: {len(files)} PDFs from {corpus}")

    # In proxy auth mode the core requires the shared service token + an attested
    # identity on every call; in header mode both are ignored. Sourced from env so
    # the script stays keyless by default.
    auth_headers: dict[str, str] = {"X-User-Email": os.environ.get("E2E_USER_EMAIL", "e2e@local")}
    if os.environ.get("APP_SERVICE_TOKEN"):
        auth_headers["X-Service-Token"] = os.environ["APP_SERVICE_TOKEN"]

    timeout = httpx.Timeout(args.http_timeout, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, headers=auth_headers) as client:
        # 0. Health ping — fail fast if the API is down.
        try:
            h = await client.get(f"{base}/health")
            h.raise_for_status()
        except Exception as e:  # noqa: BLE001
            print(f"API not reachable at {base}/health: {e}", file=sys.stderr)
            return 2

        # 1. Create batch.
        r = await client.post(
            f"{base}/batches", json={"label": args.label, "context_mode": args.context_mode}
        )
        r.raise_for_status()
        batch_id = r.json()["batch_id"]
        print(f"Batch {batch_id}: uploading {len(files)} files "
              f"(upload concurrency {args.upload_concurrency})")

        # 2. Upload with bounded concurrency — timed. Each POST returns immediately
        #    (queued/duplicate); ingest runs async on the server worker pool.
        sem = asyncio.Semaphore(args.upload_concurrency)
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(_upload_one(client, base, batch_id, p, sem, dedup=not args.no_dedup) for p in files)
        )
        upload_wall_s = time.perf_counter() - t0
        accepted = sum(1 for x in results if x.get("status") == "queued")
        dup_on_upload = sum(1 for x in results if x.get("status") == "duplicate")
        print(f"  uploaded in {upload_wall_s:.1f}s "
              f"({accepted} queued, {dup_on_upload} duplicate)")

        # 3. Poll until terminal (ready + duplicate + failed == total).
        print("  ingesting (polling)…")
        while True:
            r = await client.get(f"{base}/batches/{batch_id}", params={"summary_only": True})
            r.raise_for_status()
            s = r.json()
            counts = s["counts"]
            total = s["total"]
            done = sum(counts.get(k, 0) for k in TERMINAL_STATUSES)
            elapsed = time.perf_counter() - t0
            print(f"    {done}/{total} done  counts={counts}  ({elapsed:.0f}s)")
            if done >= total:
                break
            if elapsed > args.timeout:
                print(f"  TIMEOUT after {elapsed:.0f}s "
                      f"(>{args.timeout}s); reporting partial state.", file=sys.stderr)
                break
            await asyncio.sleep(args.poll_interval)
        t_terminal = time.perf_counter()
        ingest_total_wall_s = t_terminal - t0
        ingest_only_wall_s = t_terminal - (t0 + upload_wall_s)

        # 4. Final summary incl. per-doc list (failures + token totals).
        r = await client.get(f"{base}/batches/{batch_id}")
        r.raise_for_status()
        summary = r.json()
        counts = summary["counts"]
        totals = summary["totals"]
        ready = counts.get("ready", 0)
        failures = [
            {"id": d["id"], "name": d["name"], "error": d.get("error")}
            for d in summary.get("docs", []) if d["status"] == "failed"
        ]

        # 5. Retrieval — dense query set, latency timed client-side.
        print(f"Retrieval: {len(QUERY_SET)} dense queries (k={args.k})")
        per_query = []
        for q in QUERY_SET:
            tq = time.perf_counter()
            r = await client.post(
                f"{base}/query",
                json={"query": q, "k": args.k, "mode": args.mode, "rerank": False},
            )
            latency_ms = (time.perf_counter() - tq) * 1000
            r.raise_for_status()
            qr = r.json()
            usage = qr.get("usage", {})
            per_query.append({
                "query": q,
                "latency_ms": round(latency_ms, 2),
                "hits": len(qr.get("results", [])),
                "embed_tokens": usage.get("query_embed_tokens", 0),
                "embed_cost_usd": usage.get("query_embed_cost_usd", 0.0),
            })

    # --- Assemble the report -------------------------------------------------
    latencies = [q["latency_ms"] for q in per_query]
    cost_total = float(totals.get("total_cost_usd", 0.0))
    retr_cost = sum(q["embed_cost_usd"] for q in per_query)
    report = {
        "run": {
            "label": args.label,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "api": base,
            "batch_id": batch_id,
            "mode": args.mode,
            "context_mode": args.context_mode,
            "files_in_corpus": len(files),
            "dup_rate": args.dup_rate,
            "upload_concurrency": args.upload_concurrency,
            "k": args.k,
        },
        "ingest": {
            "total_in_batch": summary["total"],
            "counts": counts,
            "ready": ready,
            "upload_wall_s": round(upload_wall_s, 2),
            "ingest_only_wall_s": round(ingest_only_wall_s, 2),
            "ingest_total_wall_s": round(ingest_total_wall_s, 2),
            "throughput_docs_per_s": round(ready / ingest_total_wall_s, 3)
            if ingest_total_wall_s > 0 else 0.0,
            "chunk_count_total": totals.get("chunk_count", 0),
            "cost": {
                "total_usd": round(cost_total, 6),
                "avg_usd_per_doc": round(cost_total / ready, 6) if ready else 0.0,
            },
            "tokens": {
                "llm_input_tokens": totals.get("llm_input_tokens", 0),
                "llm_output_tokens": totals.get("llm_output_tokens", 0),
                "embed_tokens": totals.get("embed_tokens", 0),
                "avg_llm_input_per_doc": round(totals.get("llm_input_tokens", 0) / ready, 1)
                if ready else 0.0,
                "avg_llm_output_per_doc": round(totals.get("llm_output_tokens", 0) / ready, 1)
                if ready else 0.0,
                "avg_embed_per_doc": round(totals.get("embed_tokens", 0) / ready, 1)
                if ready else 0.0,
            },
            "failures": failures,
        },
        "retrieval": {
            "query_count": len(per_query),
            "latency_ms": {
                "p50": round(_pct(latencies, 50), 2),
                "p95": round(_pct(latencies, 95), 2),
                "min": round(min(latencies), 2) if latencies else 0.0,
                "max": round(max(latencies), 2) if latencies else 0.0,
                "mean": round(statistics.mean(latencies), 2) if latencies else 0.0,
            },
            "avg_hits": round(statistics.mean(q["hits"] for q in per_query), 2)
            if per_query else 0.0,
            "query_embed_tokens_total": sum(q["embed_tokens"] for q in per_query),
            "query_embed_cost_usd_total": round(retr_cost, 8),
            "per_query": per_query,
        },
        "grand_total_cost_usd": round(cost_total + retr_cost, 6),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    json_path = out_dir / f"{args.label}-{stamp}.json"
    md_path = out_dir / f"{args.label}-{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_render_md(report))

    print(_render_md(report))
    print(f"\nReport written:\n  {json_path}\n  {md_path}")
    return 0


def _render_md(r: dict) -> str:
    run, ing, ret = r["run"], r["ingest"], r["retrieval"]
    cost, tok, lat = ing["cost"], ing["tokens"], ret["latency_ms"]
    lines = [
        f"# E2E bulk test — {run['label']}",
        "",
        f"- **API**: {run['api']}  ·  **batch**: `{run['batch_id']}`  ·  "
        f"**mode**: {run['mode']}  ·  **context**: {run.get('context_mode', 'per_chunk')}  ·  "
        f"**k**: {run['k']}",
        f"- **Run (UTC)**: {run['timestamp_utc']}  ·  "
        f"**corpus**: {run['files_in_corpus']} files  ·  "
        f"**dup-rate**: {run['dup_rate']}  ·  "
        f"**upload concurrency**: {run['upload_concurrency']}",
        "",
        "## Ingest",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| total in batch | {ing['total_in_batch']} |",
        f"| counts | {ing['counts']} |",
        f"| ready | {ing['ready']} |",
        f"| upload wall | {ing['upload_wall_s']} s |",
        f"| ingest-only wall | {ing['ingest_only_wall_s']} s |",
        f"| ingest total wall | {ing['ingest_total_wall_s']} s |",
        f"| throughput | {ing['throughput_docs_per_s']} docs/s |",
        f"| chunks total | {ing['chunk_count_total']} |",
        f"| **cost total** | **${cost['total_usd']}** |",
        f"| cost / doc | ${cost['avg_usd_per_doc']} |",
        f"| LLM input tokens | {tok['llm_input_tokens']:,} ({tok['avg_llm_input_per_doc']}/doc) |",
        f"| LLM output tokens | {tok['llm_output_tokens']:,} ({tok['avg_llm_output_per_doc']}/doc) |",
        f"| embed tokens | {tok['embed_tokens']:,} ({tok['avg_embed_per_doc']}/doc) |",
        f"| failures | {len(ing['failures'])} |",
        "",
        "## Retrieval",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| queries | {ret['query_count']} |",
        f"| latency p50 / p95 | {lat['p50']} / {lat['p95']} ms |",
        f"| latency min / max / mean | {lat['min']} / {lat['max']} / {lat['mean']} ms |",
        f"| avg hits | {ret['avg_hits']} |",
        f"| query embed tokens | {ret['query_embed_tokens_total']:,} |",
        f"| query embed cost | ${ret['query_embed_cost_usd_total']} |",
        "",
        f"## Grand total cost: ${r['grand_total_cost_usd']}",
        "",
    ]
    if ing["failures"]:
        lines += ["### Failures", ""]
        lines += [f"- `{f['name']}`: {f['error']}" for f in ing["failures"]]
        lines += [""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="E2E bulk-upload test over HTTP.")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--corpus", default="./sample_pdfs", help="dir of PDFs to upload")
    ap.add_argument("--gen", action="store_true", help="(re)generate corpus via gen_pdfs.py")
    ap.add_argument("--count", type=int, default=100, help="PDFs to generate with --gen")
    ap.add_argument("--dup-rate", type=float, default=0.0, dest="dup_rate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--upload-concurrency", type=int, default=8, dest="upload_concurrency")
    ap.add_argument("--poll-interval", type=float, default=2.0, dest="poll_interval")
    ap.add_argument("--mode", default="dense", choices=["dense", "hybrid"])
    ap.add_argument("--context-mode", default="per_chunk", dest="context_mode",
                    choices=["none", "per_chunk", "per_document"],
                    help="contextual-retrieval strategy for this batch")
    ap.add_argument("--no-dedup", action="store_true", dest="no_dedup",
                    help="skip byte-hash dedup so identical bytes can be re-ingested "
                         "under a different context_mode (benchmark A/B)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--label", default="e2e")
    ap.add_argument("--out", default="./upload-benchmarks/results")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="max seconds to wait for ingest to finish")
    ap.add_argument("--http-timeout", type=float, default=120.0, dest="http_timeout")
    args = ap.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
