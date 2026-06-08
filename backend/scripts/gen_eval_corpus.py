"""Generate a *distinguishable, labeled* legal-PDF corpus for the retrieval-quality
benchmark — and emit its gold eval set.

Unlike scripts/gen_pdfs.py (load-test corpus: every doc's body is the SAME 8 boilerplate
clauses, so only header fields are unique → not labelable), every document here carries
**globally-unique, queryable facts** derived from its index, buried in neutral filler that
*cannot* collide with a marker. That makes recall@k / nDCG@k meaningful: each gold query
has exactly one relevant document.

    python backend/scripts/gen_eval_corpus.py --count 10 --target-tokens 8000 \
        --out sample_pdfs_eval --eval-out backend/benchmark/eval_set_8k_eval.jsonl --seed 7

Why this design tests the real question: every query is anchored by the doc's unique party
pair ("…agreement between {A} and {B}"). The fact-clause itself (e.g. the liability cap)
does NOT name the parties — so the bare chunk (no-context) loses the party anchor, while
per_chunk / per_document inject it via the context block. The corpus therefore probes
whether contextual retrieval disambiguates *which document's* clause to return, and whether
per_document's shared block still lets the exact target clause win over its sibling chunks.

Requires the `reportlab` dev dependency (reused from gen_pdfs):  pip install -e '.[dev]'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Reuse the PDF renderer + helpers — don't reimplement.
from gen_pdfs import _est_tokens, _write_pdf  # noqa: E402  (scripts/ on sys.path via __main__)

# 40 distinct, state-name-free company names → unique party pair per doc (2i, 2i+1).
NAMES = [
    "Acme Holdings LLC", "Beacon Industries Inc.", "Cygnus Capital Partners",
    "Delta Maritime Corp.", "Evergreen Logistics GmbH", "Falcon Pharma AG",
    "Granite Property Trust", "Helios Energy Co.", "Ironclad Security Ltd.",
    "Juniper Software Inc.", "Kestrel Ventures LLC", "Lumen Diagnostics Inc.",
    "Meridian Robotics Corp.", "Northwind Freight Co.", "Onyx Materials Ltd.",
    "Pinnacle Aerospace Inc.", "Quartz Analytics LLC", "Radian Biotech AG",
    "Summit Telecom Co.", "Tessera Foods Inc.", "Umbra Optics Ltd.",
    "Vanguard Mining Corp.", "Willow Health Inc.", "Xenon Devices LLC",
    "Yarrow Agritech Co.", "Zephyr Mobility Inc.", "Atlas Forge Ltd.",
    "Borealis Media Corp.", "Citrine Finance LLC", "Dynamo Power Co.",
    "Equinox Retail Inc.", "Fathom Marine Ltd.", "Gossamer Textiles Co.",
    "Harbor Lighting Inc.", "Indigo Press LLC", "Juno Aviation Corp.",
    "Kraken Subsea Ltd.", "Lattice Semiconductors Inc.", "Mosaic Ceramics Co.",
    "Nimbus Cloud Systems Inc.",
]

US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
]

CODENAMES = [
    "Aurora", "Basilisk", "Cobalt", "Daybreak", "Ember", "Falconry", "Gravity",
    "Harvest", "Ivory", "Juniper", "Keystone", "Lantern", "Monsoon", "Nimbus",
    "Obsidian", "Pioneer", "Quill", "Redwood", "Sapphire", "Tundra",
]

DOC_TYPES = [
    "MASTER SERVICES AGREEMENT", "MUTUAL NON-DISCLOSURE AGREEMENT",
    "SHARE PURCHASE AGREEMENT", "COMMERCIAL LEASE AGREEMENT",
    "SOFTWARE LICENSE AGREEMENT", "SUPPLY AGREEMENT",
]

# Neutral filler: NO digits, NO state names, NO "$", NO "N days|years" — so a filler
# paragraph can never contain (and thus collide with) any unique marker.
FILLER = [
    "The parties acknowledge that they have negotiated this Agreement in good faith and at arm's length.",
    "This Agreement constitutes the entire understanding between the parties with respect to its subject matter.",
    "If any provision of this Agreement is held unenforceable, the remaining provisions shall continue in full force.",
    "The failure of either party to enforce any right shall not be deemed a waiver of that right.",
    "Each party represents that it has full corporate authority to enter into and perform this Agreement.",
    "The headings in this Agreement are for convenience only and shall not affect its interpretation.",
    "This Agreement may be executed in counterparts, each of which shall be deemed an original instrument.",
    "The parties shall cooperate in good faith to give full effect to the purpose of this Agreement.",
    "Any notice required under this Agreement shall be given in writing to the receiving party's registered address.",
    "The recitals set forth above are incorporated into and made a part of this Agreement.",
    "This Agreement is binding upon and inures to the benefit of the parties and their permitted successors and assigns.",
    "Each party shall bear its own costs and expenses incurred in connection with the negotiation of this Agreement.",
    "Nothing in this Agreement shall be construed to create a partnership, joint venture, or agency between the parties.",
    "The parties agree that the terms of this Agreement are confidential and proprietary to the disclosing party.",
]


def _money(n: int) -> str:
    return f"${n:,}"


def _doc_facts(i: int) -> dict:
    a, b = NAMES[2 * i], NAMES[2 * i + 1]
    return {
        "a": a,
        "b": b,
        "matter_ref": f"M-{1000 + i}-{i:05d}",
        "state": US_STATES[i],
        "cap": _money(1_000_000 + i * 317_000),
        "notice": f"{15 + i * 7} days",
        "survival": f"{2 + i} years",
        "codename": f"Project {CODENAMES[i]}",
    }


def _fact_sections(f: dict) -> list[tuple[str, str]]:
    """(section title, body) for the 5 unique, queryable facts. Each body contains
    exactly one marker value and is semantically matched to its gold query."""
    return [
        ("GOVERNING LAW",
         f"This Agreement shall be governed by and construed in accordance with the laws of the "
         f"State of {f['state']}, without regard to its conflict-of-laws principles."),
        ("LIMITATION OF LIABILITY",
         f"The aggregate liability of each party under this Agreement shall not exceed {f['cap']} "
         f"in the aggregate, regardless of the form of action."),
        ("TERM AND TERMINATION",
         f"Either party may terminate this Agreement for convenience upon {f['notice']} prior "
         f"written notice to the other party."),
        ("CONFIDENTIALITY",
         f"The confidentiality obligations of the Receiving Party shall survive termination of "
         f"this Agreement for a period of {f['survival']}."),
        ("MATTER REFERENCE",
         f"This engagement is recorded under matter reference {f['matter_ref']} and is known "
         f"internally as {f['codename']}."),
    ]


def _gold_queries(fname: str, f: dict) -> list[dict]:
    a, b = f["a"], f["b"]
    pair = f"the agreement between {a} and {b}"
    return [
        {"query": f"Which state's law governs {pair}?", "doc": fname,
         "markers": [f"State of {f['state']}"], "note": f"governing law ({a[:12]}…)"},
        {"query": f"What is the limitation of liability cap in {pair}?", "doc": fname,
         "markers": [f["cap"]], "note": "liability cap"},
        {"query": f"How much prior notice is required to terminate {pair} for convenience?",
         "doc": fname, "markers": [f["notice"]], "note": "termination notice"},
        {"query": f"How long do the confidentiality obligations survive under {pair}?",
         "doc": fname, "markers": [f["survival"]], "note": "confidentiality survival"},
        {"query": f"What is the matter reference for {pair}?", "doc": fname,
         "markers": [f["matter_ref"]], "note": "matter reference"},
    ]


def _doc_lines(rng_seed: int, i: int, f: dict, target_tokens: int) -> tuple[str, list[str]]:
    import random

    rng = random.Random(rng_seed + i)
    dtype = DOC_TYPES[i % len(DOC_TYPES)]
    title = f"{dtype}  (No. {i:05d})"
    lines = [
        f"This {dtype.title()} (the \"Agreement\") is entered into",
        f"by and between {f['a']} (\"Party A\") and {f['b']} (\"Party B\").",
        "",
    ]
    # Interleave the 5 fact sections with neutral filler sections so the markers are
    # buried (≈20 chunks/doc), in a deterministic but shuffled order per doc.
    fact_secs = _fact_sections(f)
    rng.shuffle(fact_secs)
    n = 0
    fi = 0
    # Emit a fact section roughly every few filler sections, then top up with filler
    # until the token target is met.
    while fi < len(fact_secs) or _est_tokens(lines) < target_tokens:
        n += 1
        if fi < len(fact_secs) and (n % 3 == 0):
            title_sec, body = fact_secs[fi]
            fi += 1
            lines += [f"{n}. {title_sec}", "", body, ""]
        else:
            n_paras = rng.randint(2, 4)
            lines.append(f"{n}. GENERAL PROVISIONS")
            lines.append("")
            for _ in range(n_paras):
                lines.append(" ".join(rng.choice(FILLER) for _ in range(rng.randint(4, 7))))
                lines.append("")
            if fi >= len(fact_secs) and _est_tokens(lines) >= target_tokens:
                break
    lines += [
        "IN WITNESS WHEREOF, the parties have executed this Agreement.",
        "_________________________        _________________________",
        f"{f['a']}                              {f['b']}",
    ]
    return title, lines


def _self_check(texts: dict[str, str], gold: list[dict]) -> None:
    """Every gold marker must appear in its target doc and in NO other doc."""
    problems = []
    for q in gold:
        target = q["doc"]
        for m in q["markers"]:
            ml = m.lower()
            if ml not in texts[target].lower():
                problems.append(f"marker {m!r} NOT in target {target}")
            collisions = [d for d, t in texts.items() if d != target and ml in t.lower()]
            if collisions:
                problems.append(f"marker {m!r} ({target}) also in {collisions}")
    if problems:
        raise SystemExit("GOLD SET NOT CLEAN:\n  " + "\n  ".join(problems))


def main() -> int:
    ap = argparse.ArgumentParser(description="Distinguishable labeled eval corpus + gold set.")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--target-tokens", type=int, default=8000, dest="target_tokens")
    ap.add_argument("--out", type=Path, default=Path("./sample_pdfs_eval"))
    ap.add_argument("--eval-out", type=Path, default=Path("backend/benchmark/eval_set_8k_eval.jsonl"),
                    dest="eval_out")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.count > len(NAMES) // 2:
        raise SystemExit(f"--count {args.count} exceeds {len(NAMES) // 2} unique party pairs")
    if args.count > len(US_STATES) or args.count > len(CODENAMES):
        raise SystemExit(f"--count {args.count} exceeds unique states/codenames pool")

    args.out.mkdir(parents=True, exist_ok=True)
    texts: dict[str, str] = {}
    gold: list[dict] = []
    tok_total = 0
    for i in range(args.count):
        f = _doc_facts(i)
        fname = f"doc_{i:05d}.pdf"
        title, lines = _doc_lines(args.seed, i, f, args.target_tokens)
        texts[fname] = "\n".join(lines)
        tok_total += _est_tokens(lines)
        _write_pdf(args.out / fname, title, lines)
        gold += _gold_queries(fname, f)

    _self_check(texts, gold)

    args.eval_out.parent.mkdir(parents=True, exist_ok=True)
    with args.eval_out.open("w") as fh:
        fh.write("# Gold eval set for the distinguishable 8k corpus (gen_eval_corpus.py).\n")
        fh.write("# Every marker is unique to one document (self-checked at generation).\n")
        for q in gold:
            fh.write(json.dumps(q) + "\n")

    avg = tok_total // args.count if args.count else 0
    print(f"Wrote {args.count} PDFs to {args.out} (~{avg} est. tokens/doc), "
          f"{len(gold)} gold queries to {args.eval_out}")
    print("GOLD SET CLEAN — every marker is unique to its target document.")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import gen_pdfs`
    raise SystemExit(main())
