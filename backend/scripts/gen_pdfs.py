"""Generate synthetic legal PDFs for the bulk-upload load test.

    python backend/scripts/gen_pdfs.py --count 1000 --out ./sample_pdfs
    python backend/scripts/gen_pdfs.py --count 1000 --dup-rate 0.05 --seed 42

No real PDFs ship in samples/, so this fabricates distinct, text-extractable
legal documents (randomized parties, dates, clause numbers, dollar amounts and
length) — every file has unique content so byte- and content-hash dedup treats
them as separate. `--dup-rate` re-emits a fraction as exact byte duplicates to
exercise the dedup path.

Requires the `reportlab` dev dependency:  pip install -e '.[dev]'
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

PARTIES = [
    "Acme Holdings LLC", "Beta Industries Inc.", "Cygnus Capital Partners",
    "Delta Maritime Corp.", "Evergreen Logistics GmbH", "Falcon Pharma AG",
    "Granite Property Trust", "Helios Energy Co.", "Ironclad Security Ltd.",
    "Juniper Software Inc.", "Kestrel Ventures", "Lumen Diagnostics",
]
DOC_TYPES = [
    "MUTUAL NON-DISCLOSURE AGREEMENT", "MASTER SERVICES AGREEMENT",
    "COMMERCIAL LEASE AGREEMENT", "SHARE PURCHASE AGREEMENT",
    "EMPLOYMENT AGREEMENT", "SOFTWARE LICENSE AGREEMENT",
]
SECTION_TITLES = [
    "DEFINITIONS", "CONFIDENTIALITY", "TERM AND TERMINATION",
    "REPRESENTATIONS AND WARRANTIES", "INDEMNIFICATION",
    "LIMITATION OF LIABILITY", "GOVERNING LAW AND JURISDICTION",
    "DISPUTE RESOLUTION", "ASSIGNMENT", "FORCE MAJEURE",
    "INTELLECTUAL PROPERTY", "PAYMENT TERMS", "NOTICES",
    "ENTIRE AGREEMENT", "SEVERABILITY", "DATA PROTECTION",
    "COMPLIANCE WITH LAWS", "INSURANCE", "AUDIT RIGHTS", "NON-SOLICITATION",
]
CLAUSES = [
    "Confidential Information means any non-public information disclosed by one "
    "party to the other, whether orally or in writing.",
    "The Receiving Party shall hold the Confidential Information in strict "
    "confidence and shall not disclose it to any third party.",
    "This Agreement shall be governed by and construed in accordance with the "
    "laws of the State of Delaware.",
    "Either party may terminate this Agreement upon thirty (30) days' written "
    "notice to the other party.",
    "The aggregate liability of each party under this Agreement shall not exceed "
    "the total fees paid in the preceding twelve (12) months.",
    "The indemnification cap set forth in this Section shall survive termination "
    "of this Agreement for a period of two (2) years.",
    "Neither party shall be liable for any failure to perform due to causes "
    "beyond its reasonable control, including acts of God and war.",
    "Any dispute arising out of this Agreement shall be resolved by binding "
    "arbitration administered under the rules of the AAA.",
]


# Rough token estimate for English prose: ~4 characters per token.
def _est_tokens(lines: list[str]) -> int:
    return sum(len(s) for s in lines) // 4


def _doc_text(rng: random.Random, i: int, target_tokens: int = 0) -> tuple[str, list[str]]:
    a, b = rng.sample(PARTIES, 2)
    dtype = rng.choice(DOC_TYPES)
    date = f"{rng.randint(2018, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    amount = rng.choice([250_000, 500_000, 1_000_000, 2_000_000, 5_000_000])
    title = f"{dtype}  (No. {i:05d})"
    lines = [
        f"This {dtype.title()} (the \"Agreement\") is entered into as of {date}",
        f"by and between {a} (\"Disclosing Party\") and {b} (\"Receiving Party\").",
        "",
        f"Matter Reference: M-{rng.randint(1000, 9999)}-{i:05d}",
        f"Consideration: USD {amount:,}.",
        "",
    ]
    if target_tokens > 0:
        # Realistic legal prose: a handful of numbered SECTIONS, each with a few
        # multi-sentence paragraphs. The chunker treats every numbered line as a
        # section boundary, so the old one-clause-per-line layout exploded an 8k
        # doc into ~256 tiny chunks. Grouping sentences into section bodies yields
        # realistic ~1.5k-char chunks (~20 per 8k-token doc).
        section = 0
        while _est_tokens(lines) < target_tokens:
            section += 1
            lines.append(f"{section}. {rng.choice(SECTION_TITLES)}")
            lines.append("")
            for _ in range(rng.randint(2, 4)):
                para = " ".join(rng.choice(CLAUSES) for _ in range(rng.randint(4, 7)))
                lines.append(para)
                lines.append("")
    else:
        # variable length: 5–14 numbered clauses
        for n in range(1, rng.randint(5, 14) + 1):
            lines.append(f"{n}. {rng.choice(CLAUSES)}")
            lines.append("")
    lines.append(f"IN WITNESS WHEREOF, the parties have executed this Agreement as of {date}.")
    lines.append(f"_________________________        _________________________")
    lines.append(f"{a}                              {b}")
    return title, lines


def _write_pdf(path: Path, title: str, lines: list[str]) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    x, y = inch, height - inch
    c.setFont("Helvetica-Bold", 13)
    c.drawString(x, y, title)
    y -= 0.4 * inch
    c.setFont("Helvetica", 10)
    for line in lines:
        if y < inch:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - inch
        # naive wrap at ~95 chars
        for seg in (_wrap(line, 95) or [""]):
            c.drawString(x, y, seg)
            y -= 0.22 * inch
    c.save()


def _wrap(text: str, width: int) -> list[str]:
    if len(text) <= width:
        return [text]
    out, cur = [], ""
    for word in text.split():
        if len(cur) + len(word) + 1 > width:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100, help="number of unique PDFs")
    ap.add_argument("--out", type=Path, default=Path("./sample_pdfs"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dup-rate", type=float, default=0.0,
                    help="fraction of extra exact-duplicate files to emit (0..1)")
    ap.add_argument("--target-tokens", type=int, default=0,
                    help="approx tokens per doc (~4 chars/token); 0 = random 5–14 clauses")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    written = 0
    tok_total = 0
    for i in range(args.count):
        title, lines = _doc_text(rng, i, args.target_tokens)
        tok_total += _est_tokens(lines)
        _write_pdf(args.out / f"doc_{i:05d}.pdf", title, lines)
        written += 1
        # optionally emit an exact byte-duplicate copy
        if args.dup_rate and rng.random() < args.dup_rate:
            src = (args.out / f"doc_{i:05d}.pdf").read_bytes()
            (args.out / f"doc_{i:05d}_dup.pdf").write_bytes(src)
            written += 1
        if written % 100 == 0:
            print(f"  {written} files written…")

    avg_tok = tok_total // args.count if args.count else 0
    print(f"Wrote {written} PDFs to {args.out} (~{avg_tok} est. tokens/doc)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
