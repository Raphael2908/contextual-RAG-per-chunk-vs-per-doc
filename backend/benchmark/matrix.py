"""The benchmark matrix — which (LLM × embedding) combos to sweep.

Every model is a LiteLLM model string, so you can drop in ANY model on the
market — Anthropic, OpenAI, Gemini, Cohere, or anything via `openrouter/...`.
Edit this list freely; each new model just needs its provider API key in the env.

`llm_model = None` is the no-context control: bare chunks, no LLM enrichment —
the baseline that isolates exactly what contextual chunking costs.

`context_mode` is the third axis: "per_chunk" (one LLM call/chunk → a situating
sentence) vs "per_document" (one longer LLM call/doc, reused across every chunk,
carrying the doc id + filename). A combo with `llm_model=None` is always the
no-context control regardless of `context_mode`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Combo:
    label: str
    llm_model: str | None      # None = no-context control
    embedding_model: str
    embedding_dim: int | None = None  # let the provider report it when None
    context_mode: str = "per_chunk"   # "per_chunk" | "per_document" (ignored if llm_model is None)


# Seeded defaults. The first is the architecture's baseline (Haiku + voyage-law-2).
DEFAULT_MATRIX: list[Combo] = [
    Combo(
        label="haiku + voyage-law-2",
        llm_model="anthropic/claude-haiku-4-5",
        embedding_model="voyage/voyage-law-2",
        embedding_dim=1024,
    ),
    # Same models as the baseline but ONE longer context block per document — the
    # A/B for "does per-document context retrieve as well as per-chunk, far cheaper?".
    Combo(
        label="haiku per-doc + voyage-law-2",
        llm_model="anthropic/claude-haiku-4-5",
        embedding_model="voyage/voyage-law-2",
        embedding_dim=1024,
        context_mode="per_document",
    ),
    Combo(
        label="sonnet + voyage-law-2",
        llm_model="anthropic/claude-sonnet-4-6",
        embedding_model="voyage/voyage-law-2",
        embedding_dim=1024,
    ),
    Combo(
        label="haiku + voyage-3-large",
        llm_model="anthropic/claude-haiku-4-5",
        embedding_model="voyage/voyage-3-large",
        embedding_dim=1024,
    ),
    Combo(
        label="no-context + voyage-law-2 (control)",
        llm_model=None,
        embedding_model="voyage/voyage-law-2",
        embedding_dim=1024,
    ),
    # Examples of swapping in other vendors (uncomment + set keys to use):
    # Combo("gpt-4o-mini + openai-3-large", "openai/gpt-4o-mini",
    #       "openai/text-embedding-3-large", 3072),
    # Combo("openrouter gemini-flash + voyage-3", "openrouter/google/gemini-2.5-flash",
    #       "voyage/voyage-3-large", 1024),
]


# A keyless matrix for `APP_USE_FAKES=1` smoke runs — proves the harness wiring
# end to end with no network or credentials.
FAKE_MATRIX: list[Combo] = [
    Combo("fake-haiku + fake-embed", "fake/haiku", "fake/hash-1024", 1024),
    Combo("fake-haiku per-doc + fake-embed", "fake/haiku", "fake/hash-1024", 1024,
          context_mode="per_document"),
    Combo("no-context + fake-embed (control)", None, "fake/hash-1024", 1024),
]
