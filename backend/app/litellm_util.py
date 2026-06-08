"""Shared helpers for talking to LiteLLM and extracting usage/cost uniformly."""

from __future__ import annotations

from typing import Any

# Bare-name -> provider-prefixed, so `.env.example` values (`claude-haiku-4-5`,
# `voyage-law-2`) work alongside LiteLLM's canonical `anthropic/...`, `voyage/...`.
_PROVIDER_PREFIXES: list[tuple[str, str]] = [
    ("claude", "anthropic/"),
    ("voyage", "voyage/"),
    ("text-embedding", "openai/"),
    ("gpt", "openai/"),
    ("gemini", "gemini/"),
]


def normalize_model(model: str) -> str:
    """Return a LiteLLM-resolvable model string, inferring a provider if missing."""
    if "/" in model:  # already provider-qualified (incl. openrouter/...)
        return model
    for needle, prefix in _PROVIDER_PREFIXES:
        if model.startswith(needle):
            return prefix + model
    return model


def cost_of(response: Any) -> float:
    """Best-effort dollar cost of a LiteLLM response, 0.0 if unpriced."""
    hidden = getattr(response, "_hidden_params", None) or {}
    cost = hidden.get("response_cost")
    if cost is not None:
        return float(cost)
    try:  # fall back to recomputing from the model's price sheet
        from litellm import completion_cost

        return float(completion_cost(completion_response=response) or 0.0)
    except Exception:
        return 0.0


def usage_of(response: Any) -> dict[str, int]:
    """Normalize a LiteLLM usage object into a flat int dict."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    def _g(name: str) -> int:
        val = getattr(usage, name, None)
        if val is None and isinstance(usage, dict):
            val = usage.get(name)
        return int(val or 0)

    return {
        "input_tokens": _g("prompt_tokens") or _g("input_tokens"),
        "output_tokens": _g("completion_tokens") or _g("output_tokens"),
        "total_tokens": _g("total_tokens"),
        "cache_read_tokens": _g("cache_read_input_tokens"),
        "cache_creation_tokens": _g("cache_creation_input_tokens"),
    }
