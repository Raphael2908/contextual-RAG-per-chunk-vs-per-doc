"""The contextualizer LLM, behind a Protocol with a real + fake impl.

Real impl routes through LiteLLM, so any model on the market (Anthropic, OpenAI,
Gemini, or anything via `openrouter/...`) is just a config string. The parent
document is sent as a prompt-cached block so per-chunk enrichment is cheap on
providers that support caching.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.litellm_util import cost_of, normalize_model, usage_of


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    cache_active: bool = False


@runtime_checkable
class LLM(Protocol):
    model: str

    def complete(
        self,
        *,
        instruction: str,
        cached_context: str | None = None,
        user: str,
        max_tokens: int = 512,
    ) -> LLMResult: ...


class LiteLLMContextualizer:
    """LiteLLM-backed contextualizer. `cached_context` (the full document) is
    marked `cache_control: ephemeral` so repeated per-chunk calls reuse it."""

    def __init__(self, model: str, *, temperature: float = 0.0):
        self.model = normalize_model(model)
        self.temperature = temperature

    def complete(
        self,
        *,
        instruction: str,
        cached_context: str | None = None,
        user: str,
        max_tokens: int = 512,
    ) -> LLMResult:
        import litellm

        system_blocks: list[dict] = [{"type": "text", "text": instruction}]
        if cached_context:
            system_blocks.append(
                {
                    "type": "text",
                    "text": cached_context,
                    "cache_control": {"type": "ephemeral"},
                }
            )
        messages = [
            {"role": "system", "content": system_blocks},
            {"role": "user", "content": user},
        ]
        resp = litellm.completion(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        u = usage_of(resp)
        cache_creation = u.get("cache_creation_tokens", 0)
        cache_read = u.get("cache_read_tokens", 0)
        return LLMResult(
            text=text,
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            cost_usd=cost_of(resp),
            cache_active=bool(cache_creation or cache_read),
        )


class FakeLLM:
    """Deterministic, dependency-free. Returns a short, stable 'situating
    sentence' derived from the input — no network, no cost."""

    def __init__(self, model: str = "fake/echo"):
        self.model = model

    def complete(
        self,
        *,
        instruction: str,
        cached_context: str | None = None,
        user: str,
        max_tokens: int = 512,
    ) -> LLMResult:
        digest = hashlib.sha256(user.encode("utf-8")).hexdigest()[:8]
        head = " ".join(user.split()[:12])
        return LLMResult(
            text=f"This passage ({digest}) concerns: {head}.",
            input_tokens=len(user.split()),
            output_tokens=12,
            cost_usd=0.0,
            cache_active=cached_context is not None,
        )
