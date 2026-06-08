"""Per-stage instrumentation — latency, tokens, and dollar cost.

This is what makes the benchmark a benchmark: every ingestion stage is timed and
its model usage/cost recorded, so combos can be compared on speed and cost.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Iterator


@dataclass
class StageMetric:
    stage: str
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@contextmanager
def timed(stage: str, sink: list[StageMetric]) -> Iterator[StageMetric]:
    """Time a block and append a StageMetric to `sink`.

    Yield the metric so the caller can fold in token/cost numbers from a model
    response before the block exits.
    """
    metric = StageMetric(stage=stage)
    start = time.perf_counter()
    try:
        yield metric
    finally:
        metric.latency_ms = (time.perf_counter() - start) * 1000.0
        sink.append(metric)
