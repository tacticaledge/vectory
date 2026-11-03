"""Provider-neutral token usage and cost accounting for completed LLM calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from components.model_catalog import get_model_pricing


@dataclass(frozen=True)
class TokenUsage:
    """Billed token counts for one call, normalized so each bucket has one price.

    ``input_tokens`` excludes cache reads and cache writes. ``output_tokens``
    includes reasoning/thinking tokens; ``reasoning_tokens`` is informational.
    """

    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cached_input_tokens + self.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


def _count(source: Any, name: str) -> int:
    value = getattr(source, name, None) if source is not None else None
    return value if isinstance(value, int) else 0


def _reported(usage: Any, *names: str) -> bool:
    return usage is not None and all(isinstance(getattr(usage, name, None), int) for name in names)


def usage_from_openai(usage: Any) -> TokenUsage | None:
    """Normalize Chat Completions usage, where prompt_tokens includes cached tokens.

    Returns None when the response did not report its token counts, so the cost stays
    unknown rather than $0. Absent detail fields (cached, reasoning) count as zero.
    """
    if not _reported(usage, "prompt_tokens", "completion_tokens"):
        return None
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    cached = _count(prompt_details, "cached_tokens")
    return TokenUsage(
        input_tokens=max(_count(usage, "prompt_tokens") - cached, 0),
        cached_input_tokens=cached,
        output_tokens=_count(usage, "completion_tokens"),
        reasoning_tokens=_count(completion_details, "reasoning_tokens"),
    )


def usage_from_anthropic(usage: Any) -> TokenUsage | None:
    """Normalize Messages usage, where input_tokens already excludes cache traffic."""
    if not _reported(usage, "input_tokens", "output_tokens"):
        return None
    return TokenUsage(
        input_tokens=_count(usage, "input_tokens"),
        cached_input_tokens=_count(usage, "cache_read_input_tokens"),
        cache_write_tokens=_count(usage, "cache_creation_input_tokens"),
        output_tokens=_count(usage, "output_tokens"),
        reasoning_tokens=_count(getattr(usage, "output_tokens_details", None), "thinking_tokens"),
    )


def usage_cost(provider: str, model: str, usage: TokenUsage) -> float | None:
    """Return the USD list-price cost of one usage record, or None when the model is unpriced."""
    pricing = get_model_pricing(provider, model)
    if pricing is None:
        return None
    cached_rate = pricing.get("cached_input")
    write_rate = pricing.get("cache_write")
    per_million = (
        usage.input_tokens * pricing["input"]
        + usage.cached_input_tokens * (pricing["input"] if cached_rate is None else cached_rate)
        + usage.cache_write_tokens * (pricing["input"] if write_rate is None else write_rate)
        + usage.output_tokens * pricing["output"]
    )
    return per_million / 1_000_000
