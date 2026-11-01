import sys
from pathlib import Path
from types import SimpleNamespace

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from components.usage import TokenUsage, usage_cost, usage_from_anthropic, usage_from_openai


def test_openai_usage_separates_cached_prompt_tokens():
    usage = SimpleNamespace(
        prompt_tokens=1200,
        completion_tokens=300,
        prompt_tokens_details=SimpleNamespace(cached_tokens=1000),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=120),
    )

    normalized = usage_from_openai(usage)

    assert normalized == TokenUsage(
        input_tokens=200, cached_input_tokens=1000, output_tokens=300, reasoning_tokens=120
    )
    assert normalized.total_tokens == 1500


def test_openai_usage_tolerates_missing_details():
    assert usage_from_openai(SimpleNamespace(prompt_tokens=10, completion_tokens=2)) == TokenUsage(
        input_tokens=10, output_tokens=2
    )
    assert usage_from_openai(None) is None
    assert usage_from_anthropic(None) is None


def test_anthropic_usage_keeps_cache_reads_and_writes_separate():
    usage = SimpleNamespace(
        input_tokens=50,
        cache_read_input_tokens=800,
        cache_creation_input_tokens=0,
        output_tokens=400,
        output_tokens_details=SimpleNamespace(thinking_tokens=250),
    )

    assert usage_from_anthropic(usage) == TokenUsage(
        input_tokens=50, cached_input_tokens=800, output_tokens=400, reasoning_tokens=250
    )


def test_cost_prices_each_token_bucket_at_its_own_rate():
    usage = TokenUsage(
        input_tokens=1_000_000,
        cached_input_tokens=1_000_000,
        cache_write_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    # claude-sonnet-5-5: $2 input, $0.10 cache read, $2.50 cache write, $10 output
    assert usage_cost("anthropic", "claude-sonnet-5-5", usage) == pytest.approx(14.60)


def test_openai_cache_writes_bill_at_input_rate():
    usage = TokenUsage(cache_write_tokens=1_000_000)

    assert usage_cost("openai", "gpt-5-mini", usage) == pytest.approx(0.25)


def test_unpriced_model_has_unknown_cost():
    assert usage_cost("openai", "not-a-model", TokenUsage(input_tokens=1)) is None


def test_usage_adds_bucketwise():
    total = TokenUsage(input_tokens=1, output_tokens=2) + TokenUsage(cached_input_tokens=3, reasoning_tokens=4)

    assert total == TokenUsage(input_tokens=1, cached_input_tokens=3, output_tokens=2, reasoning_tokens=4)


if __name__ == "__main__":
    from tests.utils import pytest_this_file

    raise SystemExit(pytest_this_file(__file__))
